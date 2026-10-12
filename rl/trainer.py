"""Framework-independent PPO training loop.

The agent supplies the network operations (act, value, predict,
train_step, save). Everything else - collecting rollouts, bootstrapping
truncated episodes, advantages, minibatches, evaluation, checkpoints
and logs - is identical for every framework.

    python -m rl.trainer --framework torch --tracks train --run-name torch_multi
"""

import argparse
import csv
import dataclasses
import os
import time

import numpy as np

from rl.common import (
    PPOConfig,
    RolloutBuffer,
    VecEnv,
    compute_gae,
    minibatch_indices,
)
from rl.evaluate import evaluate_policy


def make_agent(framework, config):
    if framework == "torch":
        from rl.ppo_torch import TorchPPOAgent

        return TorchPPOAgent(config)

    raise ValueError(f"Unknown framework: {framework}")


MODEL_EXTENSIONS = {
    "pytorch": ".pt",
}


class CsvLog:
    def __init__(self, path):
        self.path = path
        self.fields = None

    def write(self, row):
        new_file = self.fields is None

        if new_file:
            self.fields = list(row)

        with open(self.path, "a" if not new_file else "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)

            if new_file:
                writer.writeheader()

            writer.writerow(row)


def train(agent, config, run_dir, log=print):
    """Train ``agent`` with PPO; returns the best worst-track fitness."""
    os.makedirs(run_dir, exist_ok=True)
    config.save(os.path.join(run_dir, "config.json"))

    extension = MODEL_EXTENSIONS[agent.framework]
    best_path = os.path.join(run_dir, "best" + extension)
    final_path = os.path.join(run_dir, "final" + extension)

    progress_log = CsvLog(os.path.join(run_dir, "progress.csv"))
    eval_log = CsvLog(os.path.join(run_dir, "eval.csv"))

    rng = np.random.default_rng(config.seed)

    vec_env = VecEnv(
        config.tracks,
        config.num_envs,
        seed=config.seed,
        max_steps=config.max_steps
    )

    buffer = RolloutBuffer(
        config.num_steps,
        config.num_envs,
        vec_env.observation_size,
        vec_env.action_size
    )

    observations = vec_env.reset()

    global_step = 0
    best_score = -np.inf
    start_time = time.perf_counter()

    for update in range(1, config.num_updates + 1):
        if config.anneal_lr:
            fraction = 1.0 - (update - 1) / config.num_updates
            learning_rate = config.learning_rate * fraction
        else:
            learning_rate = config.learning_rate

        # --------------------------------------------------
        # COLLECT ROLLOUT
        # --------------------------------------------------

        buffer.reset()
        episodes = []

        for _ in range(config.num_steps):
            actions, log_probs, values = agent.act(observations)

            (
                next_observations,
                rewards,
                terminated,
                truncated,
                final_observations,
                finished,
            ) = vec_env.step(actions)

            # A truncated episode did not really end: bootstrap with the
            # value of the state the time limit cut off.
            for index in np.flatnonzero(truncated):
                final_value = agent.value(final_observations[index][None, :])
                rewards[index] += config.gamma * final_value[0]

            buffer.add(
                observations,
                actions,
                log_probs,
                rewards,
                terminated | truncated,
                values
            )

            observations = next_observations
            episodes += finished
            global_step += config.num_envs

        # --------------------------------------------------
        # ADVANTAGES
        # --------------------------------------------------

        advantages, returns = compute_gae(
            buffer.rewards,
            buffer.values,
            buffer.dones,
            agent.value(observations),
            config.gamma,
            config.gae_lambda
        )

        batch = buffer.flattened(advantages, returns)

        # --------------------------------------------------
        # OPTIMIZE
        # --------------------------------------------------

        stats = []

        for _ in range(config.update_epochs):
            for indices in minibatch_indices(
                config.batch_size,
                config.minibatch_size,
                rng
            ):
                minibatch = {key: value[indices] for key, value in batch.items()}

                if config.normalize_advantages:
                    adv = minibatch["advantages"]
                    minibatch["advantages"] = (
                        (adv - adv.mean()) / (adv.std() + 1e-8)
                    ).astype(np.float32)

                stats.append(agent.train_step(minibatch, learning_rate))

        # --------------------------------------------------
        # LOG
        # --------------------------------------------------

        elapsed = time.perf_counter() - start_time

        row = {
            "update": update,
            "global_step": global_step,
            "time": round(elapsed, 1),
            "steps_per_second": round(global_step / elapsed),
            "learning_rate": learning_rate,
            "episodes": len(episodes),
            "episode_return": _mean(e["return"] for e in episodes),
            "episode_laps": _mean(e["laps"] for e in episodes),
            "episode_checkpoints": _mean(e["checkpoints"] for e in episodes),
            "crash_rate": _mean(e["crashed"] for e in episodes),
        }

        for key in stats[0]:
            row[key] = float(np.mean([s[key] for s in stats]))

        progress_log.write(row)

        log(
            f"update {update:4d}/{config.num_updates} "
            f"step {global_step:>9,} "
            f"return {row['episode_return']:8.2f} "
            f"laps {row['episode_laps']:5.2f} "
            f"crash {row['crash_rate']:5.0%} "
            f"kl {row['approx_kl']:.4f} "
            f"{row['steps_per_second']:,} steps/s"
        )

        # --------------------------------------------------
        # EVALUATE
        # --------------------------------------------------

        if update % config.eval_interval == 0 or update == config.num_updates:
            results = evaluate_policy(
                agent.as_policy(),
                tracks=config.tracks,
                episodes=config.eval_episodes,
                max_steps=config.max_steps
            )

            for track_name, metrics in results.items():
                eval_log.write({
                    "update": update,
                    "global_step": global_step,
                    "track": track_name,
                    **metrics,
                })

            # Maximin, like the multi-track neuroevolution fitness.
            score = min(m["fitness"] for m in results.values())

            if score > best_score:
                best_score = score
                agent.save(best_path)

            log(
                "  eval: "
                + ", ".join(
                    f"{name} laps {m['laps']:.1f} fit {m['fitness']:.0f}"
                    for name, m in results.items()
                )
                + f" | worst {score:.0f} (best {best_score:.0f})"
            )

    agent.save(final_path)

    return best_score


def _mean(values):
    values = [float(v) for v in values]
    return float(np.mean(values)) if values else float("nan")


# ==================================================
# CLI
# ==================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a PPO agent.")
    parser.add_argument("--framework", choices=["torch"], default="torch")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--runs-dir", default="runs")

    # Every PPOConfig field becomes an option, e.g. --total-timesteps.
    for field in dataclasses.fields(PPOConfig):
        option = "--" + field.name.replace("_", "-")

        if field.type in ("bool", bool):
            parser.add_argument(
                option,
                type=lambda text: text.lower() in ("1", "true", "yes"),
                default=field.default
            )
        else:
            parser.add_argument(
                option,
                type=type(field.default),
                default=field.default
            )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    config = PPOConfig(**{
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(PPOConfig)
    })

    run_name = args.run_name or (
        f"{args.framework}_{config.tracks.replace(':', '')}_seed{config.seed}"
    )

    run_dir = os.path.join(args.runs_dir, run_name)

    agent = make_agent(args.framework, config)

    print(f"Run directory: {run_dir}")
    print(
        f"{config.num_updates} updates x {config.batch_size} steps "
        f"= {config.num_updates * config.batch_size:,} timesteps"
    )

    best = train(agent, config, run_dir)

    print(f"Best worst-track evaluation fitness: {best:.1f}")


if __name__ == "__main__":
    main()
