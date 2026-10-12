"""Evaluate any saved policy in RacingEnv with the same metrics.

Supports neuroevolution brains (.npz) and PyTorch agents (.pt). Every
policy is evaluated on the same tracks with the same seeds, so results
are directly comparable.

    python -m rl.evaluate best_brain_multitrack.npz --tracks all
    python -m rl.evaluate runs/torch_multi/best.pt --tracks all --episodes 10
"""

import argparse
import statistics

from envs.racing_env import RacingEnv
from rl.common import resolve_tracks


# Evaluation seeds are separate from training seeds.
EVAL_SEED = 10_000


def load_policy(path):
    """Callable observation -> action for a saved model file."""
    if path.endswith(".npz"):
        from envs.brain_policy import BrainPolicy

        return BrainPolicy.load(path)

    if path.endswith(".pt"):
        from rl.ppo_torch import TorchPPOAgent

        return TorchPPOAgent.load(path).as_policy()

    raise ValueError(f"Unknown model type: {path}")


def run_episode(env, policy, seed, track_index):
    observation, info = env.reset(
        seed=seed,
        options={"track_index": track_index}
    )

    while True:
        observation, _, terminated, truncated, info = env.step(
            policy(observation)
        )

        if terminated or truncated:
            return info


def evaluate_policy(
    policy,
    tracks="train",
    episodes=10,
    seed=EVAL_SEED,
    max_steps=1800
):
    """Per-track summary over ``episodes`` seeded episodes."""
    track_list = resolve_tracks(tracks)

    env = RacingEnv(
        tracks=track_list,
        max_steps=max_steps
    )

    results = {}

    for track_index, track in enumerate(track_list):
        infos = [
            run_episode(env, policy, seed + episode, track_index)
            for episode in range(episodes)
        ]

        laps = [info["laps"] for info in infos]

        results[track.name] = {
            "episodes": episodes,
            "laps": statistics.mean(laps),
            "best_laps": max(laps),
            "worst_laps": min(laps),
            "checkpoints": statistics.mean(i["checkpoints"] for i in infos),
            "frames": statistics.mean(i["frames_alive"] for i in infos),
            "speed": statistics.mean(i["average_speed"] for i in infos),
            "crash_rate": sum(i["crashed"] for i in infos) / episodes,
            "fitness": statistics.mean(i["fitness"] for i in infos),
        }

    return results


def format_results(results):
    lines = [
        f"{'Track':<20} {'Laps':>6} {'Best':>5} {'Worst':>5} "
        f"{'Checkp.':>8} {'Frames':>8} {'Speed':>6} {'Crash':>6} "
        f"{'Fitness':>9}"
    ]

    for name, r in results.items():
        lines.append(
            f"{name:<20} {r['laps']:>6.2f} {r['best_laps']:>5} "
            f"{r['worst_laps']:>5} {r['checkpoints']:>8.2f} "
            f"{r['frames']:>8.1f} {r['speed']:>6.3f} "
            f"{r['crash_rate']:>6.0%} {r['fitness']:>9.1f}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("--tracks", default="all")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument("--output", help="also write the table to a file")
    args = parser.parse_args()

    results = evaluate_policy(
        load_policy(args.model),
        tracks=args.tracks,
        episodes=args.episodes,
        seed=args.seed
    )

    table = (
        f"Model: {args.model}\n"
        f"Episodes per track: {args.episodes} (seeds {args.seed}-"
        f"{args.seed + args.episodes - 1})\n\n"
        + format_results(results)
    )

    print(table)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(table + "\n")


if __name__ == "__main__":
    main()
