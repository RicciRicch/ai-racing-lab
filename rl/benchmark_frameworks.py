"""Time the framework-specific PPO operations of both agents.

Training time mixes environment simulation (identical for both
frameworks) with network calls. This benchmark isolates the network
calls, with the batch sizes the trainer uses, and alternates between
frameworks to reduce the effect of CPU frequency changes.

    python -m rl.benchmark_frameworks
"""

import os
import time

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from rl.common import PPOConfig
from rl.trainer import make_agent


ROUNDS = 5


def time_call(function, repeats):
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - start) / repeats * 1e6


def operations(agent, config):
    rng = np.random.default_rng(0)

    rollout_observations = rng.random((config.num_envs, 6), dtype=np.float32)
    single_observation = rollout_observations[:1]

    minibatch_observations = rng.random((config.minibatch_size, 6), dtype=np.float32)
    actions, log_probs, _ = agent.act(minibatch_observations)

    minibatch = {
        "observations": minibatch_observations,
        "actions": actions,
        "log_probs": log_probs,
        "advantages": rng.normal(size=config.minibatch_size).astype(np.float32),
        "returns": rng.normal(size=config.minibatch_size).astype(np.float32),
    }

    return {
        # (call, repeats, calls per 2,048-step update)
        "act (8 envs)": (lambda: agent.act(rollout_observations), 2000, config.num_steps),
        "value (1 obs)": (lambda: agent.value(single_observation), 2000, 1),
        "predict (1 obs)": (lambda: agent.predict(single_observation), 2000, 0),
        "train_step (256)": (
            lambda: agent.train_step(minibatch, config.learning_rate),
            300,
            config.update_epochs * config.num_minibatches,
        ),
    }


def main():
    config = PPOConfig()

    agents = {
        "pytorch": make_agent("torch", config),
        "tensorflow": make_agent("tf", config),
    }

    ops = {name: operations(agent, config) for name, agent in agents.items()}

    # Warm-up: builds tf.function graphs and allocator pools.
    for name in agents:
        for function, _, _ in ops[name].values():
            for _ in range(20):
                function()

    timings = {name: {op: [] for op in ops[name]} for name in agents}

    for _ in range(ROUNDS):
        for name in agents:
            for op, (function, repeats, _) in ops[name].items():
                timings[name][op].append(time_call(function, repeats))

    print(f"{'operation':<20} {'PyTorch':>12} {'TensorFlow':>12} {'TF / PT':>8}")

    for op in ops["pytorch"]:
        pt = np.median(timings["pytorch"][op])
        tf = np.median(timings["tensorflow"][op])
        print(f"{op:<20} {pt:>9.1f} us {tf:>9.1f} us {tf / pt:>7.2f}x")

    print()
    print("Network time per 2,048-step update (rollout calls + optimization):")

    for name in agents:
        total = sum(
            np.median(timings[name][op]) * calls
            for op, (_, _, calls) in ops[name].items()
        ) / 1e3
        print(f"  {name:<11} {total:7.1f} ms")


if __name__ == "__main__":
    main()
