"""Watch a policy drive in RacingEnv.

Examples (run from the project root):

    python -m envs.demo --policy brain --model best_brain_multitrack.npz
    python -m envs.demo --policy brain --tracks unseen --episodes 3
    python -m envs.demo --policy random --tracks all
"""

import argparse

import numpy as np

from envs.brain_policy import BrainPolicy
from envs.racing_env import RacingEnv


def make_policy(args, env):
    if args.policy == "brain":
        return BrainPolicy.load(args.model)

    if args.policy == "random":
        return lambda observation: env.action_space.sample()

    # Idle: no steering, no throttle.
    return lambda observation: np.array([0.0, -1.0], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", choices=["brain", "random", "idle"], default="brain")
    parser.add_argument("--model", default="best_brain_multitrack.npz")
    parser.add_argument("--tracks", choices=["train", "unseen", "all"], default="train")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = RacingEnv(
        tracks=args.tracks,
        track_selection="cycle",
        render_mode="human",
    )

    env.action_space.seed(args.seed)
    policy = make_policy(args, env)

    for episode in range(args.episodes):
        observation, info = env.reset(
            seed=args.seed + episode
        )

        episode_return = 0.0

        while True:
            observation, reward, terminated, truncated, info = env.step(
                policy(observation)
            )
            episode_return += reward

            if terminated or truncated:
                break

        print(
            f"Episode {episode + 1}: {info['track']:<20} "
            f"return={episode_return:8.2f} laps={info['laps']} "
            f"checkpoints={info['checkpoints']} "
            f"crashed={info['crashed']}"
        )

    env.close()


if __name__ == "__main__":
    main()
