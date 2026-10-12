"""Draw the paths driven by saved models on every track.

Each panel shows one track with its checkpoints and the trajectory of
every given model over one evaluation episode, which makes racing
lines - and missed checkpoints - visible.

    python -m rl.plot_trajectories runs/ne_multi_seed1/best.npz runs/torch_multi_seed0/best.pt \
        --labels "NE" "PPO" --output results/rl/trajectories.png
"""

import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib.pyplot as plt
import numpy as np
import pygame

from envs.racing_env import RacingEnv
from envs.track_geometry import HEIGHT, TRACK_SETS, WIDTH
from rl.evaluate import EVAL_SEED, load_policy


def drive(policy, track_index, seed):
    env = RacingEnv(tracks="all")
    observation, _ = env.reset(seed=seed, options={"track_index": track_index})

    path = [(env.car.x, env.car.y)]

    while True:
        observation, _, terminated, truncated, info = env.step(policy(observation))
        path.append((env.car.x, env.car.y))

        if terminated or truncated:
            return np.array(path), info


def track_image(track):
    surface = pygame.Surface((WIDTH, HEIGHT))
    track.draw(surface)

    for checkpoint in track.checkpoints:
        pygame.draw.rect(surface, (0, 200, 255), checkpoint, 3)

    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(m) for m in args.models]
    policies = [load_policy(model) for model in args.models]
    colors = ["#ffd400", "#ff4d4d", "#ffffff", "#c77dff"]

    tracks = TRACK_SETS["all"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    for track_index, (track, axis) in enumerate(zip(tracks, axes.ravel())):
        axis.imshow(track_image(track))

        for policy, label, color in zip(policies, labels, colors):
            path, info = drive(policy, track_index, args.seed)
            status = "crash" if info["crashed"] else f"{info['laps']} laps"

            axis.plot(path[:, 0], path[:, 1], color=color, lw=1.0,
                      label=f"{label}: {status}")

        axis.set_title(track.name)
        axis.legend(fontsize=8, loc="upper left")
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(args.output, dpi=90)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
