"""Measure simulation speed: original pixel-based Car vs RacingEnv.

    python -m envs.benchmark
"""

import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from brain import Brain
from car import Car
from envs.brain_policy import BrainPolicy
from envs.racing_env import RacingEnv
from envs.track_geometry import HEIGHT, TRACK_SETS, WIDTH

STEPS = 20_000
MODEL_FILE = "best_brain_multitrack.npz"


def load_brain():
    if os.path.exists(MODEL_FILE):
        return Brain.load(MODEL_FILE)

    return Brain()


def benchmark_original(brain, track):
    screen = pygame.Surface((WIDTH, HEIGHT))
    track.draw(screen)

    car = Car(track.start_x, track.start_y)
    car.angle = track.start_angle
    car.brain = brain

    start = time.perf_counter()

    for _ in range(STEPS):
        if not car.alive:
            car = Car(track.start_x, track.start_y)
            car.angle = track.start_angle
            car.brain = brain

        car.neural_control(screen)
        car.check_collision(screen)

        if car.alive:
            car.update_checkpoint(track.checkpoints)
            car.update_progress_reward(track.checkpoints)

    return STEPS / (time.perf_counter() - start)


def benchmark_env(brain, track_index):
    env = RacingEnv(tracks="train", randomize_start=False)
    policy = BrainPolicy(brain)

    observation, _ = env.reset(seed=0, options={"track_index": track_index})

    start = time.perf_counter()

    for _ in range(STEPS):
        observation, _, terminated, truncated, _ = env.step(policy(observation))

        if terminated or truncated:
            observation, _ = env.reset(options={"track_index": track_index})

    return STEPS / (time.perf_counter() - start)


def main():
    brain = load_brain()

    print(f"{'Track':<20} {'original':>12} {'RacingEnv':>12} {'speed-up':>9}")

    for index, track in enumerate(TRACK_SETS["train"]):
        original = benchmark_original(brain, track)
        env = benchmark_env(brain, index)

        print(
            f"{track.name:<20} {original:>9,.0f}/s {env:>9,.0f}/s "
            f"{env / original:>8.1f}x"
        )


if __name__ == "__main__":
    main()
