import csv
import os
import random

import numpy as np
import pytest

from brain import Brain
from car import Car
from envs.track_geometry import TRACK_SETS
from rl.evaluate import load_policy
from rl.neuroevolution import (
    NEConfig,
    drive_population,
    evolution_settings,
    next_generation,
    train,
)


def tiny_config(**overrides):
    values = dict(
        tracks="train",
        total_timesteps=1,          # one generation
        max_steps=40,
        population_size=8,
        parent_count=3,
        elite_count=2,
        eval_episodes=1,
    )
    values.update(overrides)
    return NEConfig(**values)


def read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def test_evolution_settings_match_main_py():
    config = NEConfig()

    assert evolution_settings(0, config) == ("FINE", 0.01, 0.03, 0.08, 2)
    assert evolution_settings(8, config) == ("EXPLORE", 0.03, 0.08, 0.20, 5)
    assert evolution_settings(18, config) == ("ESCAPE", 0.05, 0.15, 0.35, 10)


def test_steps_are_counted_per_living_car():
    np.random.seed(0)
    brains = [Brain() for _ in range(5)]
    track = TRACK_SETS["train"][0]

    _, cars, steps = drive_population(brains, track, max_steps=60)

    assert steps == sum(car.frames_alive for car in cars)
    assert steps <= 5 * 60


def test_population_drives_like_the_original_car():
    # A candidate in drive_population follows the same trajectory as
    # the original Car driven by Car.neural_control.
    import pygame

    from envs.track_geometry import HEIGHT, WIDTH

    track = TRACK_SETS["train"][0]
    if os.path.exists("best_brain_continuous.npz"):
        brain = Brain.load("best_brain_continuous.npz")
    else:
        brain = Brain()

    random.seed(5)
    _, cars, _ = drive_population([brain], track, max_steps=300)

    random.seed(5)
    offsets = (random.uniform(-8, 8), random.uniform(-4, 4), random.uniform(-4, 4))

    screen = pygame.Surface((WIDTH, HEIGHT))
    track.draw(screen)
    original = Car(track.start_x + offsets[0], track.start_y + offsets[1])
    original.angle = track.start_angle + offsets[2]
    original.brain = brain

    for _ in range(300):
        if not original.alive:
            break
        original.neural_control(screen)
        original.check_collision(screen)
        if original.alive:
            original.update_checkpoint(track.checkpoints)
            original.update_progress_reward(track.checkpoints)

    car = cars[0]
    assert (car.x, car.y, car.angle, car.laps_completed, car.alive) == (
        original.x, original.y, original.angle,
        original.laps_completed, original.alive,
    )


def test_next_generation_keeps_elites_and_size():
    np.random.seed(0)
    config = NEConfig()
    ranked = [Brain() for _ in range(config.population_size)]

    population = next_generation(ranked, evolution_settings(0, config), config)

    assert len(population) == config.population_size

    for elite, original in zip(population[:2], ranked[:2]):
        np.testing.assert_array_equal(elite.weights1, original.weights1)
        assert elite is not original


def test_training_run_logs_and_saves(tmp_path):
    config = tiny_config(total_timesteps=1500)
    best = train(config, str(tmp_path), log=lambda message: None)

    progress = read_csv(tmp_path / "progress.csv")
    assert len(progress) >= 2
    assert int(progress[-1]["global_step"]) >= 1500
    assert float(progress[-1]["best_fitness"]) == pytest.approx(best)

    steps = [int(row["global_step"]) for row in progress]
    assert steps == sorted(steps)

    # Three training tracks per evaluated generation.
    assert len(read_csv(tmp_path / "eval.csv")) == 3 * len(progress)

    for name in ("best.npz", "final.npz"):
        policy = load_policy(str(tmp_path / name))
        assert policy(np.zeros(6, np.float32)).shape == (2,)


def test_same_seed_gives_same_evolution(tmp_path):
    runs = []
    for name in ("a", "b"):
        train(tiny_config(total_timesteps=800, seed=3), str(tmp_path / name),
              log=lambda message: None)
        runs.append([
            (r["global_step"], r["champion_fitness"])
            for r in read_csv(tmp_path / name / "progress.csv")
        ])

    assert runs[0] == runs[1]
