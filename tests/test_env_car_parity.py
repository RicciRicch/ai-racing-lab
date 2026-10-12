"""EnvCar must drive exactly like the original pixel-based Car."""

import os

import numpy as np
import pygame
import pytest

from brain import Brain
from car import Car
from envs.env_car import EnvCar
from envs.track_geometry import HEIGHT, TRACK_SETS, WIDTH, TrackGeometry


class RecordingPolicy:
    """Brain stand-in that records the inputs Car.neural_control passes."""

    def __init__(self, policy):
        self.policy = policy
        self.last_inputs = None

    def predict(self, inputs):
        self.last_inputs = list(inputs)
        return self.policy(inputs)


def wall_follower(inputs):
    # Simple hand-written driver that survives long enough to visit
    # checkpoints, so the comparison covers corners and laps.
    left_far, left, front, right, right_far, speed = inputs

    steering = float(np.clip(
        2.0 * (left + 0.5 * left_far - right - 0.5 * right_far),
        -1,
        1
    ))

    throttle = 1.0 if front > 0.6 else 0.3

    return steering, throttle


def random_brain_policy(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    brain = Brain()
    np.random.set_state(state)
    return brain.predict


def saved_brain_policies():
    policies = []

    for filename in (
        "best_brain_continuous.npz",
        "best_brain_multitrack.npz",
    ):
        if os.path.exists(filename):
            policies.append(
                pytest.param(Brain.load(filename).predict, id=filename)
            )

    return policies


POLICIES = [
    pytest.param(wall_follower, id="wall_follower"),
    pytest.param(random_brain_policy(1), id="random_brain_1"),
    pytest.param(random_brain_policy(2), id="random_brain_2"),
    *saved_brain_policies(),
]


def state(car):
    return (
        car.x,
        car.y,
        car.angle,
        car.velocity,
        car.alive,
        car.checkpoints_passed,
        car.current_checkpoint,
        car.laps_completed,
        car.progress_reward,
        car.speed_reward,
        car.frames_alive,
    )


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    "track",
    TRACK_SETS["all"],
    ids=lambda track: track.name
)
def test_trajectories_are_identical(track, policy):
    screen = pygame.Surface((WIDTH, HEIGHT))
    track.draw(screen)

    original = Car(track.start_x + 3.7, track.start_y - 1.3)
    original.angle = track.start_angle + 2.5
    original.brain = RecordingPolicy(policy)

    env_car = EnvCar(
        track.start_x + 3.7,
        track.start_y - 1.3,
        track.start_angle + 2.5,
        TrackGeometry(track),
    )

    for step in range(1800):
        if not original.alive:
            break

        # Original game loop order (main.py / evaluation scripts).
        original.neural_control(screen)
        original.check_collision(screen)

        if original.alive:
            original.update_checkpoint(track.checkpoints)
            original.update_progress_reward(track.checkpoints)

        # Environment version of the same step.
        inputs = env_car.observation()
        assert inputs == original.brain.last_inputs, f"step {step}"

        env_car.apply_controls(*policy(inputs))
        env_car.check_track_collision()

        if env_car.alive:
            env_car.update_checkpoint(track.checkpoints)
            env_car.update_progress_reward(track.checkpoints)

        assert state(env_car) == state(original), f"step {step}"


def test_wall_follower_reaches_checkpoints():
    # Guards the parity test itself: it must exercise more than
    # a straight line into the grass.
    track = TRACK_SETS["train"][0]
    car = EnvCar(
        track.start_x,
        track.start_y,
        track.start_angle,
        TrackGeometry(track),
    )

    for _ in range(1800):
        car.apply_controls(*wall_follower(car.observation()))
        car.check_track_collision()

        if not car.alive:
            break

        car.update_checkpoint(track.checkpoints)

    assert car.checkpoints_passed >= 2
