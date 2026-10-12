import os

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import envs  # noqa: F401  (registers AIRacing-v0)
from envs.brain_policy import BrainPolicy
from envs.racing_env import RacingEnv, track_fitness
from envs.track_geometry import TRACK_SETS


def test_passes_gymnasium_env_checker():
    check_env(gym.make("AIRacing-v0", render_mode="rgb_array").unwrapped)


def test_registered_with_gymnasium():
    env = gym.make("AIRacing-v0", tracks="unseen")

    observation, info = env.reset(seed=0)

    assert observation.shape == (6,)
    assert info["track"] == "Unseen Narrow Oval"


def test_spaces_match_brain_interface():
    env = RacingEnv()

    assert env.observation_space.shape == (6,)
    assert env.action_space.shape == (2,)
    assert env.action_space.low.tolist() == [-1, -1]
    assert env.action_space.high.tolist() == [1, 1]


def run_episode(env, policy, **reset_kwargs):
    observation, info = env.reset(**reset_kwargs)
    total_reward = 0.0

    while True:
        observation, reward, terminated, truncated, info = env.step(
            policy(observation)
        )
        total_reward += reward

        if terminated or truncated:
            return total_reward, terminated, truncated, info


def full_throttle(observation):
    return np.array([0.0, 1.0], dtype=np.float32)


def test_return_equals_scaled_neuroevolution_fitness():
    env = RacingEnv(reward_scale=0.01)

    total, _, _, info = run_episode(env, full_throttle, seed=3)

    assert total == pytest.approx(info["fitness"] * 0.01)
    assert info["fitness"] == pytest.approx(track_fitness(env.car))


def test_crash_terminates_episode():
    env = RacingEnv(tracks="train")

    total, terminated, truncated, info = run_episode(
        env,
        full_throttle,
        seed=0,
        options={"track_index": 0}
    )

    assert terminated and not truncated
    assert info["crashed"]

    with pytest.raises(RuntimeError):
        env.step(full_throttle(None))


def test_truncates_after_max_steps():
    env = RacingEnv(max_steps=25)

    # Zero throttle: friction keeps the car still, so it never crashes.
    idle = lambda observation: np.array([0.0, -1.0], dtype=np.float32)

    _, terminated, truncated, info = run_episode(env, idle, seed=0)

    assert truncated and not terminated
    assert info["frames_alive"] == 25


def test_same_seed_gives_same_episode():
    rewards = []

    for _ in range(2):
        env = RacingEnv(tracks="all")
        rewards.append(run_episode(env, full_throttle, seed=42))

    assert rewards[0] == rewards[1]


def test_track_selection_options():
    env = RacingEnv(tracks="all", track_selection="cycle")

    names = [env.reset(seed=0)[1]["track"] for _ in range(5)]

    assert names == [
        track.name for track in TRACK_SETS["all"]
    ] + [TRACK_SETS["all"][0].name]

    _, info = env.reset(options={"track_index": 2})
    assert info["track"] == TRACK_SETS["all"][2].name


def test_actions_are_clipped():
    env = RacingEnv(randomize_start=False)
    env.reset(seed=0)

    env.step(np.array([5.0, 9.0]))

    assert env.car.last_steering == 1.0
    assert env.car.last_throttle == 1.0


@pytest.mark.skipif(
    not os.path.exists("best_brain_continuous.npz"),
    reason="trained single-track model not available"
)
def test_saved_champion_drives_through_env():
    # The single-track champion completes 10 laps on the Classic Oval
    # in results/single_track_evaluation.txt; it must do the same here.
    env = RacingEnv(tracks="train", randomize_start=False)
    policy = BrainPolicy.load("best_brain_continuous.npz")

    _, terminated, truncated, info = run_episode(
        env,
        policy,
        seed=0,
        options={"track_index": 0}
    )

    assert truncated and not terminated
    assert info["laps"] == 10


def test_rgb_array_render():
    env = RacingEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env.step(full_throttle(None))

    frame = env.render()

    assert frame.shape == (700, 1000, 3)
    assert frame.dtype == np.uint8

    env.close()


def test_human_render_runs_headless():
    # conftest.py selects SDL's dummy video driver.
    env = RacingEnv(render_mode="human")
    env.metadata = {**env.metadata, "render_fps": 0}
    env.reset(seed=0)
    env.step(full_throttle(None))
    env.close()
