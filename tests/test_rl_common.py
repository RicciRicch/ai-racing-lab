import numpy as np
import pytest

from rl.common import (
    PPOConfig,
    RolloutBuffer,
    VecEnv,
    compute_gae,
    minibatch_indices,
    resolve_tracks,
)


def reference_gae(rewards, values, dones, last_value, gamma, lam):
    # Direct definition for a single environment:
    # A_t = sum_k (gamma * lam)^k * delta_{t+k}, stopping at episode ends.
    steps = len(rewards)
    next_values = list(values[1:]) + [last_value]
    deltas = [
        rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        for t in range(steps)
    ]

    advantages = []
    for t in range(steps):
        total, factor = 0.0, 1.0
        for k in range(t, steps):
            total += factor * deltas[k]
            if dones[k]:
                break
            factor *= gamma * lam
        advantages.append(total)

    return np.array(advantages)


def test_gae_matches_reference_definition():
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=(12, 1)).astype(np.float32)
    values = rng.normal(size=(12, 1)).astype(np.float32)
    dones = np.zeros((12, 1), np.float32)
    dones[4] = dones[9] = 1.0
    last_values = np.array([0.7], np.float32)

    advantages, returns = compute_gae(
        rewards, values, dones, last_values, gamma=0.99, gae_lambda=0.95
    )

    expected = reference_gae(
        rewards[:, 0], values[:, 0], dones[:, 0], 0.7, 0.99, 0.95
    )

    np.testing.assert_allclose(advantages[:, 0], expected, rtol=1e-5)
    np.testing.assert_allclose(returns, advantages + values, rtol=1e-6)


def test_gae_with_lambda_one_is_discounted_return_minus_value():
    rewards = np.ones((5, 1), np.float32)
    values = np.zeros((5, 1), np.float32)
    dones = np.zeros((5, 1), np.float32)

    advantages, _ = compute_gae(
        rewards, values, dones, np.zeros(1, np.float32), 0.5, 1.0
    )

    # 1 + 0.5 + 0.25 + 0.125 + 0.0625 for the first step
    assert advantages[0, 0] == pytest.approx(1.9375)
    assert advantages[4, 0] == pytest.approx(1.0)


def test_config_derived_sizes_and_round_trip(tmp_path):
    config = PPOConfig(num_envs=4, num_steps=128, num_minibatches=4,
                       total_timesteps=10_000)

    assert config.batch_size == 512
    assert config.minibatch_size == 128
    assert config.num_updates == 19

    config.save(tmp_path / "config.json")
    assert PPOConfig.load(tmp_path / "config.json") == config


def test_resolve_tracks():
    assert len(resolve_tracks("train")) == 3
    assert len(resolve_tracks("all")) == 4
    assert [t.name for t in resolve_tracks("train:0")] == ["Classic Oval"]
    assert [t.name for t in resolve_tracks("unseen")] == ["Unseen Narrow Oval"]


def test_vec_env_resets_finished_episodes_and_returns_final_obs():
    vec_env = VecEnv("train", num_envs=3, seed=0, max_steps=20)
    observations = vec_env.reset()

    assert observations.shape == (3, 6)

    idle = np.tile([0.0, -1.0], (3, 1)).astype(np.float32)
    finished = []

    for step in range(20):
        observations, rewards, terminated, truncated, finals, episodes = (
            vec_env.step(idle)
        )
        finished += episodes

    # Idle cars never crash, so all three truncate on step 20.
    assert truncated.all() and not terminated.any()
    assert sorted(finals) == [0, 1, 2]
    assert len(finished) == 3
    assert all(episode["frames_alive"] == 20 for episode in finished)

    # Returned observations already belong to the new episodes.
    assert all(env.steps == 0 for env in vec_env.envs)


def test_vec_env_episode_return_equals_scaled_fitness():
    vec_env = VecEnv("train:0", num_envs=1, seed=0)
    vec_env.reset()

    full_throttle = np.array([[0.0, 1.0]], np.float32)

    while True:
        *_, episodes = vec_env.step(full_throttle)
        if episodes:
            break

    episode = episodes[0]
    assert episode["return"] == pytest.approx(episode["fitness"] * 0.01, rel=1e-4)


def test_rollout_buffer_and_minibatches():
    buffer = RolloutBuffer(num_steps=4, num_envs=2, observation_size=6,
                           action_size=2)

    for t in range(4):
        buffer.add(
            np.full((2, 6), t), np.zeros((2, 2)), np.zeros(2),
            np.ones(2), np.zeros(2), np.zeros(2)
        )

    batch = buffer.flattened(np.zeros((4, 2)), np.zeros((4, 2)))
    assert batch["observations"].shape == (8, 6)
    assert batch["observations"][:, 0].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]

    batches = minibatch_indices(8, 4, np.random.default_rng(0))
    assert len(batches) == 2
    assert sorted(np.concatenate(batches).tolist()) == list(range(8))
