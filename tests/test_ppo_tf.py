import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from rl.common import PPOConfig
from rl.ppo_tf import TFPPOAgent


def make_agent(**overrides):
    return TFPPOAgent(PPOConfig(**overrides))


def test_act_shapes_and_log_probs():
    agent = make_agent()
    observations = np.random.default_rng(0).random((5, 6), dtype=np.float32)

    actions, log_probs, values = agent.act(observations)

    assert actions.shape == (5, 2)
    assert log_probs.shape == (5,)
    assert values.shape == (5,)

    # Log-probabilities agree with an explicit Normal density.
    mean = agent.model.actor(observations).numpy()
    std = np.exp(agent.config.initial_log_std)
    expected = (
        -0.5 * ((actions - mean) / std) ** 2
        - np.log(std)
        - 0.5 * np.log(2 * np.pi)
    ).sum(-1)

    np.testing.assert_allclose(log_probs, expected, rtol=1e-4, atol=1e-5)


def test_initial_policy_is_near_zero_and_predict_is_clipped():
    agent = make_agent()
    observations = np.ones((3, 6), np.float32)

    assert np.abs(agent.predict(observations)).max() < 0.1

    output_layer = agent.model.actor.layers[-1]
    output_layer.bias.assign(np.full(2, 5.0, np.float32))

    assert agent.predict(observations).max() == 1.0


def test_same_seed_same_network():
    first, second = make_agent(seed=3), make_agent(seed=3)

    for a, b in zip(first.model.get_weights(), second.model.get_weights()):
        np.testing.assert_array_equal(a, b)


def test_train_steps_move_policy_toward_advantaged_actions():
    # Reward actions that steer more than the current mean; PPO updates
    # must then raise the mean steering output.
    agent = make_agent()
    rng = np.random.default_rng(0)

    observations = rng.random((256, 6), dtype=np.float32)
    mean_before = agent.predict(observations)[:, 0].mean()

    actions, log_probs, _ = agent.act(observations)
    advantages = np.sign(actions[:, 0] - mean_before).astype(np.float32)

    minibatch = {
        "observations": observations,
        "actions": actions,
        "log_probs": log_probs,
        "advantages": advantages,
        "returns": np.zeros(256, np.float32),
    }

    for _ in range(5):
        stats = agent.train_step(minibatch, learning_rate=1e-3)

    assert agent.predict(observations)[:, 0].mean() > mean_before + 0.05
    assert set(stats) == {
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"
    }


def test_value_head_fits_returns():
    agent = make_agent()
    observations = np.random.default_rng(0).random((128, 6), dtype=np.float32)
    returns = (2 * observations[:, 2]).astype(np.float32)

    minibatch = {
        "observations": observations,
        "actions": np.zeros((128, 2), np.float32),
        "log_probs": np.zeros(128, np.float32),
        "advantages": np.zeros(128, np.float32),
        "returns": returns,
    }

    first = agent.train_step(minibatch, 1e-3)["value_loss"]
    for _ in range(200):
        last = agent.train_step(minibatch, 1e-3)["value_loss"]

    assert last < 0.2 * first


def test_save_and_load_round_trip(tmp_path):
    agent = make_agent(hidden_size=32, seed=7)
    path = tmp_path / "agent.keras"
    agent.save(path)

    loaded = TFPPOAgent.load(str(path))
    observations = np.random.default_rng(1).random((4, 6), dtype=np.float32)

    np.testing.assert_array_equal(
        agent.predict(observations), loaded.predict(observations)
    )
    assert loaded.config.hidden_size == 32

    policy = loaded.as_policy()
    assert policy(observations[0]).shape == (2,)


def test_save_requires_keras_extension(tmp_path):
    with pytest.raises(ValueError):
        make_agent().save(str(tmp_path / "agent.pt"))


def test_explicit_log_density_matches_formula():
    from rl.ppo_tf import gaussian_log_prob

    actions = np.array([[0.3, -0.2]], np.float32)
    mean = np.array([[0.1, 0.1]], np.float32)
    log_std = np.array([-0.5, 0.2], np.float32)

    std = np.exp(log_std)
    expected = np.sum(
        -0.5 * ((actions - mean) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
    )

    assert gaussian_log_prob(actions, mean, log_std).numpy()[0] == pytest.approx(
        expected, rel=1e-5
    )
