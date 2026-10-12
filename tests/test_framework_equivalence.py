"""The PyTorch and TensorFlow agents implement the same PPO.

Weights are copied from a PyTorch agent into a TensorFlow agent. Both
must then compute the same outputs, the same losses and the same
gradients, and their Adam updates must stay close.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("tensorflow")

from rl.common import PPOConfig
from rl.ppo_tf import TFPPOAgent, gaussian_log_prob
from rl.ppo_torch import TorchPPOAgent


def torch_layers(sequential):
    return [m for m in sequential if isinstance(m, torch.nn.Linear)]


def copy_torch_to_tf(torch_agent, tf_agent):
    for source, target in (
        (torch_agent.model.actor, tf_agent.model.actor),
        (torch_agent.model.critic, tf_agent.model.critic),
    ):
        for linear, dense in zip(torch_layers(source), target.layers):
            # torch.nn.Linear stores (out, in); keras Dense stores (in, out).
            dense.kernel.assign(linear.weight.detach().numpy().T)
            dense.bias.assign(linear.bias.detach().numpy())

    tf_agent.model.log_std.assign(torch_agent.model.log_std.detach().numpy())


def tf_weights(tf_agent):
    return [
        dense.kernel.numpy().T
        for dense in tf_agent.model.actor.layers + tf_agent.model.critic.layers
    ] + [tf_agent.model.log_std.numpy()]


def torch_weights(torch_agent):
    return [
        linear.weight.detach().numpy().copy()
        for linear in torch_layers(torch_agent.model.actor)
        + torch_layers(torch_agent.model.critic)
    ] + [torch_agent.model.log_std.detach().numpy().copy()]


@pytest.fixture
def agents():
    config = PPOConfig(seed=0)
    torch_agent = TorchPPOAgent(config)
    tf_agent = TFPPOAgent(config)
    copy_torch_to_tf(torch_agent, tf_agent)
    return torch_agent, tf_agent


@pytest.fixture
def minibatch(agents):
    torch_agent, _ = agents
    rng = np.random.default_rng(1)

    observations = rng.random((256, 6), dtype=np.float32)
    actions, log_probs, _ = torch_agent.act(observations)

    return {
        "observations": observations,
        "actions": actions,
        "log_probs": log_probs,
        "advantages": rng.normal(size=256).astype(np.float32),
        "returns": rng.normal(size=256).astype(np.float32),
    }


def test_same_outputs(agents, minibatch):
    torch_agent, tf_agent = agents
    observations = minibatch["observations"]

    np.testing.assert_allclose(
        torch_agent.predict(observations), tf_agent.predict(observations),
        atol=1e-6
    )
    np.testing.assert_allclose(
        torch_agent.value(observations), tf_agent.value(observations),
        atol=1e-5
    )

    # Log-density of the same actions under both policies.
    torch_log_probs = minibatch["log_probs"]
    mean, _ = tf_agent.model(observations)
    tf_log_probs = gaussian_log_prob(
        minibatch["actions"], mean, tf_agent.model.log_std
    ).numpy()

    np.testing.assert_allclose(torch_log_probs, tf_log_probs, atol=1e-5)


def weight_change_difference(torch_agent, tf_agent, initial):
    """||change_tf - change_torch|| / ||change_torch|| over all weights."""
    torch_change = np.concatenate([
        (after - before).ravel()
        for after, before in zip(torch_weights(torch_agent), initial)
    ])
    tf_change = np.concatenate([
        (after - before).ravel()
        for after, before in zip(tf_weights(tf_agent), initial)
    ])
    return np.linalg.norm(tf_change - torch_change) / np.linalg.norm(torch_change)


def test_same_losses_and_gradients(agents, minibatch):
    # Plain SGD turns the update into -lr * clipped gradient, so equal
    # updates mean equal gradients. Adam is compared separately below.
    import keras

    torch_agent, tf_agent = agents
    initial = torch_weights(torch_agent)

    torch_agent.optimizer = torch.optim.SGD(
        torch_agent.model.parameters(), lr=0.1
    )
    tf_agent.optimizer = keras.optimizers.SGD(learning_rate=0.1)
    tf_agent.optimizer.build(tf_agent.model.trainable_variables)

    torch_stats = torch_agent.train_step(minibatch, learning_rate=0.1)
    tf_stats = tf_agent.train_step(minibatch, learning_rate=0.1)

    # Losses are computed before the update, from identical weights.
    for key in ("policy_loss", "value_loss", "entropy"):
        assert tf_stats[key] == pytest.approx(torch_stats[key], rel=1e-5, abs=1e-6)

    assert weight_change_difference(torch_agent, tf_agent, initial) < 1e-4


def test_adam_updates_stay_close(agents, minibatch):
    # With the shared small adam_epsilon, PyTorch's and Keras' Adam
    # formulations differ only for near-zero gradients.
    torch_agent, tf_agent = agents
    initial = torch_weights(torch_agent)

    for _ in range(20):
        torch_agent.train_step(minibatch, learning_rate=3e-4)
        tf_agent.train_step(minibatch, learning_rate=3e-4)

    assert weight_change_difference(torch_agent, tf_agent, initial) < 0.03

    observations = minibatch["observations"]

    np.testing.assert_allclose(
        torch_agent.predict(observations), tf_agent.predict(observations),
        atol=1e-4
    )
