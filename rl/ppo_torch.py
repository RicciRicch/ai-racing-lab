"""Proximal Policy Optimization (PPO) agent in PyTorch.

Only the neural-network parts live here: the actor-critic model, action
sampling and one gradient step on a minibatch. Rollouts, advantages,
minibatching and logging are framework-independent (rl/common.py and
rl/trainer.py), so the TensorFlow agent can share them unchanged.

Policy: a Gaussian over [steering, throttle_raw]. The network predicts
the mean; the log standard deviation is a learned, state-independent
parameter. Sampled actions are not squashed; RacingEnv clips them to
[-1, 1]. Evaluation uses the mean action.
"""

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from rl.common import PPOConfig


def layer(in_features, out_features, std=np.sqrt(2)):
    """Linear layer with orthogonal initialization (standard for PPO)."""
    linear = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(linear.weight, std)
    nn.init.constant_(linear.bias, 0.0)
    return linear


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_size,
        action_size,
        hidden_size=64,
        initial_log_std=-0.5
    ):
        super().__init__()

        self.actor = nn.Sequential(
            layer(observation_size, hidden_size),
            nn.Tanh(),
            layer(hidden_size, hidden_size),
            nn.Tanh(),
            # Small output scale: initial actions stay near zero.
            layer(hidden_size, action_size, std=0.01),
        )

        self.critic = nn.Sequential(
            layer(observation_size, hidden_size),
            nn.Tanh(),
            layer(hidden_size, hidden_size),
            nn.Tanh(),
            layer(hidden_size, 1, std=1.0),
        )

        self.log_std = nn.Parameter(
            torch.full((action_size,), float(initial_log_std))
        )

    def value(self, observations):
        return self.critic(observations).squeeze(-1)

    def distribution(self, observations):
        mean = self.actor(observations)
        return Normal(mean, self.log_std.exp().expand_as(mean))


class TorchPPOAgent:
    framework = "pytorch"

    def __init__(
        self,
        config,
        observation_size=6,
        action_size=2
    ):
        self.config = config
        self.observation_size = observation_size
        self.action_size = action_size

        torch.manual_seed(config.seed)

        self.model = ActorCritic(
            observation_size,
            action_size,
            config.hidden_size,
            config.initial_log_std
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )

    # ==================================================
    # ROLLOUTS (NumPy in, NumPy out)
    # ==================================================

    @torch.no_grad()
    def act(self, observations):
        """Sample actions; returns actions, log-probabilities, values."""
        observations = torch.as_tensor(observations, dtype=torch.float32)

        distribution = self.model.distribution(observations)
        actions = distribution.sample()

        return (
            actions.numpy(),
            distribution.log_prob(actions).sum(-1).numpy(),
            self.model.value(observations).numpy(),
        )

    @torch.no_grad()
    def value(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        return self.model.value(observations).numpy()

    @torch.no_grad()
    def predict(self, observations):
        """Deterministic (mean) actions clipped to the action space."""
        observations = torch.as_tensor(observations, dtype=torch.float32)
        mean = self.model.actor(observations)
        return mean.clamp(-1.0, 1.0).numpy()

    def as_policy(self):
        """Single-observation callable, like envs.brain_policy.BrainPolicy."""
        def policy(observation):
            return self.predict(observation[None, :])[0]

        return policy

    # ==================================================
    # LEARNING
    # ==================================================

    def train_step(self, minibatch, learning_rate):
        """One PPO gradient step. Advantages are already normalized."""
        config = self.config

        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

        observations = torch.as_tensor(minibatch["observations"])
        actions = torch.as_tensor(minibatch["actions"])
        old_log_probs = torch.as_tensor(minibatch["log_probs"])
        advantages = torch.as_tensor(minibatch["advantages"])
        returns = torch.as_tensor(minibatch["returns"])

        distribution = self.model.distribution(observations)
        log_probs = distribution.log_prob(actions).sum(-1)
        entropy = distribution.entropy().sum(-1).mean()
        values = self.model.value(observations)

        log_ratio = log_probs - old_log_probs
        ratio = log_ratio.exp()

        # Clipped surrogate objective.
        policy_loss = torch.max(
            -advantages * ratio,
            -advantages * ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef)
        ).mean()

        value_loss = 0.5 * ((values - returns) ** 2).mean()

        loss = (
            policy_loss
            - config.ent_coef * entropy
            + config.vf_coef * value_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1).abs() > config.clip_coef).float().mean()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": approx_kl.item(),
            "clip_fraction": clip_fraction.item(),
        }

    # ==================================================
    # PERSISTENCE
    # ==================================================

    def save(self, path):
        torch.save(
            {
                "framework": self.framework,
                "config": self.config.to_dict(),
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "model": self.model.state_dict(),
            },
            path
        )

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, weights_only=True)

        agent = cls(
            PPOConfig(**checkpoint["config"]),
            checkpoint["observation_size"],
            checkpoint["action_size"]
        )

        agent.model.load_state_dict(checkpoint["model"])

        return agent
