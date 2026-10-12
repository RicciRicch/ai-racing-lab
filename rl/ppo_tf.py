"""Proximal Policy Optimization (PPO) agent in TensorFlow / Keras.

A line-by-line counterpart of rl/ppo_torch.py: same architecture,
initialization, objective and gradient clipping, behind the same
NumPy interface (act, value, predict, train_step, save, load). The
rollouts, advantages, minibatching and logging come from the shared,
framework-independent code in rl/common.py and rl/trainer.py.

TensorFlow-specific choices:
- the actor-critic is a Keras Model saved as a single .keras file
- act/predict/train_step are compiled with tf.function
- gradients come from tf.GradientTape
- the Gaussian log-density is written out explicitly, so no
  TensorFlow Probability dependency is needed
"""

import os

# Keras 3 can run on several backends; this agent is the TensorFlow one.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import tensorflow as tf

from rl.common import PPOConfig


LOG_2PI = float(np.log(2.0 * np.pi))


def dense(units, activation=None, gain=np.sqrt(2)):
    """Dense layer with orthogonal initialization (standard for PPO)."""
    return keras.layers.Dense(
        units,
        activation=activation,
        kernel_initializer=keras.initializers.Orthogonal(gain=gain),
        bias_initializer="zeros",
    )


@keras.saving.register_keras_serializable(package="ai_racing_lab")
class ActorCritic(keras.Model):
    def __init__(
        self,
        observation_size=6,
        action_size=2,
        hidden_size=64,
        initial_log_std=-0.5,
        ppo_config=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.initial_log_std = initial_log_std
        self.ppo_config = ppo_config

        self.actor = keras.Sequential([
            dense(hidden_size, "tanh"),
            dense(hidden_size, "tanh"),
            # Small output scale: initial actions stay near zero.
            dense(action_size, gain=0.01),
        ])

        self.critic = keras.Sequential([
            dense(hidden_size, "tanh"),
            dense(hidden_size, "tanh"),
            dense(1, gain=1.0),
        ])

        self.log_std = self.add_weight(
            name="log_std",
            shape=(action_size,),
            initializer=keras.initializers.Constant(initial_log_std),
        )

        # Create all weights now so they can be saved and restored.
        self(tf.zeros((1, observation_size)))

    def call(self, observations):
        mean = self.actor(observations)
        value = tf.squeeze(self.critic(observations), axis=-1)
        return mean, value

    def get_config(self):
        return {
            **super().get_config(),
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,
            "initial_log_std": self.initial_log_std,
            "ppo_config": self.ppo_config,
        }


def gaussian_log_prob(actions, mean, log_std):
    """Sum over action dimensions of a diagonal Normal log-density."""
    z = (actions - mean) / tf.exp(log_std)
    return tf.reduce_sum(-0.5 * z ** 2 - log_std - 0.5 * LOG_2PI, axis=-1)


class TFPPOAgent:
    framework = "tensorflow"

    def __init__(
        self,
        config,
        observation_size=6,
        action_size=2
    ):
        self.config = config
        self.observation_size = observation_size
        self.action_size = action_size

        # Seeds Python, NumPy and TensorFlow (weight initialization).
        keras.utils.set_random_seed(config.seed)

        self.model = ActorCritic(
            observation_size,
            action_size,
            config.hidden_size,
            config.initial_log_std,
            ppo_config=config.to_dict(),
        )

        self.optimizer = keras.optimizers.Adam(
            learning_rate=config.learning_rate,
            epsilon=1e-5
        )
        self.optimizer.build(self.model.trainable_variables)

        # Dedicated generator for action sampling.
        self.rng = tf.random.Generator.from_seed(config.seed)

        observation_spec = tf.TensorSpec([None, observation_size], tf.float32)

        self._act = tf.function(
            self._act_graph,
            input_signature=[observation_spec]
        )
        self._value = tf.function(
            lambda observations: self.model(observations)[1],
            input_signature=[observation_spec]
        )
        self._predict = tf.function(
            lambda observations: tf.clip_by_value(
                self.model.actor(observations), -1.0, 1.0
            ),
            input_signature=[observation_spec]
        )
        self._train = tf.function(
            self._train_graph,
            input_signature=[
                observation_spec,
                tf.TensorSpec([None, action_size], tf.float32),
                tf.TensorSpec([None], tf.float32),
                tf.TensorSpec([None], tf.float32),
                tf.TensorSpec([None], tf.float32),
            ]
        )

    # ==================================================
    # ROLLOUTS (NumPy in, NumPy out)
    # ==================================================

    def _act_graph(self, observations):
        mean, value = self.model(observations)
        log_std = self.model.log_std

        noise = self.rng.normal(tf.shape(mean))
        actions = mean + tf.exp(log_std) * noise

        log_probs = tf.reduce_sum(
            -0.5 * noise ** 2 - log_std - 0.5 * LOG_2PI,
            axis=-1
        )

        return actions, log_probs, value

    def act(self, observations):
        """Sample actions; returns actions, log-probabilities, values."""
        actions, log_probs, values = self._act(
            tf.convert_to_tensor(observations, tf.float32)
        )
        return actions.numpy(), log_probs.numpy(), values.numpy()

    def value(self, observations):
        return self._value(
            tf.convert_to_tensor(observations, tf.float32)
        ).numpy()

    def predict(self, observations):
        """Deterministic (mean) actions clipped to the action space."""
        return self._predict(
            tf.convert_to_tensor(observations, tf.float32)
        ).numpy()

    def as_policy(self):
        """Single-observation callable, like envs.brain_policy.BrainPolicy."""
        def policy(observation):
            return self.predict(observation[None, :])[0]

        return policy

    # ==================================================
    # LEARNING
    # ==================================================

    def _train_graph(self, observations, actions, old_log_probs, advantages, returns):
        config = self.config
        variables = self.model.trainable_variables

        with tf.GradientTape() as tape:
            mean, values = self.model(observations)
            log_std = self.model.log_std

            log_probs = gaussian_log_prob(actions, mean, log_std)

            # Entropy of a diagonal Gaussian does not depend on the state.
            entropy = tf.reduce_sum(0.5 + 0.5 * LOG_2PI + log_std)

            log_ratio = log_probs - old_log_probs
            ratio = tf.exp(log_ratio)

            # Clipped surrogate objective.
            policy_loss = tf.reduce_mean(tf.maximum(
                -advantages * ratio,
                -advantages * tf.clip_by_value(
                    ratio, 1 - config.clip_coef, 1 + config.clip_coef
                )
            ))

            value_loss = 0.5 * tf.reduce_mean((values - returns) ** 2)

            loss = (
                policy_loss
                - config.ent_coef * entropy
                + config.vf_coef * value_loss
            )

        gradients = tape.gradient(loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, config.max_grad_norm)
        self.optimizer.apply_gradients(zip(gradients, variables))

        approx_kl = tf.reduce_mean((ratio - 1) - log_ratio)
        clip_fraction = tf.reduce_mean(
            tf.cast(tf.abs(ratio - 1) > config.clip_coef, tf.float32)
        )

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    def train_step(self, minibatch, learning_rate):
        """One PPO gradient step. Advantages are already normalized."""
        self.optimizer.learning_rate = learning_rate

        stats = self._train(
            tf.convert_to_tensor(minibatch["observations"], tf.float32),
            tf.convert_to_tensor(minibatch["actions"], tf.float32),
            tf.convert_to_tensor(minibatch["log_probs"], tf.float32),
            tf.convert_to_tensor(minibatch["advantages"], tf.float32),
            tf.convert_to_tensor(minibatch["returns"], tf.float32),
        )

        return {key: float(value) for key, value in stats.items()}

    # ==================================================
    # PERSISTENCE
    # ==================================================

    def save(self, path):
        if not str(path).endswith(".keras"):
            raise ValueError("TensorFlow agents are saved as .keras files")

        self.model.save(path)

    @classmethod
    def load(cls, path):
        model = keras.models.load_model(path)

        agent = cls(
            PPOConfig(**model.ppo_config),
            model.observation_size,
            model.action_size
        )

        agent.model.set_weights(model.get_weights())

        return agent
