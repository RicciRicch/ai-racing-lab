"""Framework-independent parts of PPO.

Everything here uses only NumPy, so the PyTorch agent (rl/ppo_torch.py)
and the TensorFlow agent (phase 3) share the same configuration, the
same environment handling and the same advantage computation. The two
implementations then differ only in the neural-network code, which
keeps the framework comparison fair.
"""

import dataclasses
import json

import numpy as np

from envs.racing_env import RacingEnv
from envs.track_geometry import TRACK_SETS


# ==================================================
# CONFIGURATION
# ==================================================

@dataclasses.dataclass
class PPOConfig:
    # Environment
    tracks: str = "train"
    track_sampling: str = "uniform"     # "uniform" or "weakest"
    weakest_prob: float = 0.5
    num_envs: int = 8
    max_steps: int = 1800

    # Rollouts
    total_timesteps: int = 1_000_000
    num_steps: int = 256            # per environment, per update

    # Optimization
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 10
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True

    # Network
    hidden_size: int = 64
    initial_log_std: float = -0.5

    # Bookkeeping
    seed: int = 0
    eval_interval: int = 10         # updates between evaluations
    eval_episodes: int = 3          # per track

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self):
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self):
        return self.total_timesteps // self.batch_size

    def to_dict(self):
        return dataclasses.asdict(self)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as file:
            return cls(**json.load(file))


# ==================================================
# TRACKS
# ==================================================

def resolve_tracks(spec):
    """Track list from a spec string.

    "train", "unseen", "all"  -> a whole track set
    "train:0"                 -> a single track from a set by index
    """
    if ":" in spec:
        set_name, index = spec.split(":")
        return [TRACK_SETS[set_name][int(index)]]

    return list(TRACK_SETS[spec])


# ==================================================
# TRACK SAMPLING
# ==================================================

class WeakestTrackSampler:
    """Pick the track the agent currently drives worst more often.

    The RL counterpart of the maximin fitness used by multi-track
    neuroevolution: instead of scoring a policy by its weakest track,
    training time is shifted toward that track.

    With probability ``weakest_prob`` the track with the lowest moving
    average of episode returns is chosen; otherwise a uniform random
    track. Tracks without any finished episode count as weakest.
    """

    def __init__(self, num_tracks, weakest_prob=0.5, smoothing=0.1, seed=0):
        self.num_tracks = num_tracks
        self.weakest_prob = weakest_prob
        self.smoothing = smoothing
        self.rng = np.random.default_rng(seed)
        self.average_returns = np.full(num_tracks, np.nan)

    def update(self, track_index, episode_return):
        current = self.average_returns[track_index]

        if np.isnan(current):
            self.average_returns[track_index] = episode_return
        else:
            self.average_returns[track_index] = (
                (1 - self.smoothing) * current
                + self.smoothing * episode_return
            )

    def weakest(self):
        unseen = np.flatnonzero(np.isnan(self.average_returns))

        if unseen.size:
            return int(unseen[0])

        return int(np.argmin(self.average_returns))

    def sample(self):
        if self.rng.random() < self.weakest_prob:
            return self.weakest()

        return int(self.rng.integers(self.num_tracks))


# ==================================================
# VECTOR ENVIRONMENT
# ==================================================

class VecEnv:
    """Steps several RacingEnv instances in lockstep.

    Finished episodes are reset immediately. For every finished
    environment the last observation of the old episode is returned in
    ``final_observations`` so truncated episodes can be bootstrapped
    with the value of that state.
    """

    def __init__(
        self,
        tracks,
        num_envs,
        seed=0,
        max_steps=1800,
        track_sampling="uniform",
        weakest_prob=0.5
    ):
        if track_sampling not in ("uniform", "weakest"):
            raise ValueError("track_sampling must be 'uniform' or 'weakest'")

        self.envs = [
            RacingEnv(
                tracks=resolve_tracks(tracks),
                max_steps=max_steps
            )
            for _ in range(num_envs)
        ]

        self.num_envs = num_envs
        self.seed = seed

        self.observation_size = self.envs[0].observation_space.shape[0]
        self.action_size = self.envs[0].action_space.shape[0]

        self._returns = np.zeros(num_envs)

        # "uniform" leaves track choice to each RacingEnv's own seeded
        # generator; "weakest" picks tracks through a shared sampler.
        self.sampler = None

        if track_sampling == "weakest":
            self.sampler = WeakestTrackSampler(
                len(self.envs[0].tracks),
                weakest_prob=weakest_prob,
                seed=seed
            )

    def _reset_options(self):
        if self.sampler is None:
            return None

        return {"track_index": self.sampler.sample()}

    def reset(self):
        observations = [
            env.reset(seed=self.seed + index, options=self._reset_options())[0]
            for index, env in enumerate(self.envs)
        ]

        self._returns[:] = 0.0

        return np.stack(observations)

    def step(self, actions):
        observations = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        final_observations = {}
        finished_episodes = []

        for index, (env, action) in enumerate(zip(self.envs, actions)):
            observation, reward, term, trunc, info = env.step(action)

            rewards[index] = reward
            terminated[index] = term
            truncated[index] = trunc
            self._returns[index] += reward

            if term or trunc:
                final_observations[index] = observation

                finished_episodes.append({
                    **info,
                    "return": float(self._returns[index]),
                })

                if self.sampler is not None:
                    self.sampler.update(
                        info["track_index"],
                        self._returns[index]
                    )

                self._returns[index] = 0.0

                # Later resets continue the env's seeded generator.
                observation, _ = env.reset(options=self._reset_options())

            observations.append(observation)

        return (
            np.stack(observations),
            rewards,
            terminated,
            truncated,
            final_observations,
            finished_episodes,
        )


# ==================================================
# ROLLOUT BUFFER
# ==================================================

class RolloutBuffer:
    """Fixed-size storage for one PPO update (steps x envs)."""

    def __init__(self, num_steps, num_envs, observation_size, action_size):
        shape = (num_steps, num_envs)

        self.observations = np.zeros(shape + (observation_size,), np.float32)
        self.actions = np.zeros(shape + (action_size,), np.float32)
        self.log_probs = np.zeros(shape, np.float32)
        self.rewards = np.zeros(shape, np.float32)
        self.dones = np.zeros(shape, np.float32)
        self.values = np.zeros(shape, np.float32)

        self.step = 0

    def add(self, observations, actions, log_probs, rewards, dones, values):
        t = self.step

        self.observations[t] = observations
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values

        self.step += 1

    def flattened(self, advantages, returns):
        """All samples as flat arrays, ready for minibatching."""
        def flat(array):
            return array.reshape((-1,) + array.shape[2:])

        return {
            "observations": flat(self.observations),
            "actions": flat(self.actions),
            "log_probs": flat(self.log_probs),
            "values": flat(self.values),
            "advantages": flat(advantages),
            "returns": flat(returns),
        }

    def reset(self):
        self.step = 0


# ==================================================
# ADVANTAGES
# ==================================================

def compute_gae(
    rewards,
    values,
    dones,
    last_values,
    gamma,
    gae_lambda
):
    """Generalized Advantage Estimation.

    rewards, values, dones: arrays of shape (steps, envs).
    dones[t] marks that the episode ended after step t.
    last_values: value of the observation after the final step.

    Truncated episodes should already have gamma * V(final_obs) added
    to their last reward, so a single ``dones`` flag is enough here.
    """
    num_steps = rewards.shape[0]

    advantages = np.zeros_like(rewards)
    gae = np.zeros_like(last_values)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_values = last_values
        else:
            next_values = values[t + 1]

        not_done = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_values * not_done - values[t]
        gae = delta + gamma * gae_lambda * not_done * gae

        advantages[t] = gae

    returns = advantages + values

    return advantages, returns


def minibatch_indices(batch_size, minibatch_size, rng):
    """Shuffled index arrays covering the batch once."""
    order = rng.permutation(batch_size)

    return [
        order[start:start + minibatch_size]
        for start in range(0, batch_size, minibatch_size)
    ]
