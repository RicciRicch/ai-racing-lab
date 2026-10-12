"""Gymnasium environment for AI Racing Lab.

Importing this package registers ``AIRacing-v0``:

    import gymnasium as gym
    import envs

    env = gym.make("AIRacing-v0", tracks="train")
"""

from gymnasium.envs.registration import register

register(
    id="AIRacing-v0",
    entry_point="envs.racing_env:RacingEnv",
)
