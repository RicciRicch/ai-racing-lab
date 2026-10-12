"""Run a neuroevolution Brain as a policy inside RacingEnv."""

import numpy as np

from brain import Brain


class BrainPolicy:
    """Maps observations to the Brain's raw tanh outputs.

    RacingEnv converts ``action[1]`` to throttle with ``(a + 1) / 2``,
    the same conversion Brain.predict applies, so the raw outputs are
    passed through unchanged.
    """

    def __init__(self, brain):
        self.brain = brain

    @classmethod
    def load(cls, filename):
        return cls(Brain.load(filename))

    def __call__(self, observation):
        inputs = np.asarray(observation, dtype=float)

        hidden = np.tanh(
            inputs @ self.brain.weights1 + self.brain.bias1
        )

        return np.tanh(
            hidden @ self.brain.weights2 + self.brain.bias2
        ).astype(np.float32)
