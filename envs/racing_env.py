"""Gymnasium environment wrapping the AI Racing Lab simulation.

Observation (6 floats in [0, 1]), identical to the neuroevolution input:

    [left_far, left, front, right, right_far, speed]

Action (2 floats in [-1, 1]), identical to the raw tanh outputs of Brain:

    [steering, throttle_raw]    throttle = (throttle_raw + 1) / 2

Reward: the per-step change of the neuroevolution track fitness
(main.calculate_track_fitness), multiplied by ``reward_scale``. The
undiscounted episode return is therefore exactly
``fitness * reward_scale``, so RL and neuroevolution optimize the same
objective and their scores can be compared directly.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.env_car import EnvCar
from envs.track_geometry import TRACK_SETS, get_geometry


# ==================================================
# FITNESS
# ==================================================
#
# Same weights as calculate_track_fitness in main.py. main.py starts
# the game when imported, so the formula is repeated here.

CHECKPOINT_WEIGHT = 1000
PROGRESS_WEIGHT = 0.5
SPEED_WEIGHT = 0.05
SURVIVAL_WEIGHT = 0.01
CRASH_PENALTY = 100


def track_fitness(car):
    fitness = (
        car.checkpoints_passed * CHECKPOINT_WEIGHT
        + car.progress_reward * PROGRESS_WEIGHT
        + car.speed_reward * SPEED_WEIGHT
        + car.frames_alive * SURVIVAL_WEIGHT
    )

    if not car.alive:
        fitness -= CRASH_PENALTY

    return fitness


# ==================================================
# ENVIRONMENT
# ==================================================

class RacingEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(
        self,
        tracks="train",
        track_selection="random",
        max_steps=1800,
        randomize_start=True,
        reward_scale=0.01,
        render_mode=None
    ):
        """
        tracks: "train", "unseen", "all", or a list of Track objects.
        track_selection: "random" samples a track each reset,
            "cycle" goes through them in order. reset(options=
            {"track_index": i}) always picks track i.
        max_steps: episode length before truncation (1800 like
            MAX_FRAMES_PER_TRACK in main.py).
        randomize_start: apply the same start noise as training and
            evaluation (x +/-8 px, y +/-4 px, angle +/-4 degrees).
        """
        if isinstance(tracks, str):
            tracks = TRACK_SETS[tracks]

        if track_selection not in ("random", "cycle"):
            raise ValueError(
                "track_selection must be 'random' or 'cycle'"
            )

        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(
                f"Unsupported render_mode: {render_mode}"
            )

        self.tracks = list(tracks)
        self.track_selection = track_selection
        self.max_steps = max_steps
        self.randomize_start = randomize_start
        self.reward_scale = reward_scale
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        self.track = None
        self.track_index = None
        self.car = None
        self.steps = 0
        self.fitness = 0.0

        self._next_cycle_index = 0

    # ==================================================
    # RESET
    # ==================================================

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        options = options or {}

        if "track_index" in options:
            self.track_index = int(options["track_index"])
        elif self.track_selection == "cycle":
            self.track_index = self._next_cycle_index
            self._next_cycle_index = (
                (self._next_cycle_index + 1) % len(self.tracks)
            )
        else:
            self.track_index = int(
                self.np_random.integers(len(self.tracks))
            )

        self.track = self.tracks[self.track_index]

        x_offset = y_offset = angle_offset = 0.0

        if self.randomize_start:
            x_offset = self.np_random.uniform(-8, 8)
            y_offset = self.np_random.uniform(-4, 4)
            angle_offset = self.np_random.uniform(-4, 4)

        self.car = EnvCar(
            self.track.start_x + x_offset,
            self.track.start_y + y_offset,
            self.track.start_angle + angle_offset,
            get_geometry(self.track),
        )

        self.steps = 0
        self.fitness = 0.0

        if self.render_mode == "human":
            self.render()

        return self._observation(), self._info()

    # ==================================================
    # STEP
    # ==================================================

    def step(self, action):
        if self.car is None:
            raise RuntimeError("Call reset() before step().")

        if not self.car.alive:
            raise RuntimeError(
                "Episode has terminated. Call reset()."
            )

        action = np.clip(
            np.asarray(action, dtype=np.float64),
            -1.0,
            1.0
        )

        steering = float(action[0])
        throttle = float((action[1] + 1.0) / 2.0)

        # Same order as the game loop in main.py.
        self.car.apply_controls(steering, throttle)
        self.car.check_track_collision()

        if self.car.alive:
            self.car.update_checkpoint(self.track.checkpoints)
            self.car.update_progress_reward(self.track.checkpoints)

        self.steps += 1

        new_fitness = track_fitness(self.car)
        reward = (new_fitness - self.fitness) * self.reward_scale
        self.fitness = new_fitness

        terminated = not self.car.alive
        truncated = (not terminated) and self.steps >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            self._info(),
        )

    # ==================================================
    # HELPERS
    # ==================================================

    def _observation(self):
        return np.array(
            self.car.observation(),
            dtype=np.float32
        )

    def _info(self):
        car = self.car

        if car.frames_alive > 0:
            average_speed = car.speed_reward / car.frames_alive
        else:
            average_speed = 0.0

        return {
            "track": self.track.name,
            "track_index": self.track_index,
            "fitness": self.fitness,
            "checkpoints": car.checkpoints_passed,
            "laps": car.laps_completed,
            "frames_alive": car.frames_alive,
            "average_speed": average_speed,
            "crashed": not car.alive,
        }

    # ==================================================
    # RENDER
    # ==================================================

    def render(self):
        return None

    def close(self):
        pass
