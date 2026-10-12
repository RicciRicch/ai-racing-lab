"""Car controlled by external actions instead of an internal Brain.

``EnvCar`` extends the original ``Car`` so checkpoints, laps, progress
tracking and drawing stay shared with the neuroevolution code. Only two
things change:

1. Controls come from outside (an RL policy) through ``apply_controls``.
2. Sensors and collisions query a ``TrackGeometry`` mask instead of
   reading screen pixels.

Both are written to reproduce the original arithmetic exactly, and
tests/test_env_car_parity.py checks that a car driven through
``Car.neural_control`` and an ``EnvCar`` given the same controls follow
bit-identical trajectories.
"""

import math

import numpy as np

from car import Car


# Same values as Car.get_sensors / Car.cast_sensor / Car.check_collision.
SENSOR_ANGLES = (70, 35, 0, -35, -70)
SENSOR_RANGE = 220
SENSOR_STEP = 3

COLLISION_POINTS = (
    (0, 0),
    (0, -20),
    (0, 20),
    (-10, 0),
    (10, 0),
)

_SENSOR_SAMPLES = np.arange(
    0,
    SENSOR_RANGE,
    SENSOR_STEP
)


class EnvCar(Car):
    def __init__(
        self,
        x,
        y,
        angle,
        geometry
    ):
        super().__init__(x, y)

        self.angle = angle

        self.geometry = geometry

        # Actions come from the environment, not from a Brain.
        self.brain = None

    # ==================================================
    # CONTROLS
    # ==================================================

    def apply_controls(
        self,
        steering,
        throttle
    ):
        """Physics part of Car.neural_control, with controls given."""
        self.last_steering = steering
        self.last_throttle = throttle

        self.velocity += throttle * self.acceleration

        self.velocity -= self.friction

        self.velocity = max(
            0,
            min(
                self.velocity,
                self.max_speed
            )
        )

        if self.velocity > 0.05:
            self.angle += steering * self.rotation_speed

        self.move()

        self.frames_alive += 1

        self.speed_reward += self.velocity / self.max_speed

    # ==================================================
    # SENSORS
    # ==================================================

    def sensor_distances(self):
        """Distances for left_far, left, front, right, right_far."""
        distances = []

        for relative_angle in SENSOR_ANGLES:
            radians = math.radians(
                self.angle + relative_angle
            )

            # Same expression as Car.cast_sensor, evaluated for all
            # samples at once. astype(int) truncates like int().
            xs = (
                self.x - math.sin(radians) * _SENSOR_SAMPLES
            ).astype(int)

            ys = (
                self.y - math.cos(radians) * _SENSOR_SAMPLES
            ).astype(int)

            blocked = self.geometry.blocked_array(xs, ys)

            if blocked.any():
                distances.append(
                    int(_SENSOR_SAMPLES[blocked.argmax()])
                )
            else:
                distances.append(SENSOR_RANGE)

        return distances

    def observation(self):
        """Normalized inputs in the same order the Brain receives them."""
        values = [
            distance / SENSOR_RANGE
            for distance in self.sensor_distances()
        ]

        values.append(
            self.velocity / self.max_speed
        )

        return values

    def sensor_endpoints(self):
        """End points of each ray, for rendering."""
        endpoints = []

        for relative_angle, distance in zip(
            SENSOR_ANGLES,
            self.sensor_distances()
        ):
            radians = math.radians(
                self.angle + relative_angle
            )

            endpoints.append((
                int(self.x - math.sin(radians) * distance),
                int(self.y - math.cos(radians) * distance),
            ))

        return endpoints

    # ==================================================
    # COLLISION
    # ==================================================

    def check_track_collision(self):
        """Car.check_collision using the geometry mask."""
        radians = math.radians(
            self.angle
        )

        for offset_x, offset_y in COLLISION_POINTS:
            rotated_x = (
                offset_x * math.cos(radians)
                + offset_y * math.sin(radians)
            )

            rotated_y = (
                -offset_x * math.sin(radians)
                + offset_y * math.cos(radians)
            )

            x = int(self.x + rotated_x)
            y = int(self.y + rotated_y)

            if self.geometry.is_blocked(x, y):
                self.alive = False
                self.velocity = 0
                return
