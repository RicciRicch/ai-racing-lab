import math

import pygame

from brain import Brain


class Car:
    def __init__(self, x, y):
        # ==================================================
        # POSITION
        # ==================================================

        self.x = x
        self.y = y

        # -90 degrees = facing right
        self.angle = -90

        # ==================================================
        # PHYSICS
        # ==================================================

        self.velocity = 0

        self.acceleration = 0.18

        # Higher than before.
        # Agent now has to learn when full speed
        # is safe.
        self.max_speed = 8.0

        self.friction = 0.035

        self.rotation_speed = 3.0

        # ==================================================
        # SIZE
        # ==================================================

        self.width = 30
        self.height = 50

        # ==================================================
        # STATE
        # ==================================================

        self.alive = True

        # ==================================================
        # BRAIN
        # ==================================================

        self.brain = Brain()

        # ==================================================
        # TRAINING DATA
        # ==================================================

        self.frames_alive = 0

        self.fitness = 0

        # Sum of normalized speeds.
        self.speed_reward = 0

        # Last neural-network outputs,
        # useful for visualization/debugging.
        self.last_steering = 0
        self.last_throttle = 0

        # ==================================================
        # CHECKPOINTS
        # ==================================================

        self.current_checkpoint = 0

        self.checkpoints_passed = 0

        self.checkpoint_locked = False

        # ==================================================
        # LAPS
        # ==================================================

        self.laps_completed = 0

        # ==================================================
        # CONTINUOUS PROGRESS
        # ==================================================

        self.previous_checkpoint_distance = None

        self.progress_reward = 0

    # ==================================================
    # NEURAL CONTROL
    # ==================================================

    def neural_control(
        self,
        screen
    ):
        sensors = self.get_sensors(
            screen
        )

        distances = [
            distance
            for distance, _
            in sensors
        ]

        # Sensor values -> [0, 1]
        sensor_values = [
            distance / 220
            for distance
            in distances
        ]

        # Speed -> [0, 1]
        speed_value = (
            self.velocity
            / self.max_speed
        )

        inputs = (
            sensor_values
            + [speed_value]
        )

        steering, throttle = (
            self.brain.predict(
                inputs
            )
        )

        self.last_steering = (
            steering
        )

        self.last_throttle = (
            throttle
        )

        # ------------------------------------------
        # THROTTLE
        # ------------------------------------------

        self.velocity += (
            throttle
            * self.acceleration
        )

        # Natural friction means the agent
        # can slow down by reducing throttle.
        self.velocity -= (
            self.friction
        )

        self.velocity = max(
            0,
            min(
                self.velocity,
                self.max_speed
            )
        )

        # ------------------------------------------
        # STEERING
        # ------------------------------------------

        if self.velocity > 0.05:
            self.angle += (
                steering
                * self.rotation_speed
            )

        # ------------------------------------------
        # MOVEMENT
        # ------------------------------------------

        self.move()

        self.frames_alive += 1

        # A small measurement of how quickly
        # the agent is moving.
        self.speed_reward += (
            self.velocity
            / self.max_speed
        )

    # ==================================================
    # MOVE
    # ==================================================

    def move(self):
        radians = math.radians(
            self.angle
        )

        self.x -= (
            math.sin(radians)
            * self.velocity
        )

        self.y -= (
            math.cos(radians)
            * self.velocity
        )

    # ==================================================
    # DRAW
    # ==================================================

    def draw(
        self,
        screen,
        color=(220, 50, 50)
    ):
        car_surface = pygame.Surface(
            (
                self.width,
                self.height
            ),
            pygame.SRCALPHA
        )

        car_surface.fill(
            color
        )

        rotated_car = (
            pygame.transform.rotate(
                car_surface,
                self.angle
            )
        )

        rect = (
            rotated_car.get_rect(
                center=(
                    int(self.x),
                    int(self.y)
                )
            )
        )

        screen.blit(
            rotated_car,
            rect
        )

    # ==================================================
    # RECT
    # ==================================================

    def get_rect(self):
        return pygame.Rect(
            int(
                self.x
                - self.width / 2
            ),
            int(
                self.y
                - self.height / 2
            ),
            self.width,
            self.height
        )

    # ==================================================
    # COLLISION
    # ==================================================

    def check_collision(
        self,
        screen
    ):
        grass_color = (
            40,
            130,
            60
        )

        check_points = [
            (0, 0),
            (0, -20),
            (0, 20),
            (-10, 0),
            (10, 0),
        ]

        radians = math.radians(
            self.angle
        )

        for (
            offset_x,
            offset_y
        ) in check_points:

            rotated_x = (
                offset_x
                * math.cos(radians)
                + offset_y
                * math.sin(radians)
            )

            rotated_y = (
                -offset_x
                * math.sin(radians)
                + offset_y
                * math.cos(radians)
            )

            x = int(
                self.x
                + rotated_x
            )

            y = int(
                self.y
                + rotated_y
            )

            if (
                x < 0
                or x >= screen.get_width()
                or y < 0
                or y >= screen.get_height()
            ):
                self.alive = False
                self.velocity = 0
                return

            pixel_color = (
                screen.get_at(
                    (x, y)
                )[:3]
            )

            if (
                pixel_color
                == grass_color
            ):
                self.alive = False
                self.velocity = 0
                return

    # ==================================================
    # SENSOR
    # ==================================================

    def cast_sensor(
        self,
        screen,
        relative_angle,
        max_distance=220
    ):
        sensor_angle = (
            self.angle
            + relative_angle
        )

        radians = math.radians(
            sensor_angle
        )

        for distance in range(
            0,
            max_distance,
            3
        ):
            x = int(
                self.x
                - math.sin(radians)
                * distance
            )

            y = int(
                self.y
                - math.cos(radians)
                * distance
            )

            if (
                x < 0
                or x >= screen.get_width()
                or y < 0
                or y >= screen.get_height()
            ):
                return (
                    distance,
                    (x, y)
                )

            pixel_color = (
                screen.get_at(
                    (x, y)
                )[:3]
            )

            if (
                pixel_color
                == (
                    40,
                    130,
                    60
                )
            ):
                return (
                    distance,
                    (x, y)
                )

        end_x = int(
            self.x
            - math.sin(radians)
            * max_distance
        )

        end_y = int(
            self.y
            - math.cos(radians)
            * max_distance
        )

        return (
            max_distance,
            (
                end_x,
                end_y
            )
        )

    # ==================================================
    # SENSORS
    # ==================================================

    def get_sensors(
        self,
        screen
    ):
        # left_far, left, front,
        # right, right_far

        angles = [
            70,
            35,
            0,
            -35,
            -70
        ]

        sensors = []

        for angle in angles:
            sensors.append(
                self.cast_sensor(
                    screen,
                    angle
                )
            )

        return sensors

    # ==================================================
    # DRAW SENSORS
    # ==================================================

    def draw_sensors(
        self,
        screen
    ):
        sensors = self.get_sensors(
            screen
        )

        for (
            distance,
            end_point
        ) in sensors:

            pygame.draw.line(
                screen,
                (
                    255,
                    255,
                    0
                ),
                (
                    int(self.x),
                    int(self.y)
                ),
                end_point,
                2
            )

            pygame.draw.circle(
                screen,
                (
                    255,
                    0,
                    0
                ),
                end_point,
                4
            )

    # ==================================================
    # CHECKPOINT
    # ==================================================

    def update_checkpoint(
        self,
        checkpoints
    ):
        checkpoint = (
            checkpoints[
                self.current_checkpoint
            ]
        )

        if (
            self.get_rect()
            .colliderect(
                checkpoint
            )
        ):
            if not (
                self.checkpoint_locked
            ):
                self.checkpoint_locked = True

                self.checkpoints_passed += 1

                self.current_checkpoint += 1

                if (
                    self.current_checkpoint
                    >= len(checkpoints)
                ):
                    self.current_checkpoint = 0

                    self.laps_completed += 1

                self.previous_checkpoint_distance = (
                    None
                )

        else:
            self.checkpoint_locked = False

    # ==================================================
    # CONTINUOUS PROGRESS
    # ==================================================

    def update_progress_reward(
        self,
        checkpoints
    ):
        checkpoint = (
            checkpoints[
                self.current_checkpoint
            ]
        )

        target_x = checkpoint.centerx
        target_y = checkpoint.centery

        distance = math.hypot(
            target_x - self.x,
            target_y - self.y
        )

        if (
            self.previous_checkpoint_distance
            is not None
        ):
            improvement = (
                self.previous_checkpoint_distance
                - distance
            )

            self.progress_reward += (
                improvement
            )

        self.previous_checkpoint_distance = (
            distance
        )