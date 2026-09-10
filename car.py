import math
import pygame


class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.angle = -90
        self.velocity = 0

        self.acceleration = 0.12
        self.max_speed = 5.5
        self.friction = 0.03
        self.rotation_speed = 2.5

        self.width = 30
        self.height = 50

        self.alive = True

    def update(self, keys):
        if keys[pygame.K_UP]:
            self.velocity += self.acceleration

        if keys[pygame.K_DOWN]:
            self.velocity -= self.acceleration

        self.velocity = max(
            -self.max_speed / 2,
            min(self.velocity, self.max_speed)
        )

        if abs(self.velocity) > 0.1:
            direction = 1 if self.velocity > 0 else -1

            if keys[pygame.K_LEFT]:
                self.angle += self.rotation_speed * direction

            if keys[pygame.K_RIGHT]:
                self.angle -= self.rotation_speed * direction

        if self.velocity > 0:
            self.velocity -= self.friction

        elif self.velocity < 0:
            self.velocity += self.friction

        if abs(self.velocity) < self.friction:
            self.velocity = 0

        radians = math.radians(self.angle)

        self.x -= math.sin(radians) * self.velocity
        self.y -= math.cos(radians) * self.velocity

    def autonomous_control(self, screen):
        sensors = self.get_sensors(screen)

        distances = [
            distance
            for distance, _ in sensors
        ]

        left_far, left, front, right, right_far = distances

        # Gas
        self.velocity += self.acceleration

        # Nemoj odmah maksimalnom brzinom
        target_speed = 4.0

        if self.velocity > target_speed:
            self.velocity = target_speed

        # Ako je ispred tesno, uspori
        if front < 90:
            self.velocity *= 0.97

        # --------------------------------------------------
        # STEERING
        # --------------------------------------------------

        # Ako je prepreka direktno ispred,
        # biraj otvoreniju stranu
        if front < 65:

            if left_far > right_far:
                # Turn left
                self.angle += self.rotation_speed

            else:
                # Turn right
                self.angle -= self.rotation_speed

        else:

            # Pokušaj da bude približno
            # u sredini puta
            difference = left - right

            if difference > 18:
                # Više prostora levo -> idi malo levo
                self.angle += self.rotation_speed * 0.45

            elif difference < -18:
                # Više prostora desno -> idi malo desno
                self.angle -= self.rotation_speed * 0.45

        # Hitna korekcija od ivice
        if left < 30:
            self.angle -= self.rotation_speed * 0.8

        if right < 30:
            self.angle += self.rotation_speed * 0.8

        # --------------------------------------------------
        # MOVEMENT
        # --------------------------------------------------

        radians = math.radians(self.angle)

        self.x -= math.sin(radians) * self.velocity
        self.y -= math.cos(radians) * self.velocity

    def draw(self, screen):
        car_surface = pygame.Surface(
            (self.width, self.height),
            pygame.SRCALPHA
        )

        car_surface.fill((220, 50, 50))

        rotated_car = pygame.transform.rotate(
            car_surface,
            self.angle
        )

        rect = rotated_car.get_rect(
            center=(int(self.x), int(self.y))
        )

        screen.blit(rotated_car, rect)

    def get_rect(self):
        return pygame.Rect(
            int(self.x - self.width / 2),
            int(self.y - self.height / 2),
            self.width,
            self.height
        )

    def check_collision(self, screen):
        car_x = int(self.x)
        car_y = int(self.y)

        if (
            car_x < 0
            or car_x >= screen.get_width()
            or car_y < 0
            or car_y >= screen.get_height()
        ):
            self.alive = False
            self.velocity = 0
            return

        grass_color = (40, 130, 60)

        # Proveravamo više tačaka oko auta,
        # ne samo njegov centar
        check_points = [
            (0, 0),
            (0, -20),
            (0, 20),
            (-10, 0),
            (10, 0),
        ]

        radians = math.radians(self.angle)

        for offset_x, offset_y in check_points:
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

            if (
                x < 0
                or x >= screen.get_width()
                or y < 0
                or y >= screen.get_height()
            ):
                self.alive = False
                self.velocity = 0
                return

            pixel_color = screen.get_at((x, y))[:3]

            if pixel_color == grass_color:
                self.alive = False
                self.velocity = 0
                return

    def cast_sensor(
        self,
        screen,
        relative_angle,
        max_distance=220
    ):
        sensor_angle = self.angle + relative_angle
        radians = math.radians(sensor_angle)

        for distance in range(
            0,
            max_distance,
            3
        ):
            x = int(
                self.x
                - math.sin(radians) * distance
            )

            y = int(
                self.y
                - math.cos(radians) * distance
            )

            if (
                x < 0
                or x >= screen.get_width()
                or y < 0
                or y >= screen.get_height()
            ):
                return distance, (x, y)

            pixel_color = screen.get_at((x, y))[:3]

            if pixel_color == (40, 130, 60):
                return distance, (x, y)

        end_x = int(
            self.x
            - math.sin(radians) * max_distance
        )

        end_y = int(
            self.y
            - math.cos(radians) * max_distance
        )

        return max_distance, (
            end_x,
            end_y
        )

    def get_sensors(self, screen):
        # Redosled:
        # left_far, left, front, right, right_far
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

    def draw_sensors(self, screen):
        sensors = self.get_sensors(screen)

        for distance, end_point in sensors:
            pygame.draw.line(
                screen,
                (255, 255, 0),
                (int(self.x), int(self.y)),
                end_point,
                2
            )

            pygame.draw.circle(
                screen,
                (255, 0, 0),
                end_point,
                4
            )
    