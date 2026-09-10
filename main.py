import time
import pygame

from car import Car


# --------------------------------------------------
# SETUP
# --------------------------------------------------

pygame.init()

WIDTH = 1000
HEIGHT = 700

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "AI Racing Lab"
)

clock = pygame.time.Clock()

font = pygame.font.SysFont(
    "Arial",
    24
)


# --------------------------------------------------
# CAR
# --------------------------------------------------

START_X = 500
START_Y = 590

car = Car(
    START_X,
    START_Y
)

car.angle = -90


# --------------------------------------------------
# GAME MODE
# --------------------------------------------------

autonomous_mode = True


# --------------------------------------------------
# CHECKPOINTS
# --------------------------------------------------

checkpoints = [
    pygame.Rect(
        470,
        520,
        60,
        20
    ),

    pygame.Rect(
        760,
        320,
        20,
        60
    ),

    pygame.Rect(
        470,
        150,
        60,
        20
    ),

    pygame.Rect(
        220,
        320,
        20,
        60
    ),
]

current_checkpoint = 0
checkpoint_locked = False

lap_count = 0
lap_start_time = time.time()
best_lap = None


# --------------------------------------------------
# RESET FUNCTION
# --------------------------------------------------

def reset_game():
    global car
    global current_checkpoint
    global checkpoint_locked
    global lap_count
    global lap_start_time

    car = Car(
        START_X,
        START_Y
    )

    car.angle = -90

    current_checkpoint = 0
    checkpoint_locked = False

    lap_count = 0
    lap_start_time = time.time()


# --------------------------------------------------
# GAME LOOP
# --------------------------------------------------

running = True

while running:

    # --------------------------------------------------
    # EVENTS
    # --------------------------------------------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            # Reset
            if event.key == pygame.K_r:
                reset_game()

            # Human / AI mode
            if event.key == pygame.K_SPACE:
                autonomous_mode = (
                    not autonomous_mode
                )


    # --------------------------------------------------
    # INPUT
    # --------------------------------------------------

    keys = pygame.key.get_pressed()


    # --------------------------------------------------
    # DRAW WORLD
    # --------------------------------------------------

    # Grass
    screen.fill(
        (40, 130, 60)
    )

    # Outer road
    pygame.draw.ellipse(
        screen,
        (70, 70, 70),
        (100, 70, 800, 560)
    )

    # Inner grass
    pygame.draw.ellipse(
        screen,
        (40, 130, 60),
        (280, 200, 440, 300)
    )


    # --------------------------------------------------
    # UPDATE CAR
    # --------------------------------------------------

    if car.alive:

        if autonomous_mode:
            car.autonomous_control(
                screen
            )

        else:
            car.update(
                keys
            )


    # --------------------------------------------------
    # COLLISION
    # --------------------------------------------------

    car.check_collision(
        screen
    )


    # --------------------------------------------------
    # CHECKPOINT LOGIC
    # --------------------------------------------------

    if car.alive:

        car_rect = car.get_rect()

        checkpoint = checkpoints[
            current_checkpoint
        ]

        if car_rect.colliderect(
            checkpoint
        ):

            if not checkpoint_locked:

                checkpoint_locked = True

                current_checkpoint += 1

                # Lap completed
                if (
                    current_checkpoint
                    >= len(checkpoints)
                ):

                    current_checkpoint = 0

                    lap_count += 1

                    lap_time = (
                        time.time()
                        - lap_start_time
                    )

                    lap_start_time = (
                        time.time()
                    )

                    if (
                        best_lap is None
                        or lap_time < best_lap
                    ):
                        best_lap = lap_time

                    print(
                        f"Lap {lap_count}: "
                        f"{lap_time:.2f}s"
                    )

        else:
            checkpoint_locked = False


    # --------------------------------------------------
    # DRAW CHECKPOINTS
    # --------------------------------------------------

    for i, checkpoint in enumerate(
        checkpoints
    ):

        color = (
            0,
            200,
            255
        )

        if i == current_checkpoint:
            color = (
                255,
                200,
                0
            )

        pygame.draw.rect(
            screen,
            color,
            checkpoint,
            3
        )


    # --------------------------------------------------
    # DRAW CAR
    # --------------------------------------------------

    car.draw(
        screen
    )

    if car.alive:
        car.draw_sensors(
            screen
        )


    # --------------------------------------------------
    # HUD
    # --------------------------------------------------

    current_time = (
        time.time()
        - lap_start_time
    )

    lap_text = font.render(
        f"Lap: {lap_count}",
        True,
        (255, 255, 255)
    )

    time_text = font.render(
        f"Time: {current_time:.2f}s",
        True,
        (255, 255, 255)
    )

    if best_lap is None:
        best_string = "Best: --"

    else:
        best_string = (
            f"Best: {best_lap:.2f}s"
        )

    best_text = font.render(
        best_string,
        True,
        (255, 255, 255)
    )

    checkpoint_text = font.render(
        (
            f"Checkpoint: "
            f"{current_checkpoint + 1}"
            f"/{len(checkpoints)}"
        ),
        True,
        (255, 255, 255)
    )

    mode = (
        "AI"
        if autonomous_mode
        else "HUMAN"
    )

    mode_text = font.render(
        f"Mode: {mode}",
        True,
        (255, 255, 255)
    )

    controls_text = font.render(
        "SPACE: Human/AI | R: Reset",
        True,
        (255, 255, 255)
    )


    # HUD position
    screen.blit(
        lap_text,
        (20, 20)
    )

    screen.blit(
        time_text,
        (20, 50)
    )

    screen.blit(
        best_text,
        (20, 80)
    )

    screen.blit(
        checkpoint_text,
        (20, 110)
    )

    screen.blit(
        mode_text,
        (20, 140)
    )

    screen.blit(
        controls_text,
        (20, 170)
    )


    # --------------------------------------------------
    # CRASH MESSAGE
    # --------------------------------------------------

    if not car.alive:

        crash_text = font.render(
            "CRASHED - Press R to restart",
            True,
            (255, 100, 100)
        )

        screen.blit(
            crash_text,
            (
                WIDTH // 2 - 160,
                20
            )
        )


    # --------------------------------------------------
    # UPDATE SCREEN
    # --------------------------------------------------

    pygame.display.flip()

    clock.tick(60)


# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

pygame.quit()