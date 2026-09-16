import os
import random
import statistics

import pygame

from brain import Brain
from car import Car


# ==================================================
# SETTINGS
# ==================================================

WIDTH = 1000
HEIGHT = 700

MODEL_FILE = "best_brain_continuous.npz"

EPISODES = 10

MAX_FRAMES_PER_EPISODE = 1800

START_X = 500
START_Y = 600
START_ANGLE = -90


# ==================================================
# PYGAME
# ==================================================

pygame.init()

screen = pygame.display.set_mode(
    (
        WIDTH,
        HEIGHT
    )
)

pygame.display.set_caption(
    "AI Racing Lab - Unseen Track Evaluation"
)

clock = pygame.time.Clock()


# ==================================================
# TRACK COLORS
# ==================================================

GRASS_COLOR = (
    40,
    130,
    60
)

ROAD_COLOR = (
    70,
    70,
    70
)


# ==================================================
# TRACK 2 CHECKPOINTS
# ==================================================
#
# Clockwise:
#
# bottom-right
# right
# top-right
# top-left
# left
# bottom-left
#

checkpoints = [
    pygame.Rect(
        650,
        500,
        50,
        20
    ),

    pygame.Rect(
        800,
        310,
        20,
        60
    ),

    pygame.Rect(
        650,
        150,
        50,
        20
    ),

    pygame.Rect(
        300,
        150,
        50,
        20
    ),

    pygame.Rect(
        180,
        310,
        20,
        60
    ),

    pygame.Rect(
        300,
        500,
        50,
        20
    ),
]


# ==================================================
# DRAW UNSEEN TRACK
# ==================================================

def draw_track():
    screen.fill(
        GRASS_COLOR
    )

    # Outer road shape
    pygame.draw.ellipse(
        screen,
        ROAD_COLOR,
        (
            110,
            90,
            780,
            520
        )
    )

    # Inner grass creates a narrower road
    pygame.draw.ellipse(
        screen,
        GRASS_COLOR,
        (
            330,
            215,
            340,
            270
        )
    )

    # Add side narrowing sections
    pygame.draw.rect(
        screen,
        GRASS_COLOR,
        (
            110,
            250,
            90,
            200
        )
    )

    pygame.draw.rect(
        screen,
        GRASS_COLOR,
        (
            800,
            250,
            90,
            200
        )
    )


# ==================================================
# DRAW CHECKPOINTS
# ==================================================

def draw_checkpoints():
    for checkpoint in checkpoints:
        pygame.draw.rect(
            screen,
            (
                0,
                200,
                255
            ),
            checkpoint,
            2
        )


# ==================================================
# CREATE CAR
# ==================================================

def create_evaluation_car(
    brain
):
    x_offset = random.uniform(
        -8,
        8
    )

    y_offset = random.uniform(
        -4,
        4
    )

    angle_offset = random.uniform(
        -4,
        4
    )

    car = Car(
        START_X + x_offset,
        START_Y + y_offset
    )

    car.angle = (
        START_ANGLE
        + angle_offset
    )

    car.brain = (
        brain.copy()
    )

    return car


# ==================================================
# RUN EPISODE
# ==================================================

def run_episode(
    brain,
    episode_number
):
    car = create_evaluation_car(
        brain
    )

    frames = 0

    running_episode = True

    while running_episode:

        # ------------------------------------------
        # EVENTS
        # ------------------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # ------------------------------------------
        # DRAW TRACK FIRST
        # ------------------------------------------

        draw_track()

        # ------------------------------------------
        # UPDATE CAR
        # ------------------------------------------

        if car.alive:

            car.neural_control(
                screen
            )

            car.check_collision(
                screen
            )

            if car.alive:

                car.update_checkpoint(
                    checkpoints
                )

                car.update_progress_reward(
                    checkpoints
                )

        frames += 1

        # ------------------------------------------
        # DRAW CHECKPOINTS
        # ------------------------------------------

        draw_checkpoints()

        # ------------------------------------------
        # DRAW CAR
        # ------------------------------------------

        if car.alive:

            car.draw(
                screen,
                (
                    255,
                    220,
                    0
                )
            )

            car.draw_sensors(
                screen
            )

        pygame.display.flip()

        # Run faster than normal demo
        clock.tick(
            240
        )

        # ------------------------------------------
        # STOP CONDITIONS
        # ------------------------------------------

        if not car.alive:
            running_episode = False

        if (
            frames
            >= MAX_FRAMES_PER_EPISODE
        ):
            running_episode = False

    # ==================================================
    # METRICS
    # ==================================================

    if car.frames_alive > 0:

        average_normalized_speed = (
            car.speed_reward
            / car.frames_alive
        )

    else:

        average_normalized_speed = 0

    crashed = (
        not car.alive
    )

    result = {
        "episode": episode_number,
        "laps": car.laps_completed,
        "checkpoints": car.checkpoints_passed,
        "frames": frames,
        "frames_alive": car.frames_alive,
        "average_speed": average_normalized_speed,
        "crashed": crashed,
    }

    return result


# ==================================================
# MAIN
# ==================================================

def main():

    # ----------------------------------------------
    # CHECK MODEL
    # ----------------------------------------------

    if not os.path.exists(
        MODEL_FILE
    ):
        print(
            f"Model not found: "
            f"{MODEL_FILE}"
        )

        print(
            "Train the continuous-control "
            "model first."
        )

        return

    # ----------------------------------------------
    # LOAD MODEL
    # ----------------------------------------------

    print(
        f"Loading model: "
        f"{MODEL_FILE}"
    )

    print(
        "Evaluating on UNSEEN TRACK."
    )

    brain = Brain.load(
        MODEL_FILE
    )

    results = []

    # ----------------------------------------------
    # RUN EPISODES
    # ----------------------------------------------

    for episode in range(
        1,
        EPISODES + 1
    ):

        print()
        print(
            f"Running unseen-track episode "
            f"{episode}/{EPISODES}..."
        )

        result = run_episode(
            brain,
            episode
        )

        results.append(
            result
        )

        print(
            f"Laps: "
            f"{result['laps']}"
        )

        print(
            f"Checkpoints: "
            f"{result['checkpoints']}"
        )

        print(
            f"Frames: "
            f"{result['frames']}"
        )

        print(
            f"Average normalized speed: "
            f"{result['average_speed']:.3f}"
        )

        print(
            f"Crashed: "
            f"{result['crashed']}"
        )

    # ==================================================
    # AGGREGATE
    # ==================================================

    laps = [
        result["laps"]
        for result in results
    ]

    checkpoints_passed = [
        result["checkpoints"]
        for result in results
    ]

    frames = [
        result["frames"]
        for result in results
    ]

    speeds = [
        result["average_speed"]
        for result in results
    ]

    crash_count = sum(
        1
        for result in results
        if result["crashed"]
    )

    crash_rate = (
        crash_count
        / EPISODES
        * 100
    )

    # ==================================================
    # SUMMARY
    # ==================================================

    print()
    print(
        "=" * 65
    )

    print(
        "UNSEEN TRACK EVALUATION SUMMARY"
    )

    print(
        "=" * 65
    )

    print(
        f"Episodes: "
        f"{EPISODES}"
    )

    print(
        f"Average laps: "
        f"{statistics.mean(laps):.2f}"
    )

    print(
        f"Best laps: "
        f"{max(laps)}"
    )

    print(
        f"Worst laps: "
        f"{min(laps)}"
    )

    print(
        f"Average checkpoints: "
        f"{statistics.mean(checkpoints_passed):.2f}"
    )

    print(
        f"Average frames: "
        f"{statistics.mean(frames):.2f}"
    )

    print(
        f"Average normalized speed: "
        f"{statistics.mean(speeds):.3f}"
    )

    print(
        f"Crash count: "
        f"{crash_count}/{EPISODES}"
    )

    print(
        f"Crash rate: "
        f"{crash_rate:.1f}%"
    )

    print(
        "=" * 65
    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    main()

    pygame.quit()