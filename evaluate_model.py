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
START_Y = 590
START_ANGLE = -90


# ==================================================
# CHECKPOINTS
# ==================================================

checkpoints = [
    # Right
    pygame.Rect(
        760,
        320,
        20,
        60
    ),

    # Top
    pygame.Rect(
        470,
        150,
        60,
        20
    ),

    # Left
    pygame.Rect(
        220,
        320,
        20,
        60
    ),

    # Bottom
    pygame.Rect(
        470,
        520,
        60,
        20
    ),
]


# ==================================================
# PYGAME SETUP
# ==================================================

pygame.init()

screen = pygame.display.set_mode(
    (
        WIDTH,
        HEIGHT
    )
)

pygame.display.set_caption(
    "AI Racing Lab - Model Evaluation"
)

clock = pygame.time.Clock()


# ==================================================
# DRAW TRACK
# ==================================================

def draw_track():
    screen.fill(
        (40, 130, 60)
    )

    pygame.draw.ellipse(
        screen,
        (70, 70, 70),
        (
            100,
            70,
            800,
            560
        )
    )

    pygame.draw.ellipse(
        screen,
        (40, 130, 60),
        (
            280,
            200,
            440,
            300
        )
    )


# ==================================================
# CREATE EVALUATION CAR
# ==================================================

def create_evaluation_car(
    brain
):
    # Small perturbations make evaluation
    # more meaningful than repeating the exact
    # same deterministic run.

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

    car.brain = brain.copy()

    return car


# ==================================================
# RUN ONE EPISODE
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

        # Sensors inspect screen pixels.
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

        # Evaluation is intentionally faster
        # than normal demo rendering.
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
            "Train a continuous-control "
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
            f"Running episode "
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
    # AGGREGATE RESULTS
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

    # ----------------------------------------------
    # SUMMARY
    # ----------------------------------------------

    print()
    print(
        "=" * 65
    )

    print(
        "MODEL EVALUATION SUMMARY"
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

    crash_rate = (
        crash_count
        / EPISODES
        * 100
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