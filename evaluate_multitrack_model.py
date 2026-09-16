import os
import random
import statistics

import pygame

from brain import Brain
from car import Car
from tracks import TRAINING_TRACKS


# ==================================================
# SETTINGS
# ==================================================

WIDTH = 1000
HEIGHT = 700

MODEL_FILE = "best_brain_multitrack.npz"

EPISODES_PER_TRACK = 10

MAX_FRAMES_PER_EPISODE = 1800

RANDOM_SEED = 42


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
    "AI Racing Lab - Multi-Track Model Evaluation"
)

clock = pygame.time.Clock()


# ==================================================
# RANDOMNESS
# ==================================================

random.seed(
    RANDOM_SEED
)


# ==================================================
# DRAW CHECKPOINTS
# ==================================================

def draw_checkpoints(
    track
):
    for checkpoint in track.checkpoints:

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
# CREATE EVALUATION CAR
# ==================================================

def create_evaluation_car(
    brain,
    track
):
    # Small controlled perturbations
    # test robustness around the start state.

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
        track.start_x + x_offset,
        track.start_y + y_offset
    )

    car.angle = (
        track.start_angle
        + angle_offset
    )

    car.brain = (
        brain.copy()
    )

    return car


# ==================================================
# RUN ONE EPISODE
# ==================================================

def run_episode(
    brain,
    track,
    episode_number
):
    car = create_evaluation_car(
        brain,
        track
    )

    frames = 0

    lap_frames = []

    previous_laps = 0

    last_lap_frame = 0

    steering_sum = 0.0

    throttle_sum = 0.0

    speed_sum = 0.0

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
        #
        # Important because sensors and collision
        # read pixels directly from the surface.
        # ------------------------------------------

        track.draw(
            screen
        )

        # ------------------------------------------
        # UPDATE CAR
        # ------------------------------------------

        if car.alive:

            car.neural_control(
                screen
            )

            steering_sum += abs(
                car.last_steering
            )

            throttle_sum += (
                car.last_throttle
            )

            speed_sum += (
                car.velocity
                / car.max_speed
            )

            car.check_collision(
                screen
            )

            if car.alive:

                car.update_checkpoint(
                    track.checkpoints
                )

                car.update_progress_reward(
                    track.checkpoints
                )

        frames += 1

        # ------------------------------------------
        # LAP DETECTION
        # ------------------------------------------

        if (
            car.laps_completed
            > previous_laps
        ):

            current_lap_frames = (
                frames
                - last_lap_frame
            )

            lap_frames.append(
                current_lap_frames
            )

            last_lap_frame = (
                frames
            )

            previous_laps = (
                car.laps_completed
            )

        # ------------------------------------------
        # DRAW CHECKPOINTS AFTER SENSOR/COLLISION
        # ------------------------------------------

        draw_checkpoints(
            track
        )

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

        # Faster visual evaluation
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

        average_speed = (
            speed_sum
            / car.frames_alive
        )

        average_throttle = (
            throttle_sum
            / car.frames_alive
        )

        average_abs_steering = (
            steering_sum
            / car.frames_alive
        )

    else:

        average_speed = 0.0

        average_throttle = 0.0

        average_abs_steering = 0.0

    if lap_frames:

        average_lap_frames = (
            statistics.mean(
                lap_frames
            )
        )

    else:

        average_lap_frames = None

    result = {
        "track": track.name,
        "episode": episode_number,
        "laps": car.laps_completed,
        "checkpoints": car.checkpoints_passed,
        "frames": frames,
        "frames_alive": car.frames_alive,
        "average_speed": average_speed,
        "average_throttle": average_throttle,
        "average_abs_steering": average_abs_steering,
        "average_lap_frames": average_lap_frames,
        "crashed": not car.alive,
    }

    return result


# ==================================================
# EVALUATE ONE TRACK
# ==================================================

def evaluate_track(
    brain,
    track
):
    results = []

    print()
    print(
        "=" * 70
    )

    print(
        f"EVALUATING TRACK: "
        f"{track.name}"
    )

    print(
        "=" * 70
    )

    for episode in range(
        1,
        EPISODES_PER_TRACK + 1
    ):

        print()
        print(
            f"Episode "
            f"{episode}/"
            f"{EPISODES_PER_TRACK}"
        )

        result = run_episode(
            brain,
            track,
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
            f"Average speed: "
            f"{result['average_speed']:.3f}"
        )

        print(
            f"Average throttle: "
            f"{result['average_throttle']:.3f}"
        )

        print(
            f"Average |steering|: "
            f"{result['average_abs_steering']:.3f}"
        )

        print(
            f"Crashed: "
            f"{result['crashed']}"
        )

    return results


# ==================================================
# TRACK SUMMARY
# ==================================================

def print_track_summary(
    track,
    results
):
    laps = [
        result["laps"]
        for result in results
    ]

    checkpoints = [
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

    throttles = [
        result["average_throttle"]
        for result in results
    ]

    steering_values = [
        result["average_abs_steering"]
        for result in results
    ]

    crashes = sum(
        1
        for result in results
        if result["crashed"]
    )

    crash_rate = (
        crashes
        / len(results)
        * 100
    )

    lap_times = [
        result["average_lap_frames"]
        for result in results
        if result["average_lap_frames"]
        is not None
    ]

    print()
    print(
        "-" * 70
    )

    print(
        f"TRACK SUMMARY: "
        f"{track.name}"
    )

    print(
        "-" * 70
    )

    print(
        f"Episodes: "
        f"{len(results)}"
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
        f"{statistics.mean(checkpoints):.2f}"
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
        f"Average throttle: "
        f"{statistics.mean(throttles):.3f}"
    )

    print(
        f"Average |steering|: "
        f"{statistics.mean(steering_values):.3f}"
    )

    if lap_times:

        print(
            f"Average lap duration: "
            f"{statistics.mean(lap_times):.1f} frames"
        )

    else:

        print(
            "Average lap duration: N/A"
        )

    print(
        f"Crash count: "
        f"{crashes}/{len(results)}"
    )

    print(
        f"Crash rate: "
        f"{crash_rate:.1f}%"
    )

    print(
        "-" * 70
    )


# ==================================================
# GLOBAL SUMMARY
# ==================================================

def print_global_summary(
    all_results
):
    all_laps = [
        result["laps"]
        for result in all_results
    ]

    all_checkpoints = [
        result["checkpoints"]
        for result in all_results
    ]

    all_speeds = [
        result["average_speed"]
        for result in all_results
    ]

    all_throttles = [
        result["average_throttle"]
        for result in all_results
    ]

    all_steering = [
        result["average_abs_steering"]
        for result in all_results
    ]

    crashes = sum(
        1
        for result in all_results
        if result["crashed"]
    )

    crash_rate = (
        crashes
        / len(all_results)
        * 100
    )

    print()
    print(
        "=" * 70
    )

    print(
        "MULTI-TRACK MODEL EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Model: "
        f"{MODEL_FILE}"
    )

    print(
        f"Tracks evaluated: "
        f"{len(TRAINING_TRACKS)}"
    )

    print(
        f"Total episodes: "
        f"{len(all_results)}"
    )

    print(
        f"Average laps per episode: "
        f"{statistics.mean(all_laps):.2f}"
    )

    print(
        f"Average checkpoints per episode: "
        f"{statistics.mean(all_checkpoints):.2f}"
    )

    print(
        f"Average normalized speed: "
        f"{statistics.mean(all_speeds):.3f}"
    )

    print(
        f"Average throttle: "
        f"{statistics.mean(all_throttles):.3f}"
    )

    print(
        f"Average |steering|: "
        f"{statistics.mean(all_steering):.3f}"
    )

    print(
        f"Total crashes: "
        f"{crashes}/{len(all_results)}"
    )

    print(
        f"Overall crash rate: "
        f"{crash_rate:.1f}%"
    )

    print()
    print(
        "PER-TRACK RESULTS"
    )

    for track in TRAINING_TRACKS:

        track_results = [
            result
            for result in all_results
            if result["track"]
            == track.name
        ]

        average_laps = (
            statistics.mean(
                result["laps"]
                for result
                in track_results
            )
        )

        track_crashes = sum(
            1
            for result
            in track_results
            if result["crashed"]
        )

        track_crash_rate = (
            track_crashes
            / len(track_results)
            * 100
        )

        print(
            f"{track.name}: "
            f"avg laps = "
            f"{average_laps:.2f}, "
            f"crash rate = "
            f"{track_crash_rate:.1f}%"
        )

    print(
        "=" * 70
    )


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
            "Run multi-track training first."
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

    all_results = []

    # ----------------------------------------------
    # EVALUATE EVERY TRAINING TRACK
    # ----------------------------------------------

    for track in TRAINING_TRACKS:

        results = evaluate_track(
            brain,
            track
        )

        all_results.extend(
            results
        )

        print_track_summary(
            track,
            results
        )

    # ----------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------

    print_global_summary(
        all_results
    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    main()

    pygame.quit()