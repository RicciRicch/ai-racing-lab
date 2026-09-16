import csv
import os
import random
import statistics
import time

import pygame

from brain import Brain
from car import Car

from tracks import TRAINING_TRACKS


# ==================================================
# SETTINGS
# ==================================================

WIDTH = 1000
HEIGHT = 700

FPS = 60

POPULATION_SIZE = 40

PARENT_COUNT = 5
ELITE_COUNT = 2

MAX_FRAMES_PER_TRACK = 1800


# ==================================================
# FILES
# ==================================================

# New model trained across multiple tracks
MODEL_FILE = "best_brain_multitrack.npz"

# Existing single-track continuous model.
# We can use it to initialize training.
SEED_MODEL_FILE = "best_brain_continuous.npz"

HISTORY_FILE = "training_history_multitrack.csv"


# ==================================================
# MODES
# ==================================================

MODE_TRAIN = "TRAIN"
MODE_DEMO = "DEMO"


# ==================================================
# ADAPTIVE EVOLUTION
# ==================================================

STAGNATION_LEVEL_1 = 8
STAGNATION_LEVEL_2 = 18

MIN_IMPROVEMENT = 1.0


# ==================================================
# PYGAME
# ==================================================

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("AI Racing Lab - Multi-Track Training")

clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 24)

small_font = pygame.font.SysFont("Arial", 18)


# ==================================================
# CANDIDATE
# ==================================================


class Candidate:
    """
    One neural network being evaluated.

    The same brain is tested on every
    training track during the generation.
    """

    def __init__(self, brain=None):
        if brain is None:
            self.brain = Brain()

        else:
            self.brain = brain

        self.track_scores = []

        self.track_laps = []

        self.track_checkpoints = []

        self.track_crashes = []

    @property
    def fitness(self):
        if not self.track_scores:
            return 0

        return min(
            self.track_scores
        )

    @property
    def mean_track_score(self):
        if not self.track_scores:
            return 0

        return statistics.mean(self.track_scores)

    @property
    def worst_track_score(self):
        if not self.track_scores:
            return 0

        return min(self.track_scores)

    @property
    def total_laps(self):
        return sum(self.track_laps)

    @property
    def total_checkpoints(self):
        return sum(self.track_checkpoints)


# ==================================================
# GLOBAL TRAINING STATE
# ==================================================

mode = MODE_TRAIN

generation = 1

current_track_index = 0

track_frame = 0

generations_without_improvement = 0

best_fitness_ever = float("-inf")

best_brain_ever = None


# Current population of neural networks
candidates = []

# Cars currently evaluating those brains
cars = []


# ==================================================
# DEMO STATE
# ==================================================

demo_car = None

demo_track_index = 0

demo_lap_start_time = None

demo_best_lap = None

demo_previous_laps = 0


# ==================================================
# TRAINING HISTORY
# ==================================================


def initialize_history_file():

    if os.path.exists(HISTORY_FILE):
        return

    with open(HISTORY_FILE, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "generation",
                "generalization_fitness",
                "mean_track_score",
                "worst_track_score",
                "population_average_fitness",
                "track_1_score",
                "track_2_score",
                "track_3_score",
                "total_laps",
                "total_checkpoints",
                "stagnation",
                "mutation_mode",
            ]
        )


def save_generation_stats(champion, average_fitness, mutation_mode):
    scores = champion.track_scores + [0, 0, 0]

    with open(HISTORY_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                generation,
                champion.fitness,
                champion.mean_track_score,
                champion.worst_track_score,
                average_fitness,
                scores[0],
                scores[1],
                scores[2],
                champion.total_laps,
                champion.total_checkpoints,
                generations_without_improvement,
                mutation_mode,
            ]
        )


initialize_history_file()


# ==================================================
# FITNESS
# ==================================================


def calculate_track_fitness(car):
    checkpoint_reward = car.checkpoints_passed * 1000

    progress_reward = car.progress_reward * 0.5

    speed_reward = car.speed_reward * 0.05

    survival_reward = car.frames_alive * 0.01

    crash_penalty = 0

    if not car.alive:
        crash_penalty = 100

    return (
        checkpoint_reward
        + progress_reward
        + speed_reward
        + survival_reward
        - crash_penalty
    )


# ==================================================
# CREATE CAR FOR TRACK
# ==================================================


def create_track_car(brain, track, x_offset, y_offset, angle_offset):
    car = Car(track.start_x + x_offset, track.start_y + y_offset)

    car.angle = track.start_angle + angle_offset

    car.brain = brain.copy()

    return car


# ==================================================
# CREATE POPULATION
# ==================================================


def create_initial_population():
    population = []

    # ----------------------------------------------
    # Existing multi-track model
    # ----------------------------------------------

    if os.path.exists(MODEL_FILE):
        print(f"Loading multi-track model: {MODEL_FILE}")

        seed_brain = Brain.load(MODEL_FILE)

    # ----------------------------------------------
    # Transfer learning from our previous model
    # ----------------------------------------------

    elif os.path.exists(SEED_MODEL_FILE):
        print("No multi-track model found.")

        print("Using previous continuous model as training seed:")

        print(SEED_MODEL_FILE)

        seed_brain = Brain.load(SEED_MODEL_FILE)

    # ----------------------------------------------
    # Fully random
    # ----------------------------------------------

    else:
        print("No trained seed found.")

        print("Starting completely random.")

        return [Candidate() for _ in range(POPULATION_SIZE)]

    # ----------------------------------------------
    # Exact seed
    # ----------------------------------------------

    population.append(Candidate(seed_brain.copy()))

    # ----------------------------------------------
    # Mutated seed population
    # ----------------------------------------------

    while len(population) < POPULATION_SIZE:
        brain = seed_brain.copy()

        brain.mutate(random.choice([0.01, 0.03, 0.05, 0.10, 0.15]))

        population.append(Candidate(brain))

    return population


# ==================================================
# COMMON PERTURBATION
# ==================================================


def generate_start_perturbation():
    """
    Every agent on this track gets the SAME
    perturbation.

    This preserves fair comparison between
    neural networks while still forcing
    robustness across generations.
    """

    return (random.uniform(-8, 8), random.uniform(-4, 4), random.uniform(-4, 4))


# ==================================================
# START TRACK EVALUATION
# ==================================================


def start_track_evaluation():
    global cars
    global track_frame

    track = TRAINING_TRACKS[current_track_index]

    (x_offset, y_offset, angle_offset) = generate_start_perturbation()

    cars = []

    for candidate in candidates:
        car = create_track_car(candidate.brain, track, x_offset, y_offset, angle_offset)

        cars.append(car)

    track_frame = 0

    print()
    print(
        f"Generation {generation} | "
        f"Track "
        f"{current_track_index + 1}"
        f"/{len(TRAINING_TRACKS)}"
        f" | {track.name}"
    )


# ==================================================
# ADAPTIVE EVOLUTION
# ==================================================


def get_evolution_settings():

    if generations_without_improvement < STAGNATION_LEVEL_1:
        return ("FINE", 0.01, 0.03, 0.08, 2)

    if generations_without_improvement < STAGNATION_LEVEL_2:
        return ("EXPLORE", 0.03, 0.08, 0.20, 5)

    return ("ESCAPE", 0.05, 0.15, 0.35, 10)


# ==================================================
# CREATE CHILD
# ==================================================


def create_child(parents, mutation_rate):
    parent_a = random.choice(parents)

    parent_b = random.choice(parents)

    child_brain = Brain.crossover(parent_a.brain, parent_b.brain)

    child_brain.mutate(mutation_rate)

    return Candidate(child_brain)


# ==================================================
# FINISH CURRENT TRACK
# ==================================================


def finish_current_track():
    global current_track_index

    track = TRAINING_TRACKS[current_track_index]

    # ----------------------------------------------
    # Save score of every candidate
    # ----------------------------------------------

    for candidate, car in zip(candidates, cars):
        score = calculate_track_fitness(car)

        candidate.track_scores.append(score)

        candidate.track_laps.append(car.laps_completed)

        candidate.track_checkpoints.append(car.checkpoints_passed)

        candidate.track_crashes.append(not car.alive)

    # ----------------------------------------------
    # Track summary
    # ----------------------------------------------

    best_car = max(cars, key=calculate_track_fitness)

    print(f"Finished {track.name}")

    print(f"Best track score: {calculate_track_fitness(best_car):.2f}")

    print(f"Best laps: {best_car.laps_completed}")

    # ----------------------------------------------
    # Next track
    # ----------------------------------------------

    current_track_index += 1

    if current_track_index < len(TRAINING_TRACKS):
        start_track_evaluation()

    else:
        finish_generation()


# ==================================================
# FINISH GENERATION
# ==================================================


def finish_generation():
    global candidates

    global generation
    global current_track_index

    global best_fitness_ever
    global best_brain_ever

    global generations_without_improvement

    # ----------------------------------------------
    # RANK BY MULTI-TRACK FITNESS
    # ----------------------------------------------

    ranked = sorted(
    candidates,
    key=lambda candidate: (
        candidate.worst_track_score,
        candidate.mean_track_score
    ),
    reverse=True
)

    champion = ranked[0]

    average_fitness = statistics.mean(candidate.fitness for candidate in candidates)

    # ----------------------------------------------
    # RECORD
    # ----------------------------------------------

    if best_fitness_ever == float("-inf"):
        improvement = float("inf")

    else:
        improvement = champion.fitness - best_fitness_ever

    new_record = False

    if improvement > MIN_IMPROVEMENT:
        best_fitness_ever = champion.fitness

        best_brain_ever = champion.brain.copy()

        best_brain_ever.save(MODEL_FILE)

        generations_without_improvement = 0

        new_record = True

    else:
        generations_without_improvement += 1

    # ----------------------------------------------
    # EVOLUTION SETTINGS
    # ----------------------------------------------

    (
        mutation_mode,
        low_mutation,
        medium_mutation,
        high_mutation,
        random_agent_count,
    ) = get_evolution_settings()

    # ----------------------------------------------
    # SAVE HISTORY
    # ----------------------------------------------

    save_generation_stats(champion, average_fitness, mutation_mode)

    # ----------------------------------------------
    # PRINT GENERATION SUMMARY
    # ----------------------------------------------

    print()
    print("=" * 70)

    print(f"MULTI-TRACK GENERATION {generation}")

    print(
    f"Generalization fitness: "
    f"{champion.fitness:.2f}"
    )

    print(
        "(fitness = worst-track performance)"
    )

    print(
        f"Mean track score: "
        f"{champion.mean_track_score:.2f}"
    )

    print(
        f"Worst track score: "
        f"{champion.worst_track_score:.2f}"
    )

    print(f"Population average fitness: {average_fitness:.2f}")

    print(
        f"Track scores: "
        + " | ".join(f"{score:.1f}" for score in champion.track_scores)
    )

    print(f"Total laps across tracks: {champion.total_laps}")

    print(f"Total checkpoints: {champion.total_checkpoints}")

    print(f"Crashes: {sum(champion.track_crashes)}/{len(TRAINING_TRACKS)}")

    print(f"Stagnation: {generations_without_improvement}")

    print(f"Evolution mode: {mutation_mode}")

    if new_record:
        print(f"NEW BEST MULTI-TRACK MODEL SAVED -> {MODEL_FILE}")

    print("=" * 70)

    # ----------------------------------------------
    # PARENTS
    # ----------------------------------------------

    parents = ranked[:PARENT_COUNT]

    new_population = []

    # ----------------------------------------------
    # ELITES
    # ----------------------------------------------

    for i in range(ELITE_COUNT):
        new_population.append(Candidate(ranked[i].brain.copy()))

    # ----------------------------------------------
    # COUNTS
    # ----------------------------------------------

    remaining_slots = POPULATION_SIZE - ELITE_COUNT - random_agent_count

    low_count = int(remaining_slots * 0.45)

    medium_count = int(remaining_slots * 0.35)

    high_count = remaining_slots - low_count - medium_count

    # ----------------------------------------------
    # LOW MUTATION
    # ----------------------------------------------

    for _ in range(low_count):
        new_population.append(create_child(parents, low_mutation))

    # ----------------------------------------------
    # MEDIUM MUTATION
    # ----------------------------------------------

    for _ in range(medium_count):
        new_population.append(create_child(parents, medium_mutation))

    # ----------------------------------------------
    # HIGH MUTATION
    # ----------------------------------------------

    for _ in range(high_count):
        new_population.append(create_child(parents, high_mutation))

    # ----------------------------------------------
    # RANDOM IMMIGRANTS
    # ----------------------------------------------

    for _ in range(random_agent_count):
        new_population.append(Candidate())

    # ----------------------------------------------
    # NEXT GENERATION
    # ----------------------------------------------

    candidates = new_population

    generation += 1

    current_track_index = 0

    start_track_evaluation()


# ==================================================
# DRAW CHECKPOINTS
# ==================================================


def draw_checkpoints(track):
    for checkpoint in track.checkpoints:
        pygame.draw.rect(screen, (0, 200, 255), checkpoint, 2)


# ==================================================
# START DEMO
# ==================================================


def start_demo():
    global demo_car

    global demo_lap_start_time
    global demo_best_lap
    global demo_previous_laps

    if not os.path.exists(MODEL_FILE):
        print("No multi-track model yet.")

        return False

    track = TRAINING_TRACKS[demo_track_index]

    brain = Brain.load(MODEL_FILE)

    demo_car = Car(track.start_x, track.start_y)

    demo_car.angle = track.start_angle

    demo_car.brain = brain

    demo_lap_start_time = time.time()

    demo_best_lap = None

    demo_previous_laps = 0

    return True


# ==================================================
# NEXT DEMO TRACK
# ==================================================


def next_demo_track():
    global demo_track_index

    demo_track_index += 1

    if demo_track_index >= len(TRAINING_TRACKS):
        demo_track_index = 0

    start_demo()


# ==================================================
# TOGGLE MODE
# ==================================================


def toggle_mode():
    global mode

    if mode == MODE_TRAIN:
        if start_demo():
            mode = MODE_DEMO

    else:
        mode = MODE_TRAIN


# ==================================================
# INITIALIZATION
# ==================================================

candidates = create_initial_population()

start_track_evaluation()


# ==================================================
# MAIN LOOP
# ==================================================

running = True

while running:
    # ----------------------------------------------
    # EVENTS
    # ----------------------------------------------

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                toggle_mode()

            if mode == MODE_DEMO:
                if event.key == pygame.K_n:
                    next_demo_track()

                if event.key == pygame.K_r:
                    start_demo()

    # ==================================================
    # TRAIN MODE
    # ==================================================

    if mode == MODE_TRAIN:
        track = TRAINING_TRACKS[current_track_index]

        # ------------------------------------------
        # DRAW TRACK
        # ------------------------------------------

        track.draw(screen)

        track_frame += 1

        # ------------------------------------------
        # UPDATE CARS
        # ------------------------------------------

        for car in cars:
            if not car.alive:
                continue

            car.neural_control(screen)

            car.check_collision(screen)

            if car.alive:
                car.update_checkpoint(track.checkpoints)

                car.update_progress_reward(track.checkpoints)

        # ------------------------------------------
        # ALIVE
        # ------------------------------------------

        alive_cars = [car for car in cars if car.alive]

        # ------------------------------------------
        # CURRENT BEST
        # ------------------------------------------

        current_best_car = max(cars, key=calculate_track_fitness)

        # ------------------------------------------
        # END TRACK?
        # ------------------------------------------

        if len(alive_cars) == 0 or track_frame >= MAX_FRAMES_PER_TRACK:
            finish_current_track()

            # State may now point to
            # another track/generation.
            track = TRAINING_TRACKS[current_track_index]

            track.draw(screen)

            alive_cars = [car for car in cars if car.alive]

            current_best_car = max(cars, key=calculate_track_fitness)

        # ------------------------------------------
        # DRAW CHECKPOINTS
        # ------------------------------------------

        draw_checkpoints(track)

        # ------------------------------------------
        # DRAW CARS
        # ------------------------------------------

        for car in cars:
            if not car.alive:
                continue

            if car is current_best_car:
                car.draw(screen, (255, 220, 0))

            else:
                car.draw(screen, (220, 50, 50))

        # ------------------------------------------
        # SENSORS
        # ------------------------------------------

        if current_best_car.alive:
            current_best_car.draw_sensors(screen)

        # ------------------------------------------
        # EVOLUTION INFO
        # ------------------------------------------

        (mutation_mode, low_mutation, medium_mutation, high_mutation, random_count) = (
            get_evolution_settings()
        )

        # ------------------------------------------
        # HUD
        # ------------------------------------------

        mode_text = font.render("MODE: MULTI-TRACK TRAIN", True, (100, 255, 100))

        generation_text = font.render(
            f"Generation: {generation}", True, (255, 255, 255)
        )

        track_text = font.render(
            (f"Track: {current_track_index + 1}/{len(TRAINING_TRACKS)} - {track.name}"),
            True,
            (255, 255, 255),
        )

        alive_text = font.render(
            (f"Alive: {len(alive_cars)}/{POPULATION_SIZE}"), True, (255, 255, 255)
        )

        fitness_text = small_font.render(
            (f"Current track fitness: {calculate_track_fitness(current_best_car):.1f}"),
            True,
            (220, 220, 220),
        )

        control_text = small_font.render(
            (
                f"Steering: "
                f"{current_best_car.last_steering:+.2f}"
                f" | Throttle: "
                f"{current_best_car.last_throttle:.2f}"
                f" | Speed: "
                f"{current_best_car.velocity:.2f}"
            ),
            True,
            (220, 220, 220),
        )

        evolution_text = small_font.render(
            (
                f"Evolution: "
                f"{mutation_mode}"
                f" | Stagnation: "
                f"{generations_without_improvement}"
            ),
            True,
            (220, 220, 220),
        )

        frame_text = small_font.render(
            (f"Track frame: {track_frame}/{MAX_FRAMES_PER_TRACK}"),
            True,
            (220, 220, 220),
        )

        controls_text = small_font.render("TAB = Demo", True, (220, 220, 220))

        screen.blit(mode_text, (20, 20))

        screen.blit(generation_text, (20, 55))

        screen.blit(track_text, (20, 85))

        screen.blit(alive_text, (20, 115))

        screen.blit(fitness_text, (20, 150))

        screen.blit(control_text, (20, 175))

        screen.blit(evolution_text, (20, 200))

        screen.blit(frame_text, (20, 225))

        screen.blit(controls_text, (20, 250))

    # ==================================================
    # DEMO MODE
    # ==================================================

    else:
        track = TRAINING_TRACKS[demo_track_index]

        track.draw(screen)

        if demo_car is not None:
            if demo_car.alive:
                demo_car.neural_control(screen)

                demo_car.check_collision(screen)

                if demo_car.alive:
                    demo_car.update_checkpoint(track.checkpoints)

                    demo_car.update_progress_reward(track.checkpoints)

            # --------------------------------------
            # LAP TIMER
            # --------------------------------------

            if demo_car.laps_completed > demo_previous_laps:
                lap_time = time.time() - demo_lap_start_time

                if demo_best_lap is None or lap_time < demo_best_lap:
                    demo_best_lap = lap_time

                print(f"{track.name} | Lap {demo_car.laps_completed}: {lap_time:.2f}s")

                demo_previous_laps = demo_car.laps_completed

                demo_lap_start_time = time.time()

            # --------------------------------------
            # DRAW
            # --------------------------------------

            draw_checkpoints(track)

            if demo_car.alive:
                demo_car.draw(screen, (255, 220, 0))

                demo_car.draw_sensors(screen)

            # --------------------------------------
            # HUD
            # --------------------------------------

            mode_text = font.render("MODE: MULTI-TRACK DEMO", True, (255, 220, 0))

            track_text = font.render(
                (
                    f"Track: "
                    f"{demo_track_index + 1}"
                    f"/{len(TRAINING_TRACKS)} "
                    f"- {track.name}"
                ),
                True,
                (255, 255, 255),
            )

            laps_text = font.render(
                (f"Laps: {demo_car.laps_completed}"), True, (255, 255, 255)
            )

            controls_text = small_font.render(
                ("N = Next track | R = Restart | TAB = Train"), True, (220, 220, 220)
            )

            screen.blit(mode_text, (20, 20))

            screen.blit(track_text, (20, 55))

            screen.blit(laps_text, (20, 85))

            screen.blit(controls_text, (20, 120))

    # ==================================================
    # DISPLAY
    # ==================================================

    pygame.display.flip()

    clock.tick(FPS)


pygame.quit()
