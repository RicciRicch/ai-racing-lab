import os
import random
import time

import pygame

from car import Car
from brain import Brain


# ==================================================
# SETTINGS
# ==================================================

WIDTH = 1000
HEIGHT = 700

FPS = 60

POPULATION_SIZE = 40

PARENT_COUNT = 5
ELITE_COUNT = 2
RANDOM_AGENT_COUNT = 4

MAX_FRAMES_PER_GENERATION = 1800

START_X = 500
START_Y = 590
START_ANGLE = -90

MODEL_FILE = "best_brain.npz"

MODE_TRAIN = "TRAIN"
MODE_DEMO = "DEMO"


# ==================================================
# PYGAME
# ==================================================

pygame.init()

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

small_font = pygame.font.SysFont(
    "Arial",
    18
)


# ==================================================
# CHECKPOINTS
# ==================================================

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


# ==================================================
# GLOBAL STATE
# ==================================================

mode = MODE_TRAIN

generation = 1
generation_frame = 0

best_fitness_ever = float("-inf")
best_checkpoints_ever = 0
best_laps_ever = 0

best_brain_ever = None

cars = []

demo_car = None

demo_lap_start_time = None
demo_best_lap = None
demo_previous_laps = 0


# ==================================================
# CREATE CAR
# ==================================================

def create_car(brain=None):
    car = Car(
        START_X,
        START_Y
    )

    car.angle = START_ANGLE

    if brain is not None:
        car.brain = brain

    return car


# ==================================================
# TRAINING POPULATION
# ==================================================

def create_initial_population():
    population = []

    if os.path.exists(
        MODEL_FILE
    ):
        print(
            f"Loading saved model: "
            f"{MODEL_FILE}"
        )

        saved_brain = Brain.load(
            MODEL_FILE
        )

        # Exact champion
        population.append(
            create_car(
                saved_brain.copy()
            )
        )

        # Mutations of champion
        while (
            len(population)
            < POPULATION_SIZE
        ):
            new_brain = (
                saved_brain.copy()
            )

            mutation_rate = (
                random.choice(
                    [
                        0.05,
                        0.10,
                        0.15,
                        0.25
                    ]
                )
            )

            new_brain.mutate(
                mutation_rate
            )

            population.append(
                create_car(
                    new_brain
                )
            )

        return population

    # No model yet
    return [
        create_car()
        for _ in range(
            POPULATION_SIZE
        )
    ]


# ==================================================
# TRACK
# ==================================================

def draw_track():
    # Grass
    screen.fill(
        (40, 130, 60)
    )

    # Road
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

    # Inner grass
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
# CHECKPOINT VISUALIZATION
# ==================================================

def draw_checkpoints():
    for checkpoint in checkpoints:
        pygame.draw.rect(
            screen,
            (0, 200, 255),
            checkpoint,
            2
        )


# ==================================================
# FITNESS
# ==================================================

def calculate_fitness(car):
    checkpoint_reward = (
        car.checkpoints_passed
        * 1000
    )

    progress_reward = (
        car.progress_reward
        * 0.5
    )

    survival_reward = (
        car.frames_alive
        * 0.01
    )

    car.fitness = (
        checkpoint_reward
        + progress_reward
        + survival_reward
    )


# ==================================================
# CREATE CHILD
# ==================================================

def create_child(
    parents,
    mutation_rate
):
    parent_a = random.choice(
        parents
    )

    parent_b = random.choice(
        parents
    )

    child_brain = (
        Brain.crossover(
            parent_a.brain,
            parent_b.brain
        )
    )

    child_brain.mutate(
        mutation_rate
    )

    return create_car(
        child_brain
    )


# ==================================================
# NEXT GENERATION
# ==================================================

def next_generation():
    global cars

    global generation
    global generation_frame

    global best_fitness_ever
    global best_checkpoints_ever
    global best_laps_ever

    global best_brain_ever

    # Final fitness
    for car in cars:
        calculate_fitness(
            car
        )

    ranked_cars = sorted(
        cars,
        key=lambda car: car.fitness,
        reverse=True
    )

    champion = ranked_cars[0]

    average_fitness = (
        sum(
            car.fitness
            for car in cars
        )
        / len(cars)
    )

    # ----------------------------------------------
    # RECORDS
    # ----------------------------------------------

    new_record = False

    if (
        champion.fitness
        > best_fitness_ever
    ):
        best_fitness_ever = (
            champion.fitness
        )

        best_brain_ever = (
            champion.brain.copy()
        )

        best_brain_ever.save(
            MODEL_FILE
        )

        new_record = True

    if (
        champion.checkpoints_passed
        > best_checkpoints_ever
    ):
        best_checkpoints_ever = (
            champion.checkpoints_passed
        )

    if (
        champion.laps_completed
        > best_laps_ever
    ):
        best_laps_ever = (
            champion.laps_completed
        )

    # ----------------------------------------------
    # TERMINAL STATS
    # ----------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        f"Generation {generation}"
    )

    print(
        f"Best fitness: "
        f"{champion.fitness:.2f}"
    )

    print(
        f"Average fitness: "
        f"{average_fitness:.2f}"
    )

    print(
        f"Checkpoints: "
        f"{champion.checkpoints_passed}"
    )

    print(
        f"Laps: "
        f"{champion.laps_completed}"
    )

    print(
        f"Progress reward: "
        f"{champion.progress_reward:.2f}"
    )

    print(
        f"Frames alive: "
        f"{champion.frames_alive}"
    )

    if new_record:
        print(
            f"NEW BEST MODEL SAVED -> "
            f"{MODEL_FILE}"
        )

    print(
        "=" * 60
    )

    # ----------------------------------------------
    # SELECT PARENTS
    # ----------------------------------------------

    parents = ranked_cars[
        :PARENT_COUNT
    ]

    new_population = []

    # ----------------------------------------------
    # ELITISM
    # ----------------------------------------------

    for i in range(
        ELITE_COUNT
    ):
        new_population.append(
            create_car(
                ranked_cars[i]
                .brain
                .copy()
            )
        )

    # ----------------------------------------------
    # LOW MUTATION
    # ----------------------------------------------

    for _ in range(10):
        new_population.append(
            create_child(
                parents,
                0.05
            )
        )

    # ----------------------------------------------
    # MEDIUM MUTATION
    # ----------------------------------------------

    for _ in range(14):
        new_population.append(
            create_child(
                parents,
                0.15
            )
        )

    # ----------------------------------------------
    # HIGH MUTATION
    # ----------------------------------------------

    high_mutation_count = (
        POPULATION_SIZE
        - ELITE_COUNT
        - 10
        - 14
        - RANDOM_AGENT_COUNT
    )

    for _ in range(
        high_mutation_count
    ):
        new_population.append(
            create_child(
                parents,
                0.30
            )
        )

    # ----------------------------------------------
    # RANDOM AGENTS
    # ----------------------------------------------

    for _ in range(
        RANDOM_AGENT_COUNT
    ):
        new_population.append(
            create_car()
        )

    cars = new_population

    generation += 1
    generation_frame = 0


# ==================================================
# RESET TRAINING
# ==================================================

def reset_training():
    global cars

    global generation
    global generation_frame

    global best_fitness_ever
    global best_checkpoints_ever
    global best_laps_ever

    global best_brain_ever

    generation = 1
    generation_frame = 0

    best_fitness_ever = float(
        "-inf"
    )

    best_checkpoints_ever = 0
    best_laps_ever = 0

    best_brain_ever = None

    cars = [
        create_car()
        for _ in range(
            POPULATION_SIZE
        )
    ]

    print(
        "Training restarted "
        "with random brains."
    )


# ==================================================
# START DEMO
# ==================================================

def start_demo():
    global demo_car

    global demo_lap_start_time
    global demo_best_lap
    global demo_previous_laps

    if not os.path.exists(
        MODEL_FILE
    ):
        print(
            "Cannot start DEMO mode."
        )

        print(
            f"{MODEL_FILE} does not exist yet."
        )

        return False

    brain = Brain.load(
        MODEL_FILE
    )

    demo_car = create_car(
        brain
    )

    demo_lap_start_time = (
        time.time()
    )

    demo_best_lap = None
    demo_previous_laps = 0

    print()
    print(
        "DEMO MODE"
    )

    print(
        f"Loaded model: "
        f"{MODEL_FILE}"
    )

    return True


# ==================================================
# RESET DEMO CAR
# ==================================================

def reset_demo_car():
    global demo_car

    global demo_lap_start_time
    global demo_previous_laps

    if not os.path.exists(
        MODEL_FILE
    ):
        return

    brain = Brain.load(
        MODEL_FILE
    )

    demo_car = create_car(
        brain
    )

    demo_lap_start_time = (
        time.time()
    )

    demo_previous_laps = 0


# ==================================================
# CHANGE MODE
# ==================================================

def toggle_mode():
    global mode
    global cars

    if mode == MODE_TRAIN:

        success = start_demo()

        if success:
            mode = MODE_DEMO

    else:
        mode = MODE_TRAIN

        # Start training from saved model
        cars = (
            create_initial_population()
        )

        print()
        print(
            "TRAIN MODE"
        )


# ==================================================
# INITIALIZE TRAINING
# ==================================================

cars = (
    create_initial_population()
)


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

            # Switch TRAIN / DEMO
            if event.key == pygame.K_TAB:
                toggle_mode()

            # Reset
            if event.key == pygame.K_r:

                if mode == MODE_TRAIN:
                    reset_training()

                elif mode == MODE_DEMO:
                    reset_demo_car()

    # ----------------------------------------------
    # DRAW WORLD
    # ----------------------------------------------

    draw_track()

    # ==================================================
    # TRAIN MODE
    # ==================================================

    if mode == MODE_TRAIN:

        generation_frame += 1

        # ------------------------------------------
        # UPDATE CARS
        # ------------------------------------------

        for car in cars:

            if not car.alive:
                continue

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

            calculate_fitness(
                car
            )

        # ------------------------------------------
        # ALIVE CARS
        # ------------------------------------------

        alive_cars = [
            car
            for car in cars
            if car.alive
        ]

        # ------------------------------------------
        # END GENERATION?
        # ------------------------------------------

        everyone_dead = (
            len(alive_cars) == 0
        )

        time_limit_reached = (
            generation_frame
            >= MAX_FRAMES_PER_GENERATION
        )

        if (
            everyone_dead
            or time_limit_reached
        ):
            next_generation()

            draw_track()

            alive_cars = [
                car
                for car in cars
                if car.alive
            ]

        # ------------------------------------------
        # CURRENT BEST
        # ------------------------------------------

        current_best_car = max(
            cars,
            key=lambda car: car.fitness
        )

        # ------------------------------------------
        # DRAW CHECKPOINTS
        # ------------------------------------------

        draw_checkpoints()

        # ------------------------------------------
        # DRAW CARS
        # ------------------------------------------

        for car in cars:

            if not car.alive:
                continue

            if (
                car is current_best_car
            ):
                car.draw(
                    screen,
                    (
                        255,
                        220,
                        0
                    )
                )

            else:
                car.draw(
                    screen,
                    (
                        220,
                        50,
                        50
                    )
                )

        # ------------------------------------------
        # BEST SENSORS
        # ------------------------------------------

        if current_best_car.alive:
            current_best_car.draw_sensors(
                screen
            )

        # ------------------------------------------
        # DISPLAY RECORD
        # ------------------------------------------

        if (
            best_fitness_ever
            == float("-inf")
        ):
            best_ever_display = 0

        else:
            best_ever_display = (
                best_fitness_ever
            )

        # ------------------------------------------
        # HUD
        # ------------------------------------------

        mode_text = font.render(
            "MODE: TRAIN",
            True,
            (
                100,
                255,
                100
            )
        )

        generation_text = font.render(
            f"Generation: {generation}",
            True,
            (
                255,
                255,
                255
            )
        )

        alive_text = font.render(
            (
                f"Alive: "
                f"{len(alive_cars)}"
                f"/{POPULATION_SIZE}"
            ),
            True,
            (
                255,
                255,
                255
            )
        )

        fitness_text = font.render(
            (
                f"Best fitness: "
                f"{current_best_car.fitness:.1f}"
            ),
            True,
            (
                255,
                255,
                255
            )
        )

        laps_text = font.render(
            (
                f"Laps: "
                f"{current_best_car.laps_completed}"
            ),
            True,
            (
                255,
                255,
                255
            )
        )

        checkpoint_text = font.render(
            (
                f"Checkpoints: "
                f"{current_best_car.checkpoints_passed}"
            ),
            True,
            (
                255,
                255,
                255
            )
        )

        frame_text = small_font.render(
            (
                f"Frame: "
                f"{generation_frame}"
                f"/{MAX_FRAMES_PER_GENERATION}"
            ),
            True,
            (
                220,
                220,
                220
            )
        )

        record_text = small_font.render(
            (
                f"Best ever: "
                f"{best_ever_display:.1f}"
                f" | Laps: "
                f"{best_laps_ever}"
                f" | Checkpoints: "
                f"{best_checkpoints_ever}"
            ),
            True,
            (
                220,
                220,
                220
            )
        )

        controls_text = small_font.render(
            (
                "TAB = Demo | "
                "R = restart training"
            ),
            True,
            (
                220,
                220,
                220
            )
        )

        screen.blit(
            mode_text,
            (20, 20)
        )

        screen.blit(
            generation_text,
            (20, 55)
        )

        screen.blit(
            alive_text,
            (20, 85)
        )

        screen.blit(
            fitness_text,
            (20, 115)
        )

        screen.blit(
            laps_text,
            (20, 145)
        )

        screen.blit(
            checkpoint_text,
            (20, 175)
        )

        screen.blit(
            frame_text,
            (20, 210)
        )

        screen.blit(
            record_text,
            (20, 235)
        )

        screen.blit(
            controls_text,
            (20, 260)
        )

    # ==================================================
    # DEMO MODE
    # ==================================================

    elif mode == MODE_DEMO:

        if demo_car is not None:

            # --------------------------------------
            # UPDATE
            # --------------------------------------

            if demo_car.alive:

                demo_car.neural_control(
                    screen
                )

                demo_car.check_collision(
                    screen
                )

                if demo_car.alive:

                    demo_car.update_checkpoint(
                        checkpoints
                    )

                    demo_car.update_progress_reward(
                        checkpoints
                    )

            # --------------------------------------
            # LAP TIMER
            # --------------------------------------

            if (
                demo_car.laps_completed
                > demo_previous_laps
            ):
                lap_time = (
                    time.time()
                    - demo_lap_start_time
                )

                print(
                    f"Demo lap "
                    f"{demo_car.laps_completed}: "
                    f"{lap_time:.2f}s"
                )

                if (
                    demo_best_lap is None
                    or lap_time
                    < demo_best_lap
                ):
                    demo_best_lap = (
                        lap_time
                    )

                    print(
                        f"NEW DEMO BEST LAP: "
                        f"{demo_best_lap:.2f}s"
                    )

                demo_previous_laps = (
                    demo_car.laps_completed
                )

                demo_lap_start_time = (
                    time.time()
                )

            # --------------------------------------
            # DRAW CHECKPOINTS
            # --------------------------------------

            draw_checkpoints()

            # --------------------------------------
            # DRAW CAR
            # --------------------------------------

            if demo_car.alive:

                demo_car.draw(
                    screen,
                    (
                        255,
                        220,
                        0
                    )
                )

                demo_car.draw_sensors(
                    screen
                )

            # --------------------------------------
            # CURRENT LAP TIME
            # --------------------------------------

            current_lap_time = (
                time.time()
                - demo_lap_start_time
            )

            # --------------------------------------
            # HUD
            # --------------------------------------

            mode_text = font.render(
                "MODE: DEMO",
                True,
                (
                    255,
                    220,
                    0
                )
            )

            laps_text = font.render(
                (
                    f"Laps: "
                    f"{demo_car.laps_completed}"
                ),
                True,
                (
                    255,
                    255,
                    255
                )
            )

            checkpoint_text = font.render(
                (
                    f"Checkpoints: "
                    f"{demo_car.checkpoints_passed}"
                ),
                True,
                (
                    255,
                    255,
                    255
                )
            )

            current_time_text = font.render(
                (
                    f"Lap time: "
                    f"{current_lap_time:.2f}s"
                ),
                True,
                (
                    255,
                    255,
                    255
                )
            )

            if demo_best_lap is None:
                best_lap_string = (
                    "Best lap: --"
                )

            else:
                best_lap_string = (
                    f"Best lap: "
                    f"{demo_best_lap:.2f}s"
                )

            best_lap_text = font.render(
                best_lap_string,
                True,
                (
                    255,
                    255,
                    255
                )
            )

            status = (
                "ALIVE"
                if demo_car.alive
                else "CRASHED"
            )

            status_text = font.render(
                f"Status: {status}",
                True,
                (
                    255,
                    255,
                    255
                )
            )

            controls_text = small_font.render(
                (
                    "TAB = Train | "
                    "R = restart demo"
                ),
                True,
                (
                    220,
                    220,
                    220
                )
            )

            screen.blit(
                mode_text,
                (20, 20)
            )

            screen.blit(
                laps_text,
                (20, 55)
            )

            screen.blit(
                checkpoint_text,
                (20, 85)
            )

            screen.blit(
                current_time_text,
                (20, 115)
            )

            screen.blit(
                best_lap_text,
                (20, 145)
            )

            screen.blit(
                status_text,
                (20, 175)
            )

            screen.blit(
                controls_text,
                (20, 210)
            )

    # ==================================================
    # DISPLAY
    # ==================================================

    pygame.display.flip()

    clock.tick(
        FPS
    )


pygame.quit()
