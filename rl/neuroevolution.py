"""Headless neuroevolution trainer with exact step counting.

main.py trains in real time in a window and does not count simulation
steps, so its sample cost could only be bounded from above. This
module runs the same algorithm headlessly on the RL environment's
track geometry, counts every simulated car step, and writes the same
logs as the PPO trainer (progress.csv, eval.csv), so all three methods
can be compared on the same axes.

The algorithm follows main.py exactly:
- population of 40 Brains, every candidate driven on every training
  track with a shared random start perturbation per track
- fitness = worst track score (maximin), ties broken by the mean
- 2 elites, 5 parents, uniform crossover + Gaussian mutation at low,
  medium and high rates (45% / 35% / 20% of the children)
- adaptive mutation: FINE / EXPLORE / ESCAPE after 8 and 18 generations
  without an improvement of more than 1.0, with 2 / 5 / 10 random
  immigrants
- the record champion is saved whenever the best fitness improves

Unlike main.py, training starts from a random population (no seed
model), so every method starts from scratch.

    python -m rl.neuroevolution --tracks train --seed 0 --run-name ne_multi_seed0
"""

import argparse
import dataclasses
import json
import os
import random
import statistics
import time

import numpy as np

from brain import Brain
from envs.brain_policy import BrainPolicy
from envs.env_car import EnvCar
from envs.racing_env import track_fitness
from envs.track_geometry import get_geometry
from rl.common import resolve_tracks
from rl.evaluate import evaluate_policy
from rl.trainer import CsvLog


@dataclasses.dataclass
class NEConfig:
    tracks: str = "train"
    total_timesteps: int = 5_000_000
    max_steps: int = 1800

    # Same values as main.py
    population_size: int = 40
    parent_count: int = 5
    elite_count: int = 2
    stagnation_level_1: int = 8
    stagnation_level_2: int = 18
    min_improvement: float = 1.0

    seed: int = 0
    eval_interval: int = 1          # generations between evaluations
    eval_episodes: int = 3          # per track

    def save(self, path):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(dataclasses.asdict(self), file, indent=2)


# ==================================================
# EVOLUTION SETTINGS (main.get_evolution_settings)
# ==================================================

def evolution_settings(stagnation, config):
    """(mode, low, medium, high mutation rate, random immigrants)."""
    if stagnation < config.stagnation_level_1:
        return ("FINE", 0.01, 0.03, 0.08, 2)

    if stagnation < config.stagnation_level_2:
        return ("EXPLORE", 0.03, 0.08, 0.20, 5)

    return ("ESCAPE", 0.05, 0.15, 0.35, 10)


# ==================================================
# EVALUATION OF ONE GENERATION
# ==================================================

def drive_population(brains, track, max_steps):
    """Drive every brain on one track; returns (fitnesses, cars, steps).

    All cars share one random start perturbation, like
    main.start_track_evaluation. The loop order per car matches the
    game loop: sense, act, collide, checkpoints, progress.
    """
    x_offset = random.uniform(-8, 8)
    y_offset = random.uniform(-4, 4)
    angle_offset = random.uniform(-4, 4)

    geometry = get_geometry(track)

    cars = [
        EnvCar(
            track.start_x + x_offset,
            track.start_y + y_offset,
            track.start_angle + angle_offset,
            geometry,
        )
        for _ in brains
    ]

    steps = 0

    for _ in range(max_steps):
        alive = [index for index, car in enumerate(cars) if car.alive]

        if not alive:
            break

        for index in alive:
            car = cars[index]

            car.apply_controls(*brains[index].predict(car.observation()))
            car.check_track_collision()

            if car.alive:
                car.update_checkpoint(track.checkpoints)
                car.update_progress_reward(track.checkpoints)

        steps += len(alive)

    return [track_fitness(car) for car in cars], cars, steps


def evaluate_generation(brains, tracks, max_steps):
    """Per-candidate track scores and the number of simulated steps."""
    scores = [[] for _ in brains]
    laps = [[] for _ in brains]
    total_steps = 0

    for track in tracks:
        fitnesses, cars, steps = drive_population(brains, track, max_steps)
        total_steps += steps

        for index, (fitness, car) in enumerate(zip(fitnesses, cars)):
            scores[index].append(fitness)
            laps[index].append(car.laps_completed)

    return scores, laps, total_steps


# ==================================================
# NEXT GENERATION (main.finish_generation)
# ==================================================

def next_generation(ranked_brains, settings, config):
    _, low, medium, high, immigrants = settings

    parents = ranked_brains[:config.parent_count]

    population = [
        ranked_brains[index].copy()
        for index in range(config.elite_count)
    ]

    remaining = config.population_size - config.elite_count - immigrants

    low_count = int(remaining * 0.45)
    medium_count = int(remaining * 0.35)
    high_count = remaining - low_count - medium_count

    for count, rate in (
        (low_count, low),
        (medium_count, medium),
        (high_count, high),
    ):
        for _ in range(count):
            child = Brain.crossover(random.choice(parents), random.choice(parents))
            child.mutate(rate)
            population.append(child)

    population += [Brain() for _ in range(immigrants)]

    return population


# ==================================================
# TRAINING LOOP
# ==================================================

def train(config, run_dir, log=print):
    """Evolve until the step budget is used; returns the best fitness."""
    os.makedirs(run_dir, exist_ok=True)
    config.save(os.path.join(run_dir, "config.json"))

    # Brain uses NumPy's global generator; main.py uses random as well.
    random.seed(config.seed)
    np.random.seed(config.seed)

    tracks = resolve_tracks(config.tracks)

    progress_log = CsvLog(os.path.join(run_dir, "progress.csv"))
    eval_log = CsvLog(os.path.join(run_dir, "eval.csv"))

    population = [Brain() for _ in range(config.population_size)]

    best_fitness = float("-inf")
    best_brain = None
    stagnation = 0

    global_step = 0
    generation = 0
    start_time = time.perf_counter()

    while global_step < config.total_timesteps:
        generation += 1

        scores, laps, steps = evaluate_generation(
            population, tracks, config.max_steps
        )
        global_step += steps

        # Rank by (worst track, mean track), like main.py.
        order = sorted(
            range(len(population)),
            key=lambda i: (min(scores[i]), statistics.mean(scores[i])),
            reverse=True
        )

        champion = order[0]
        champion_fitness = min(scores[champion])

        improvement = (
            float("inf") if best_fitness == float("-inf")
            else champion_fitness - best_fitness
        )

        if improvement > config.min_improvement:
            best_fitness = champion_fitness
            best_brain = population[champion].copy()
            best_brain.save(os.path.join(run_dir, "best.npz"))
            stagnation = 0
        else:
            stagnation += 1

        settings = evolution_settings(stagnation, config)

        elapsed = time.perf_counter() - start_time

        progress_log.write({
            "update": generation,
            "global_step": global_step,
            "time": round(elapsed, 1),
            "steps_per_second": round(global_step / elapsed),
            "generation_steps": steps,
            "champion_fitness": champion_fitness,
            "champion_mean_score": statistics.mean(scores[champion]),
            "champion_laps": sum(laps[champion]),
            "population_average_fitness": statistics.mean(
                min(candidate) for candidate in scores
            ),
            "best_fitness": best_fitness,
            "stagnation": stagnation,
            "mutation_mode": settings[0],
        })

        log(
            f"generation {generation:3d} "
            f"step {global_step:>9,} "
            f"champion {champion_fitness:9.1f} "
            f"best {best_fitness:9.1f} "
            f"laps {sum(laps[champion]):3d} "
            f"{settings[0]:<7} "
            f"{round(global_step / elapsed):,} steps/s"
        )

        if generation % config.eval_interval == 0:
            results = evaluate_policy(
                BrainPolicy(population[champion]),
                tracks=config.tracks,
                episodes=config.eval_episodes,
                max_steps=config.max_steps
            )

            for track_name, metrics in results.items():
                eval_log.write({
                    "update": generation,
                    "global_step": global_step,
                    "track": track_name,
                    **metrics,
                })

        population = next_generation(
            [population[index] for index in order],
            settings,
            config
        )

    # The first elite is the last generation's champion.
    population[0].save(os.path.join(run_dir, "final.npz"))

    return best_fitness


# ==================================================
# CLI
# ==================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Headless neuroevolution.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--runs-dir", default="runs")

    for field in dataclasses.fields(NEConfig):
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=type(field.default),
            default=field.default
        )

    args = parser.parse_args(argv)

    config = NEConfig(**{
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(NEConfig)
    })

    run_name = args.run_name or (
        f"ne_{config.tracks.replace(':', '')}_seed{config.seed}"
    )

    best = train(
        config,
        os.path.join(args.runs_dir, run_name),
        log=lambda message: print(message, flush=True)
    )

    print(f"Best worst-track training fitness: {best:.1f}")


if __name__ == "__main__":
    main()
