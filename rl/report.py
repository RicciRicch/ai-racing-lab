"""Final comparison of neuroevolution, PyTorch PPO and TensorFlow PPO.

For each scenario (single-track, multi-track) and method, it reads the
training logs of every seed and evaluates every seed's best model:

1. final performance on all four tracks (shared evaluator and seeds)
2. sample efficiency: environment steps until the evaluation first
   reaches a target, from eval.csv
3. computational cost: wall-clock time to that target and in total

    python -m rl.report --output results/rl/final_comparison.md
"""

import argparse
import csv
import glob
import os
import statistics

import matplotlib.pyplot as plt
import numpy as np

from rl.compare import average
from rl.evaluate import EVAL_SEED, evaluate_policy, load_policy
from rl.plot_runs import group_curves, plot_band, wall_clock


METHODS = [
    ("Neuroevolution (NumPy)", "ne"),
    ("PPO (PyTorch)", "torch"),
    ("PPO (TensorFlow)", "tf"),
    # Same budget as neuroevolution, multi-track only.
    ("PPO (PyTorch), 5M steps", "torch5m"),
]

SCENARIOS = [
    # name, run suffix, number of training tracks,
    # target description, target test on eval rows
    (
        "Single-track (Classic Oval)",
        "single",
        1,
        "9 laps on the Classic Oval",
        lambda rows: all(float(r["laps"]) >= 9 for r in rows),
    ),
    (
        "Multi-track (Classic, Tall, Asymmetric)",
        "multi",
        3,
        # Circling without laps scores at most ~4,000 on a track, so
        # 10,000 requires laps on every training track.
        "worst-track fitness >= 10,000 (laps on all three tracks)",
        lambda rows: min(float(r["fitness"]) for r in rows) >= 10_000,
    ),
]


def run_dirs(runs_dir, group):
    return sorted(glob.glob(os.path.join(runs_dir, group + "_seed*")))


def read_rows(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def first_success(run_dir, target):
    """(steps, seconds) of the first evaluation meeting the target."""
    evaluations = {}

    for row in read_rows(os.path.join(run_dir, "eval.csv")):
        evaluations.setdefault(int(row["global_step"]), []).append(row)

    times = {
        int(row["global_step"]): float(row["time"])
        for row in read_rows(os.path.join(run_dir, "progress.csv"))
    }

    for step in sorted(evaluations):
        if target(evaluations[step]):
            return step, times[step]

    return None, None


def training_totals(run_dir):
    last = read_rows(os.path.join(run_dir, "progress.csv"))[-1]
    return int(last["global_step"]), float(last["time"])


def format_median(values, formatter):
    found = [v for v in values if v is not None]

    if not found:
        return "not reached"

    text = formatter(statistics.median(found))

    if len(found) < len(values):
        text += f" ({len(found)}/{len(values)} seeds)"

    return text


def steps_text(value):
    return f"{value / 1e3:,.0f}k" if value < 1e6 else f"{value / 1e6:.2f}M"


def evaluate_group(runs_dir, group, episodes):
    results = []

    for run_dir in run_dirs(runs_dir, group):
        model = glob.glob(os.path.join(run_dir, "best.*"))[0]
        results.append(evaluate_policy(load_policy(model), "all", episodes))

    return results


def scenario_section(name, suffix, training_tracks, target_text, target,
                     runs_dir, episodes):
    lines = [f"## {name}", ""]

    performance = []
    efficiency = []
    seed_counts = set()

    for label, prefix in METHODS:
        group = f"{prefix}_{suffix}"
        dirs = run_dirs(runs_dir, group)

        if not dirs:
            continue

        seed_counts.add(len(dirs))

        per_seed = evaluate_group(runs_dir, group, episodes)
        mean = average(per_seed)
        tracks = list(mean)

        worst = [
            min(r[t]["fitness"] for t in tracks[:training_tracks])
            for r in per_seed
        ]

        performance.append(
            f"| {label} | "
            + " | ".join(
                f"{mean[t]['laps']:.1f} laps, {mean[t]['crash_rate']:.0%} crash"
                for t in tracks
            )
            + f" | {statistics.mean(worst):,.0f} "
            f"({min(worst):,.0f}–{max(worst):,.0f}) |"
        )

        successes = [first_success(d, target) for d in dirs]
        totals = [training_totals(d) for d in dirs]

        efficiency.append(
            f"| {label} | "
            f"{format_median([s for s, _ in successes], steps_text)} | "
            f"{format_median([t for _, t in successes], lambda v: f'{v:,.0f} s')} | "
            f"{steps_text(statistics.mean(s for s, _ in totals))} | "
            f"{statistics.mean(t for _, t in totals):,.0f} s | "
            f"{statistics.mean(s / t for s, t in totals):,.0f} |"
        )

    lines += [
        f"### Final performance ({'/'.join(map(str, sorted(seed_counts)))} seeds, "
        f"best model per seed)",
        "",
        "| Method | " + " | ".join(tracks) + " | Worst training-track fitness, mean (min–max) |",
        "|---|" + "---:|" * (len(tracks) + 1),
        *performance,
        "",
        f"### Efficiency (target: {target_text})",
        "",
        "Medians over seeds. Steps are simulated car steps (environment steps); "
        "evaluation episodes are not counted.",
        "",
        "| Method | Steps to target | Time to target | Training budget | "
        "Training time | Steps/s |",
        "|---|---:|---:|---:|---:|---:|",
        *efficiency,
        "",
        f"![{name}](final_{suffix}.png)",
        "",
    ]

    return "\n".join(lines)


def scenario_plot(suffix, runs_dir, output):
    figure, (steps_axis, time_axis) = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, prefix in METHODS:
        group = f"{prefix}_{suffix}"

        if not run_dirs(runs_dir, group):
            continue

        steps, worst, _, seeds = group_curves(runs_dir, group, "fitness")
        times = wall_clock(runs_dir, group, steps).mean(axis=0)

        plot_band(steps_axis, steps, worst, f"{label} ({seeds} seeds)")
        plot_band(time_axis, times, worst, f"{label} ({seeds} seeds)")

    steps_axis.set_xscale("log")
    time_axis.set_xscale("log")
    steps_axis.set_xlabel("environment steps (log scale)")
    time_axis.set_xlabel("training time in seconds (log scale)")

    for axis in (steps_axis, time_axis):
        axis.set_ylabel("worst-track evaluation fitness")
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=8)

    steps_axis.set_title("Sample efficiency")
    time_axis.set_title("Computational efficiency")

    figure.tight_layout()
    figure.savefig(output, dpi=110)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", default="results/rl/final_comparison.md")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    sections = [
        "# Final Comparison: Neuroevolution vs PPO (PyTorch, TensorFlow)",
        "",
        f"Generated by `python -m rl.report`. Every model is evaluated with "
        f"`rl/evaluate.py` on all four tracks, {args.episodes} episodes per "
        f"track (seeds {EVAL_SEED}–{EVAL_SEED + args.episodes - 1}), "
        f"deterministic actions. All methods train from scratch in the same "
        f"environment with the same reward/fitness.",
        "",
    ]

    for name, suffix, training_tracks, target_text, target in SCENARIOS:
        scenario_plot(suffix, args.runs_dir, os.path.join(output_dir, f"final_{suffix}.png"))
        sections.append(
            scenario_section(name, suffix, training_tracks, target_text,
                             target, args.runs_dir, args.episodes)
        )

    text = "\n".join(sections)

    with open(args.output, "w", encoding="utf-8") as file:
        file.write(text)

    print(text)


if __name__ == "__main__":
    main()
