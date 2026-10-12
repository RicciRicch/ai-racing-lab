"""Plot PPO learning curves from runs/<name>/eval.csv.

Runs that differ only in their seed suffix (``_seed0``, ``_seed1`` ...)
are grouped: the line is the mean over seeds, the band spans min-max.

    python -m rl.plot_runs torch_single torch_multi --output results/rl/torch_learning_curves.png

With --overlay, all groups share one pair of axes: worst-track fitness
against environment steps and against wall-clock training time. This
is how the PyTorch and TensorFlow agents are compared.

    python -m rl.plot_runs torch_single tf_single --overlay --output results/rl/frameworks_single.png
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def read_eval(path):
    """{global_step: {track: row}} from one eval.csv."""
    steps = defaultdict(dict)

    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            steps[int(row["global_step"])][row["track"]] = row

    return steps


def common_steps(runs):
    """Evaluation steps shared by all runs, or a common grid.

    PPO evaluates at fixed step counts, so its seeds share every step.
    Neuroevolution evaluates once per generation, and a generation's
    step count depends on when cars crash, so its seeds do not. Then
    the union of all steps inside the range covered by every run is
    used, and each run is read with its last evaluation at or before
    a step.
    """
    shared = sorted(set.intersection(*(set(run) for run in runs)))

    if len(shared) * 2 >= min(len(run) for run in runs):
        return shared

    start = max(min(run) for run in runs)
    end = min(max(run) for run in runs)

    return sorted({
        step for run in runs for step in run if start <= step <= end
    })


def value_at(run, step):
    """Row dict of the last evaluation at or before ``step``."""
    return run[max(s for s in run if s <= step)]


def group_curves(runs_dir, prefix, metric):
    """Steps and per-seed arrays for worst-track and per-track metric."""
    paths = sorted(glob.glob(os.path.join(runs_dir, prefix + "_seed*", "eval.csv")))

    if not paths:
        raise FileNotFoundError(f"No runs found for {prefix!r} in {runs_dir}")

    runs = [read_eval(path) for path in paths]
    steps = common_steps(runs)
    tracks = list(value_at(runs[0], steps[0]))

    per_track = {
        track: np.array([
            [float(value_at(run, step)[track][metric]) for step in steps]
            for run in runs
        ])
        for track in tracks
    }

    worst = np.min(np.stack(list(per_track.values())), axis=0)

    return np.array(steps), worst, per_track, len(paths)


def wall_clock(runs_dir, prefix, steps):
    """Per-seed training time (s) at each step, interpolated."""
    paths = sorted(glob.glob(os.path.join(runs_dir, prefix + "_seed*", "progress.csv")))
    times = []

    for path in paths:
        with open(path, newline="") as file:
            rows = list(csv.DictReader(file))

        logged_steps = [int(row["global_step"]) for row in rows]
        logged_times = [float(row["time"]) for row in rows]

        times.append(np.interp(steps, logged_steps, logged_times))

    return np.array(times)


def plot_overlay(args):
    figure, (steps_axis, time_axis) = plt.subplots(1, 2, figsize=(12, 4.5))

    for prefix in args.prefixes:
        steps, worst, _, seeds = group_curves(args.runs_dir, prefix, "fitness")
        times = wall_clock(args.runs_dir, prefix, steps).mean(axis=0)
        label = f"{prefix} ({seeds} seeds)"

        plot_band(steps_axis, steps, worst, label)
        plot_band(time_axis, times, worst, label)

    steps_axis.set_xlabel("environment steps")
    time_axis.set_xlabel("training time (s, mean over seeds)")

    for axis, title in (
        (steps_axis, "Worst-track fitness vs environment steps"),
        (time_axis, "Worst-track fitness vs training time"),
    ):
        axis.set_title(title)
        axis.set_ylabel("worst-track evaluation fitness")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)

    return figure


def plot_band(axis, steps, values, label):
    mean = values.mean(axis=0)
    axis.plot(steps, mean, label=label)
    axis.fill_between(steps, values.min(axis=0), values.max(axis=0), alpha=0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prefixes", nargs="+")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", default=None)
    parser.add_argument("--overlay", action="store_true",
                        help="compare groups on shared axes")
    args = parser.parse_args()

    if args.overlay:
        save_or_show(plot_overlay(args), args.output)
        return

    figure, axes = plt.subplots(
        len(args.prefixes), 2,
        figsize=(12, 4 * len(args.prefixes)),
        squeeze=False
    )

    for row, prefix in enumerate(args.prefixes):
        steps, worst, per_track, seeds = group_curves(
            args.runs_dir, prefix, "fitness"
        )
        _, _, laps, _ = group_curves(args.runs_dir, prefix, "laps")

        fitness_axis, laps_axis = axes[row]

        plot_band(fitness_axis, steps, worst, "worst track")
        for track, values in per_track.items():
            fitness_axis.plot(steps, values.mean(axis=0), "--", lw=1, label=track)

        fitness_axis.set_title(f"{prefix}: evaluation fitness ({seeds} seeds)")
        fitness_axis.set_xlabel("environment steps")
        fitness_axis.set_ylabel("fitness")
        fitness_axis.legend(fontsize=8)
        fitness_axis.grid(alpha=0.3)

        for track, values in laps.items():
            plot_band(laps_axis, steps, values, track)

        laps_axis.set_title(f"{prefix}: evaluation laps (mean, min-max)")
        laps_axis.set_xlabel("environment steps")
        laps_axis.set_ylabel("laps in 1800 frames")
        laps_axis.legend(fontsize=8)
        laps_axis.grid(alpha=0.3)

    save_or_show(figure, args.output)


def save_or_show(figure, output):
    figure.tight_layout()

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        figure.savefig(output, dpi=110)
        print(f"Saved {output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
