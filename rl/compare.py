"""Evaluate several models on every track and write a Markdown table.

Arguments are model files (.npz, .pt, .keras) or run-group prefixes. A
prefix such as ``torch_multi`` evaluates runs/torch_multi_seed*/best.* and
reports the mean over seeds plus every seed separately.

    python -m rl.compare best_brain_multitrack.npz torch_multi --output results/rl/table.md
"""

import argparse
import glob
import os
import statistics

from rl.evaluate import EVAL_SEED, evaluate_policy, load_policy


def expand(name, runs_dir):
    """[(label, [model paths])] for a model file or a run-group prefix."""
    if os.path.isfile(name):
        return [(os.path.basename(name), [name])]

    paths = sorted(glob.glob(os.path.join(runs_dir, name + "_seed*", "best.*")))

    if not paths:
        raise FileNotFoundError(f"No model file or runs for {name!r}")

    entries = [(f"{name} (mean of {len(paths)} seeds)", paths)]

    for path in paths:
        entries.append((f"  {os.path.basename(os.path.dirname(path))}", [path]))

    return entries


def average(results_list):
    """Mean of per-track metrics over several evaluation results."""
    tracks = results_list[0].keys()

    return {
        track: {
            key: statistics.mean(r[track][key] for r in results_list)
            for key in ("laps", "crash_rate", "fitness")
        }
        for track in tracks
    }


def markdown_table(rows, tracks):
    header = "| Model | " + " | ".join(tracks) + " | Worst-track fitness |"
    divider = "|---|" + "---:|" * (len(tracks) + 1)

    lines = [header, divider]

    for label, results in rows:
        cells = [
            f"{results[t]['laps']:.1f} laps, {results[t]['crash_rate']:.0%} crash, "
            f"{results[t]['fitness']:,.0f}"
            for t in tracks
        ]
        # Worst over the training tracks only (the first three).
        worst = min(results[t]["fitness"] for t in tracks[:3])
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {worst:,.0f} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+")
    parser.add_argument("--tracks", default="all")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output")
    args = parser.parse_args()

    cache = {}
    rows = []

    for name in args.models:
        for label, paths in expand(name, args.runs_dir):
            for path in paths:
                if path not in cache:
                    cache[path] = evaluate_policy(
                        load_policy(path), args.tracks, args.episodes
                    )

            rows.append((label, average([cache[p] for p in paths])))

    tracks = list(rows[0][1])

    text = (
        f"Each cell: mean laps, crash rate and fitness over {args.episodes} "
        f"episodes per track (evaluation seeds {EVAL_SEED}-"
        f"{EVAL_SEED + args.episodes - 1}, deterministic actions). "
        f"Worst-track fitness is taken over the three training tracks.\n\n"
        + markdown_table(rows, tracks)
    )

    print(text)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(text + "\n")


if __name__ == "__main__":
    main()
