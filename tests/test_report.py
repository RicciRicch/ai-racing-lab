import csv

from rl.report import first_success, format_median, steps_text, training_totals


def write_csv(path, rows):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_run(tmp_path):
    write_csv(tmp_path / "progress.csv", [
        {"global_step": 100, "time": 1.0},
        {"global_step": 200, "time": 2.5},
        {"global_step": 300, "time": 4.0},
    ])
    write_csv(tmp_path / "eval.csv", [
        {"global_step": 100, "track": "A", "laps": 2, "fitness": 500},
        {"global_step": 100, "track": "B", "laps": 9, "fitness": 3000},
        {"global_step": 200, "track": "A", "laps": 9, "fitness": 2500},
        {"global_step": 200, "track": "B", "laps": 9, "fitness": 2100},
        {"global_step": 300, "track": "A", "laps": 9, "fitness": 2600},
        {"global_step": 300, "track": "B", "laps": 9, "fitness": 2700},
    ])
    return str(tmp_path)


def test_first_success_requires_all_tracks(tmp_path):
    run = make_run(tmp_path)

    laps_target = lambda rows: all(float(r["laps"]) >= 9 for r in rows)
    assert first_success(run, laps_target) == (200, 2.5)

    fitness_target = lambda rows: min(float(r["fitness"]) for r in rows) >= 2600
    assert first_success(run, fitness_target) == (300, 4.0)

    never = lambda rows: False
    assert first_success(run, never) == (None, None)


def test_training_totals(tmp_path):
    assert training_totals(make_run(tmp_path)) == (300, 4.0)


def test_format_median_reports_missing_seeds():
    assert format_median([100, 300, None], str) == "200.0 (2/3 seeds)"
    assert format_median([None, None], str) == "not reached"
    assert steps_text(40_960) == "41k"
    assert steps_text(2_500_000) == "2.50M"
