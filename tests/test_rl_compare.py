import os

import pytest

from rl.compare import average, expand, markdown_table


def fake_results(laps, fitness):
    return {
        name: {"laps": laps, "crash_rate": 0.0, "fitness": fitness + i}
        for i, name in enumerate(["A", "B", "C", "Unseen"])
    }


def test_average_over_seeds():
    mean = average([fake_results(2, 100), fake_results(4, 300)])

    assert mean["A"]["laps"] == 3
    assert mean["B"]["fitness"] == 201


def test_markdown_table_worst_uses_training_tracks():
    table = markdown_table([("model", fake_results(1, 50))], ["A", "B", "C", "Unseen"])
    lines = table.splitlines()

    assert lines[0].startswith("| Model | A | B | C | Unseen |")
    # Worst over A, B, C = 50 even though every track differs.
    assert lines[2].endswith("| 50 |")


def test_expand_run_group(tmp_path):
    for seed in (0, 1):
        run = tmp_path / f"group_seed{seed}"
        run.mkdir()
        (run / "best.pt").write_bytes(b"")

    entries = expand("group", str(tmp_path))

    assert entries[0][0] == "group (mean of 2 seeds)"
    assert len(entries[0][1]) == 2
    assert [label.strip() for label, _ in entries[1:]] == ["group_seed0", "group_seed1"]

    with pytest.raises(FileNotFoundError):
        expand("missing", str(tmp_path))
