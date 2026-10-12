import os

import numpy as np
import pytest

from rl.evaluate import evaluate_policy, format_results, load_policy


def idle(observation):
    return np.array([0.0, -1.0], np.float32)


def test_idle_policy_never_crashes_and_scores_nothing():
    results = evaluate_policy(idle, tracks="all", episodes=2, max_steps=50)

    assert len(results) == 4

    for metrics in results.values():
        assert metrics["crash_rate"] == 0
        assert metrics["laps"] == 0
        assert metrics["frames"] == 50


def test_evaluation_is_deterministic():
    full = lambda observation: np.array([0.3, 1.0], np.float32)

    first = evaluate_policy(full, tracks="train", episodes=2)
    second = evaluate_policy(full, tracks="train", episodes=2)

    assert first == second


def test_format_results_has_one_row_per_track():
    table = format_results(evaluate_policy(idle, "train", 1, max_steps=5))

    assert len(table.splitlines()) == 4


@pytest.mark.skipif(
    not os.path.exists("best_brain_multitrack.npz"),
    reason="trained multi-track model not available"
)
def test_multitrack_brain_reproduces_unseen_track_result():
    # results/unseen_track_multitrack_model.txt: 8 laps, 0% crashes.
    results = evaluate_policy(
        load_policy("best_brain_multitrack.npz"), "unseen", episodes=5
    )

    unseen = results["Unseen Narrow Oval"]
    assert unseen["laps"] == 8
    assert unseen["crash_rate"] == 0


def test_unknown_model_type():
    with pytest.raises(ValueError):
        load_policy("model.onnx")
