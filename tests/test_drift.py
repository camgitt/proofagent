"""Tests for drift detection."""

import json

from proofagent.drift import DriftReport, detect_drift, list_history, save_history


def _make_results(statuses: dict[str, str], cost: float = 0.01) -> list[dict]:
    """Helper to create result dicts from a name->status mapping."""
    return [
        {"name": name, "status": status, "cost": cost}
        for name, status in statuses.items()
    ]


def test_save_and_list_history(tmp_path):
    results = _make_results({"test_a": "passed", "test_b": "failed"})
    save_history(results, model="gpt-4o", history_dir=tmp_path)

    entries = list_history(tmp_path)
    assert len(entries) == 1
    assert entries[0]["model"] == "gpt-4o"
    assert entries[0]["tests"]["test_a"]["status"] == "passed"
    assert entries[0]["tests"]["test_b"]["status"] == "failed"


def test_detect_drift_regression(tmp_path):
    # Baseline: both pass
    baseline = _make_results({"test_safety": "passed", "test_refusal": "passed"})
    save_history(baseline, model="gpt-4o", history_dir=tmp_path)

    # Write a second file with a later timestamp
    latest = _make_results({"test_safety": "failed", "test_refusal": "passed"})
    entry = {
        "timestamp": "2099-12-31_235959",
        "model": "gpt-4o",
        "score": 0.5,
        "total_cost": 0.02,
        "passed": 1,
        "failed": 1,
        "total": 2,
        "tests": {
            "test_safety": {"status": "failed", "cost": 0.01},
            "test_refusal": {"status": "passed", "cost": 0.01},
        },
    }
    with open(tmp_path / "2099-12-31_235959.json", "w") as f:
        json.dump(entry, f)

    report = detect_drift(history_dir=tmp_path)
    assert report is not None
    assert report.is_regression
    assert len(report.regressions) == 1
    assert report.regressions[0]["name"] == "test_safety"


def test_detect_drift_improvement(tmp_path):
    # Baseline: test_json fails
    baseline = _make_results({"test_json": "failed"})
    save_history(baseline, model="gpt-4o", history_dir=tmp_path)

    # Latest: test_json passes
    entry = {
        "timestamp": "2099-12-31_235959",
        "model": "gpt-4o",
        "score": 1.0,
        "total_cost": 0.01,
        "passed": 1,
        "failed": 0,
        "total": 1,
        "tests": {"test_json": {"status": "passed", "cost": 0.01}},
    }
    with open(tmp_path / "2099-12-31_235959.json", "w") as f:
        json.dump(entry, f)

    report = detect_drift(history_dir=tmp_path)
    assert report is not None
    assert not report.is_regression
    assert len(report.improvements) == 1
    assert report.improvements[0]["name"] == "test_json"


def test_detect_drift_not_enough_history(tmp_path):
    results = _make_results({"test_a": "passed"})
    save_history(results, model="gpt-4o", history_dir=tmp_path)

    report = detect_drift(history_dir=tmp_path)
    assert report is None


def test_detect_drift_model_filter(tmp_path):
    # Save one run for gpt-4o and one for claude
    save_history(
        _make_results({"test_a": "passed"}), model="gpt-4o", history_dir=tmp_path
    )

    entry = {
        "timestamp": "2099-12-31_235959",
        "model": "claude-sonnet",
        "score": 1.0,
        "total_cost": 0.01,
        "passed": 1,
        "failed": 0,
        "total": 1,
        "tests": {"test_a": {"status": "passed", "cost": 0.01}},
    }
    with open(tmp_path / "2099-12-31_235959.json", "w") as f:
        json.dump(entry, f)

    # Filtering by claude-sonnet should only find 1 entry, not enough
    report = detect_drift(model="claude-sonnet", history_dir=tmp_path)
    assert report is None
