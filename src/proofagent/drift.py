"""Model drift detection — compare eval runs over time."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


HISTORY_DIR = ".proofagent/history"


@dataclass
class DriftReport:
    """Result of comparing two eval runs."""

    regressions: list[dict] = field(default_factory=list)
    improvements: list[dict] = field(default_factory=list)
    cost_change: float = 0.0
    score_change: float = 0.0
    is_regression: bool = False
    baseline_timestamp: str = ""
    latest_timestamp: str = ""


def save_history(results: list[dict], model: str, history_dir: str | Path = HISTORY_DIR) -> Path:
    """Save eval results to history for drift tracking."""
    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    output_file = history_dir / f"{timestamp}.json"

    passed = sum(1 for r in results if r.get("status") == "passed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    total = passed + failed
    total_cost = sum(r.get("cost", 0) for r in results)

    entry = {
        "timestamp": timestamp,
        "model": model,
        "score": passed / total if total > 0 else 0,
        "total_cost": total_cost,
        "passed": passed,
        "failed": failed,
        "total": total,
        "tests": {
            r.get("name", f"test_{i}"): {
                "status": r.get("status", "unknown"),
                "cost": r.get("cost", 0),
            }
            for i, r in enumerate(results)
        },
    }

    with open(output_file, "w") as f:
        json.dump(entry, f, indent=2)

    return output_file


def list_history(
    history_dir: str | Path = HISTORY_DIR,
    model: str | None = None,
) -> list[dict]:
    """Load all history entries, newest first."""
    history_dir = Path(history_dir)
    if not history_dir.exists():
        return []

    entries = []
    for path in sorted(history_dir.glob("*.json"), reverse=True):
        with open(path) as f:
            entry = json.load(f)
        if model and entry.get("model") != model:
            continue
        entries.append(entry)

    return entries


def detect_drift(
    since: int = 1,
    model: str | None = None,
    history_dir: str | Path = HISTORY_DIR,
) -> DriftReport | None:
    """Compare the latest run against a previous run.

    Args:
        since: How many runs back to compare against (1 = previous run).
        model: Filter history by model name.
        history_dir: Path to history directory.

    Returns:
        DriftReport or None if not enough history.
    """
    entries = list_history(history_dir, model=model)

    if len(entries) < since + 1:
        return None

    latest = entries[0]
    baseline = entries[since]

    regressions = []
    improvements = []

    latest_tests = latest.get("tests", {})
    baseline_tests = baseline.get("tests", {})

    for name, info in latest_tests.items():
        baseline_info = baseline_tests.get(name)
        if baseline_info is None:
            continue

        old_status = baseline_info["status"]
        new_status = info["status"]

        if old_status == "passed" and new_status == "failed":
            regressions.append({"name": name, "old": old_status, "new": new_status})
        elif old_status == "failed" and new_status == "passed":
            improvements.append({"name": name, "old": old_status, "new": new_status})

    baseline_cost = baseline.get("total_cost", 0)
    latest_cost = latest.get("total_cost", 0)
    cost_change = (
        ((latest_cost - baseline_cost) / baseline_cost * 100)
        if baseline_cost > 0
        else 0.0
    )

    baseline_score = baseline.get("score", 0)
    latest_score = latest.get("score", 0)
    score_change = (latest_score - baseline_score) * 100  # percentage points

    return DriftReport(
        regressions=regressions,
        improvements=improvements,
        cost_change=cost_change,
        score_change=score_change,
        is_regression=len(regressions) > 0,
        baseline_timestamp=baseline.get("timestamp", ""),
        latest_timestamp=latest.get("timestamp", ""),
    )
