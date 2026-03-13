"""Result storage and reporting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from provably import display


def save_results(
    results: list[dict],
    output_dir: str | Path = ".provably/results",
) -> Path:
    """Save eval results to a JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eval_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "results": results,
                "summary": _summarize(results),
            },
            f,
            indent=2,
        )

    return output_file


def load_latest_results(results_dir: str | Path = ".provably/results") -> dict | None:
    """Load the most recent results file."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return None

    files = sorted(results_dir.glob("eval_*.json"), reverse=True)
    if not files:
        return None

    with open(files[0]) as f:
        return json.load(f)


def print_summary(data: dict) -> None:
    """Print a formatted summary to terminal."""
    summary = data.get("summary", {})
    results = data.get("results", [])

    print(display.header("Provably Eval Report"))
    print(
        f"  Score: {display.format_score(summary.get('passed', 0), summary.get('total', 0))}"
    )
    print(f"  Cost:  {display.cost_text(summary.get('total_cost', 0))}")
    print(f"  Time:  {summary.get('total_latency', 0):.1f}s")
    print()

    # Show failures
    failures = [r for r in results if r.get("status") == "failed"]
    if failures:
        print(display.fail_text(f"  {len(failures)} failed:"))
        for f in failures:
            print(f"    {display.fail_text('FAIL')} {f.get('name', '?')}")
            if f.get("error"):
                print(f"         {display.DIM}{f['error'][:100]}{display.RST}")
        print()


def _summarize(results: list[dict]) -> dict:
    """Create a summary dict from results."""
    passed = sum(1 for r in results if r.get("status") == "passed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    total = passed + failed
    total_cost = sum(r.get("cost", 0) for r in results)
    total_latency = sum(r.get("latency", 0) for r in results)

    return {
        "passed": passed,
        "failed": failed,
        "total": total,
        "score": passed / total if total > 0 else 0,
        "total_cost": total_cost,
        "total_latency": total_latency,
    }
