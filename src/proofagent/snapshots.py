"""Snapshot testing for LLM and agent outputs.

Inspired by Jest snapshots — saves expected outputs to disk and compares
future runs against them. First run creates the snapshot; subsequent runs
assert the output matches.

Usage:
    from proofagent import expect, LLMResult

    result = LLMResult(text="The answer is 4.")
    expect(result).matches_snapshot("math_answer")
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

DEFAULT_SNAPSHOT_DIR = Path(".proofagent/snapshots")


def _snapshot_path(name: str, snapshot_dir: Path | None = None) -> Path:
    """Return the path for a named snapshot file."""
    base = snapshot_dir or DEFAULT_SNAPSHOT_DIR
    return base / f"{name}.json"


def save_snapshot(name: str, data: dict, snapshot_dir: Path | None = None) -> Path:
    """Save a snapshot to disk.

    Args:
        name: Snapshot name (used as filename stem).
        data: Dict to serialize as the snapshot.
        snapshot_dir: Override snapshot directory (useful for testing).

    Returns:
        Path to the saved snapshot file.
    """
    path = _snapshot_path(name, snapshot_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def load_snapshot(name: str, snapshot_dir: Path | None = None) -> dict | None:
    """Load a snapshot from disk, or return None if it doesn't exist."""
    path = _snapshot_path(name, snapshot_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def delete_all_snapshots(snapshot_dir: Path | None = None) -> int:
    """Delete all snapshot files. Returns the count of files deleted."""
    base = snapshot_dir or DEFAULT_SNAPSHOT_DIR
    if not base.exists():
        return 0
    count = 0
    for f in base.glob("*.json"):
        f.unlink()
        count += 1
    return count


def list_snapshots(snapshot_dir: Path | None = None) -> list[str]:
    """Return sorted list of snapshot names (without .json extension)."""
    base = snapshot_dir or DEFAULT_SNAPSHOT_DIR
    if not base.exists():
        return []
    return sorted(f.stem for f in base.glob("*.json"))


def diff_snapshot(name: str, current: dict, snapshot_dir: Path | None = None) -> str:
    """Return a unified diff between saved snapshot and current data.

    Returns empty string if they match.
    """
    saved = load_snapshot(name, snapshot_dir)
    if saved is None:
        return ""
    saved_lines = json.dumps(saved, indent=2, sort_keys=True).splitlines(keepends=True)
    current_lines = json.dumps(current, indent=2, sort_keys=True).splitlines(keepends=True)
    diff = difflib.unified_diff(
        saved_lines, current_lines, fromfile="snapshot", tofile="current", lineterm=""
    )
    return "\n".join(diff)
