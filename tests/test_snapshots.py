"""Tests for snapshot regression testing."""

import json

import pytest

from proofagent import LLMResult, ToolCall, expect
from proofagent.snapshots import (
    delete_all_snapshots,
    list_snapshots,
    load_snapshot,
    save_snapshot,
)


def test_first_run_creates_snapshot(tmp_path):
    """First run should save the snapshot and pass."""
    result = LLMResult(text="The answer is 4.")
    expect(result).matches_snapshot("math_answer", snapshot_dir=tmp_path)

    saved = load_snapshot("math_answer", snapshot_dir=tmp_path)
    assert saved is not None
    assert saved["text"] == "The answer is 4."


def test_matching_snapshot_passes(tmp_path):
    """Same output on second run should pass."""
    result = LLMResult(text="Hello, world!", cost=0.001)
    expect(result).matches_snapshot("hello", snapshot_dir=tmp_path)

    # Run again with identical result
    result2 = LLMResult(text="Hello, world!", cost=0.001)
    expect(result2).matches_snapshot("hello", snapshot_dir=tmp_path)


def test_mismatched_snapshot_fails(tmp_path):
    """Different output should raise AssertionError with diff."""
    result = LLMResult(text="The answer is 4.")
    expect(result).matches_snapshot("math", snapshot_dir=tmp_path)

    changed = LLMResult(text="The answer is 5.")
    with pytest.raises(AssertionError, match="Snapshot 'math' does not match"):
        expect(changed).matches_snapshot("math", snapshot_dir=tmp_path)


def test_list_and_delete_snapshots(tmp_path):
    """list_snapshots and delete_all_snapshots should work correctly."""
    save_snapshot("alpha", {"text": "a"}, snapshot_dir=tmp_path)
    save_snapshot("beta", {"text": "b"}, snapshot_dir=tmp_path)

    names = list_snapshots(snapshot_dir=tmp_path)
    assert names == ["alpha", "beta"]

    count = delete_all_snapshots(snapshot_dir=tmp_path)
    assert count == 2
    assert list_snapshots(snapshot_dir=tmp_path) == []


def test_snapshot_chainable(tmp_path):
    """matches_snapshot should be chainable with other assertions."""
    result = LLMResult(
        text="The answer is 4.",
        tool_calls=[ToolCall(name="calculate", args={"expr": "2+2"})],
    )
    (
        expect(result)
        .matches_snapshot("chain_test", snapshot_dir=tmp_path)
        .contains("4")
        .tool_calls_contain("calculate")
    )
