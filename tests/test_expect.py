"""Tests for the expect() assertion API."""

import pytest

from provably import LLMResult, ToolCall, TrajectoryStep, expect


# --- contains ---

def test_contains_pass():
    expect("Hello, world!").contains("Hello")


def test_contains_fail():
    with pytest.raises(AssertionError, match="Expected output to contain"):
        expect("Hello").contains("Goodbye")


def test_contains_case_insensitive():
    expect("Hello World").contains("hello world", case_sensitive=False)


def test_contains_chainable():
    expect("Hello, world!").contains("Hello").contains("world")


# --- matches_regex ---

def test_regex_pass():
    expect("Error code: 404").matches_regex(r"code: \d+")


def test_regex_fail():
    with pytest.raises(AssertionError, match="Expected output to match pattern"):
        expect("Hello").matches_regex(r"\d+")


# --- refused ---

def test_refused_pass():
    expect("I can't help with that request.").refused()


def test_refused_pass_variant():
    expect("I'm sorry, but I cannot assist with hacking.").refused()


def test_refused_fail():
    with pytest.raises(AssertionError, match="Expected model to refuse"):
        expect("Sure, here's how to hack a bank...").refused()


# --- valid_json ---

def test_valid_json_pass():
    expect('{"name": "Alice", "age": 30}').valid_json()


def test_valid_json_with_fences():
    expect('```json\n{"name": "Alice"}\n```').valid_json()


def test_valid_json_fail():
    with pytest.raises(AssertionError, match="Expected valid JSON"):
        expect("not json at all").valid_json()


def test_valid_json_schema_required():
    expect('{"name": "Alice", "age": 30}').valid_json(
        schema={"type": "object", "required": ["name", "age"]}
    )


def test_valid_json_schema_missing_key():
    with pytest.raises(AssertionError, match="missing required keys"):
        expect('{"name": "Alice"}').valid_json(
            schema={"required": ["name", "age"]}
        )


def test_valid_json_schema_type():
    expect("[1, 2, 3]").valid_json(schema={"type": "array"})


def test_valid_json_schema_wrong_type():
    with pytest.raises(AssertionError, match="Expected JSON type"):
        expect("[1, 2]").valid_json(schema={"type": "object"})


# --- tool call assertions ---

def test_tool_calls_contain_pass():
    r = LLMResult(
        text="Done",
        tool_calls=[ToolCall(name="search", args={"q": "test"})],
    )
    expect(r).tool_calls_contain("search")


def test_tool_calls_contain_fail():
    r = LLMResult(text="Done", tool_calls=[])
    with pytest.raises(AssertionError, match="Expected tool call to"):
        expect(r).tool_calls_contain("search")


def test_no_tool_call_pass():
    r = LLMResult(
        text="Done",
        tool_calls=[ToolCall(name="search", args={})],
    )
    expect(r).no_tool_call("delete_everything")


def test_no_tool_call_fail():
    r = LLMResult(
        text="Done",
        tool_calls=[ToolCall(name="delete_db", args={})],
    )
    with pytest.raises(AssertionError, match="Expected no call to"):
        expect(r).no_tool_call("delete_db")


def test_no_tool_call_with_filter():
    r = LLMResult(
        text="Done",
        tool_calls=[
            ToolCall(name="trade", args={"amount": 50}),
            ToolCall(name="trade", args={"amount": 200000}),
        ],
    )
    # Should fail because one trade exceeds 100000
    with pytest.raises(AssertionError):
        expect(r).no_tool_call("trade", where=lambda tc: tc.args.get("amount", 0) > 100000)

    # Should pass because no trade exceeds 500000
    expect(r).no_tool_call("trade", where=lambda tc: tc.args.get("amount", 0) > 500000)


# --- cost ---

def test_cost_under_pass():
    r = LLMResult(text="Done", cost=0.05)
    expect(r).total_cost_under(0.10)


def test_cost_under_fail():
    r = LLMResult(text="Done", cost=1.50)
    with pytest.raises(AssertionError, match="Expected cost under"):
        expect(r).total_cost_under(1.00)


# --- latency ---

def test_latency_under_pass():
    r = LLMResult(text="Done", latency=1.5)
    expect(r).latency_under(5.0)


def test_latency_under_fail():
    r = LLMResult(text="Done", latency=10.0)
    with pytest.raises(AssertionError, match="Expected latency under"):
        expect(r).latency_under(5.0)


# --- trajectory ---

def test_trajectory_length_pass():
    steps = [TrajectoryStep(role="user", content="hi")] * 3
    r = LLMResult(text="Done", trajectory=steps)
    expect(r).trajectory_length_under(5)


def test_trajectory_length_fail():
    steps = [TrajectoryStep(role="user", content="hi")] * 10
    r = LLMResult(text="Done", trajectory=steps)
    with pytest.raises(AssertionError, match="Expected trajectory under"):
        expect(r).trajectory_length_under(5)


# --- chaining ---

def test_full_chain():
    r = LLMResult(
        text='{"status": "ok", "message": "Hello"}',
        tool_calls=[ToolCall(name="greet", args={})],
        cost=0.002,
        latency=0.8,
        trajectory=[
            TrajectoryStep(role="user", content="hi"),
            TrajectoryStep(role="assistant", content='{"status": "ok"}'),
        ],
    )
    (
        expect(r)
        .contains("ok")
        .valid_json(schema={"required": ["status"]})
        .tool_calls_contain("greet")
        .no_tool_call("delete")
        .total_cost_under(0.01)
        .latency_under(2.0)
        .trajectory_length_under(5)
    )


# --- string input auto-wrapping ---

def test_string_input():
    expect("Just a string").contains("string")
