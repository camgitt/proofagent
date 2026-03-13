"""Tests for LLMResult and related data structures."""

from provably import LLMResult, ToolCall, TrajectoryStep


def test_llm_result_basic():
    r = LLMResult(text="Hello")
    assert r.text == "Hello"
    assert r.cost == 0.0
    assert r.latency == 0.0
    assert r.tool_calls == []
    assert r.trajectory == []


def test_llm_result_with_tool_calls():
    tc = ToolCall(name="get_weather", args={"city": "NYC"}, result="72F")
    r = LLMResult(text="It's 72F in NYC", tool_calls=[tc])
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "get_weather"
    assert r.tool_calls[0].args == {"city": "NYC"}


def test_llm_result_with_trajectory():
    steps = [
        TrajectoryStep(role="user", content="What's the weather?"),
        TrajectoryStep(
            role="assistant",
            content="",
            tool_calls=[ToolCall(name="get_weather", args={"city": "NYC"})],
        ),
        TrajectoryStep(role="tool", content="72F"),
        TrajectoryStep(role="assistant", content="It's 72F in NYC"),
    ]
    r = LLMResult(text="It's 72F in NYC", trajectory=steps)
    assert len(r.trajectory) == 4
    assert r.trajectory[0].role == "user"
    assert r.trajectory[1].tool_calls[0].name == "get_weather"


def test_llm_result_to_dict():
    tc = ToolCall(name="search", args={"q": "test"})
    r = LLMResult(text="result", tool_calls=[tc], cost=0.01, model="gpt-4o")
    d = r.to_dict()
    assert d["text"] == "result"
    assert d["tool_calls"][0]["name"] == "search"
    assert d["cost"] == 0.01
    assert d["model"] == "gpt-4o"


def test_tool_call_to_dict():
    tc = ToolCall(name="fn", args={"x": 1}, result="ok")
    assert tc.to_dict() == {"name": "fn", "args": {"x": 1}, "result": "ok"}
