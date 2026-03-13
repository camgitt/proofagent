"""Basic example — no API key needed for these tests."""

from provably import LLMResult, ToolCall, expect


def test_json_output():
    """Test that output is valid JSON with required fields."""
    result = LLMResult(text='{"name": "Alice", "score": 95}')
    expect(result).valid_json(schema={"required": ["name", "score"]})


def test_contains_greeting():
    """Test output contains expected content."""
    result = LLMResult(text="Hello! How can I help you today?")
    expect(result).contains("Hello").contains("help")


def test_cost_and_latency():
    """Test cost and latency are within bounds."""
    result = LLMResult(text="Response", cost=0.003, latency=1.2)
    expect(result).total_cost_under(0.01).latency_under(5.0)


def test_tool_usage():
    """Test agent used the right tools."""
    result = LLMResult(
        text="The weather in NYC is 72F",
        tool_calls=[ToolCall(name="get_weather", args={"city": "NYC"})],
    )
    expect(result).tool_calls_contain("get_weather").no_tool_call("delete_data")
