"""Anthropic provider adapter. Requires: pip install provably[anthropic]"""

from __future__ import annotations

import os

from provably.providers.base import Provider
from provably.result import LLMResult, ToolCall


_COSTS = {
    "claude-opus-4-6": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004},
}


class AnthropicProvider(Provider):
    """Thin wrapper around the Anthropic Python SDK."""

    def __init__(self, api_key: str | None = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic SDK not installed. Run: pip install provably[anthropic]"
            )
        self._client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
        )

    def name(self) -> str:
        return "anthropic"

    def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> LLMResult:
        model = model or "claude-sonnet-4-6"
        start = self._time()

        # Extract system message if present
        system = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        params = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            **kwargs,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = tools

        response = self._client.messages.create(**params)
        latency = self._time() - start

        # Extract text and tool calls
        text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(name=block.name, args=block.input or {})
                )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = self._estimate_cost(model, input_tokens, output_tokens)

        return LLMResult(
            text=text,
            tool_calls=tool_calls,
            cost=cost,
            latency=latency,
            model=model,
            provider="anthropic",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        costs = _COSTS.get(model, {"input": 0.003, "output": 0.015})
        return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1000
