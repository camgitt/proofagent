"""OpenAI provider adapter. Requires: pip install provably[openai]"""

from __future__ import annotations

import os

from provably.providers.base import Provider
from provably.result import LLMResult, ToolCall


# Approximate cost per 1K tokens (USD) — updated for common models
_COSTS = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "o1": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
}


class OpenAIProvider(Provider):
    """Thin wrapper around the OpenAI Python SDK."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI SDK not installed. Run: pip install provably[openai]"
            )
        self._client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def name(self) -> str:
        return "openai"

    def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> LLMResult:
        model = model or "gpt-4o-mini"
        start = self._time()

        params = {"model": model, "messages": messages, **kwargs}
        if tools:
            params["tools"] = tools

        response = self._client.chat.completions.create(**params)
        latency = self._time() - start

        choice = response.choices[0]
        text = choice.message.content or ""

        # Extract tool calls
        tool_calls = []
        if choice.message.tool_calls:
            import json

            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        args=json.loads(tc.function.arguments)
                        if tc.function.arguments
                        else {},
                    )
                )

        # Calculate cost
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = self._estimate_cost(model, input_tokens, output_tokens)

        return LLMResult(
            text=text,
            tool_calls=tool_calls,
            cost=cost,
            latency=latency,
            model=model,
            provider="openai",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        costs = _COSTS.get(model, {"input": 0.001, "output": 0.002})
        return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1000
