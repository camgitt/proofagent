"""Google Gemini provider adapter. Requires: pip install proofagent[gemini]"""

from __future__ import annotations

import os

from proofagent.providers.base import Provider
from proofagent.result import LLMResult, ToolCall


# Approximate cost per 1K tokens (USD) — updated March 2026
_COSTS = {
    "gemini-3.1-pro-preview": {"input": 0.002, "output": 0.012},
    "gemini-3-flash-preview": {"input": 0.0005, "output": 0.003},
    "gemini-2.5-flash": {"input": 0.00015, "output": 0.001},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
}


class GeminiProvider(Provider):
    """Thin wrapper around the Google GenAI Python SDK."""

    def __init__(self, api_key: str | None = None):
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "Google GenAI SDK not installed. Run: pip install proofagent[gemini]"
            )
        self._client = genai.Client(
            api_key=api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        )

    def name(self) -> str:
        return "gemini"

    def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> LLMResult:
        from google.genai import types

        model = model or "gemini-2.5-flash"
        start = self._time()

        # Convert OpenAI-style messages to Gemini format
        contents = []
        system_instruction = None
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=content)]))
            else:
                contents.append(types.Content(role="user", parts=[types.Part(text=content)]))

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            **kwargs,
        )

        # Convert OpenAI-style tools to Gemini function declarations
        if tools:
            declarations = []
            for tool in tools:
                if "function" in tool:
                    fn = tool["function"]
                    declarations.append(types.FunctionDeclaration(
                        name=fn["name"],
                        description=fn.get("description", ""),
                        parameters=fn.get("parameters"),
                    ))
            if declarations:
                config.tools = [types.Tool(function_declarations=declarations)]

        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        latency = self._time() - start

        text = response.text or ""

        # Extract tool calls
        tool_calls = []
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    tool_calls.append(
                        ToolCall(
                            name=part.function_call.name,
                            args=dict(part.function_call.args) if part.function_call.args else {},
                        )
                    )

        # Calculate cost from usage metadata
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0
        cost = self._estimate_cost(model, input_tokens, output_tokens)

        return LLMResult(
            text=text,
            tool_calls=tool_calls,
            cost=cost,
            latency=latency,
            model=model,
            provider="gemini",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        costs = _COSTS.get(model, {"input": 0.0001, "output": 0.0004})
        return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1000
