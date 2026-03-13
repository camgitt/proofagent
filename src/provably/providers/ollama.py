"""Ollama provider adapter for local models. No API key needed."""

from __future__ import annotations

import json

import requests

from provably.providers.base import Provider
from provably.result import LLMResult, ToolCall


class OllamaProvider(Provider):
    """Thin wrapper around Ollama's REST API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self._base_url = base_url.rstrip("/")

    def name(self) -> str:
        return "ollama"

    def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> LLMResult:
        model = model or "llama3"
        start = self._time()

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs,
        }
        if tools:
            payload["tools"] = tools

        resp = requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        latency = self._time() - start

        message = data.get("message", {})
        text = message.get("content", "")

        tool_calls = []
        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            tool_calls.append(
                ToolCall(
                    name=func.get("name", ""),
                    args=func.get("arguments", {}),
                )
            )

        return LLMResult(
            text=text,
            tool_calls=tool_calls,
            cost=0.0,  # local models are free
            latency=latency,
            model=model,
            provider="ollama",
        )
