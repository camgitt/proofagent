"""Base provider protocol and shared types."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from provably.result import LLMResult, ToolCall, TrajectoryStep


class Provider(ABC):
    """Abstract base for LLM providers.

    Each provider adapter wraps a specific SDK and returns LLMResult objects
    with timing, cost, and tool call data populated.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> LLMResult:
        """Send a completion request and return a structured result."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic', 'ollama')."""
        ...

    def _time(self):
        """Return current time for latency tracking."""
        return time.monotonic()
