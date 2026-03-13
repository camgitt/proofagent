"""Pytest fixtures for provably evaluations."""

from __future__ import annotations

import pytest

from provably.config import load_config
from provably.providers import get_provider
from provably.result import LLMResult


@pytest.fixture
def provably_config():
    """Load provably configuration."""
    return load_config()


@pytest.fixture
def provably_provider(provably_config):
    """Get a configured LLM provider instance."""
    return get_provider(name=provably_config.get("provider"))


@pytest.fixture
def provably_run(provably_provider, provably_config):
    """Run an LLM completion and return an LLMResult.

    Usage:
        def test_greeting(provably_run):
            result = provably_run("Say hello", model="gpt-4o-mini")
            expect(result).contains("hello")
    """

    def _run(
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> LLMResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return provably_provider.complete(
            messages=messages,
            model=model or provably_config.get("model"),
            tools=tools,
            **kwargs,
        )

    return _run
