"""Provider registry — get a provider by name or auto-detect from env."""

from __future__ import annotations

import os

from provably.providers.base import Provider


def get_provider(name: str | None = None, **kwargs) -> Provider:
    """Get a provider instance by name, or auto-detect from environment.

    Args:
        name: Provider name ('openai', 'anthropic', 'ollama').
              If None, auto-detects based on available API keys.

    Returns:
        A Provider instance ready to use.
    """
    if name is None:
        name = _detect_provider()

    if name == "openai":
        from provably.providers.openai import OpenAIProvider

        return OpenAIProvider(**kwargs)
    elif name == "anthropic":
        from provably.providers.anthropic import AnthropicProvider

        return AnthropicProvider(**kwargs)
    elif name == "ollama":
        from provably.providers.ollama import OllamaProvider

        return OllamaProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown provider '{name}'. Available: openai, anthropic, ollama"
        )


def _detect_provider() -> str:
    """Auto-detect provider from environment variables."""
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "ollama"
