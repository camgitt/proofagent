"""Provider registry — get a provider by name or auto-detect from env."""

from __future__ import annotations

import os

from proofagent.providers.base import Provider


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
        from proofagent.providers.openai import OpenAIProvider

        return OpenAIProvider(**kwargs)
    elif name == "anthropic":
        from proofagent.providers.anthropic import AnthropicProvider

        return AnthropicProvider(**kwargs)
    elif name == "gemini":
        from proofagent.providers.gemini import GeminiProvider

        return GeminiProvider(**kwargs)
    elif name == "ollama":
        from proofagent.providers.ollama import OllamaProvider

        return OllamaProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown provider '{name}'. Available: openai, anthropic, gemini, ollama"
        )


def _detect_provider() -> str:
    """Auto-detect provider from environment variables."""
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return "gemini"
    # Check if ollama is available before defaulting to it
    try:
        import requests
        return "ollama"
    except ImportError:
        raise RuntimeError(
            "No API key found. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY\n"
            "  Or install Ollama for local models: pip install proofagent[ollama]"
        )
