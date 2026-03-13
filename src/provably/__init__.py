"""Provably — pytest for AI agents.

Evaluate, test, and certify AI agents with cryptographic compliance proofs.

Usage:
    from provably import expect, LLMResult

    result = LLMResult(text="Hello!")
    expect(result).contains("Hello")
"""

from provably.__version__ import __version__
from provably.expect import Expectation, expect
from provably.result import LLMResult, ToolCall, TrajectoryStep

__all__ = [
    "__version__",
    "expect",
    "Expectation",
    "LLMResult",
    "ToolCall",
    "TrajectoryStep",
]
