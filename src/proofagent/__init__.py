"""proofagent — pytest for AI agents.

Evaluate, test, and certify AI agents with cryptographic compliance proofs.

Usage:
    from proofagent import expect, LLMResult

    result = LLMResult(text="Hello!")
    expect(result).contains("Hello")
"""

from proofagent.__version__ import __version__
from proofagent.compare import CompareResult, compare, compare_batch
from proofagent.conversation import Conversation, ConversationTurn
from proofagent.datasets import Dataset, load_dataset
from proofagent.expect import Expectation, expect, register_assertion
from proofagent.result import LLMResult, ToolCall, TrajectoryStep

__all__ = [
    "__version__",
    "compare",
    "compare_batch",
    "CompareResult",
    "Conversation",
    "ConversationTurn",
    "Dataset",
    "expect",
    "Expectation",
    "load_dataset",
    "register_assertion",
    "LLMResult",
    "ToolCall",
    "TrajectoryStep",
]
