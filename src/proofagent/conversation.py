"""Multi-turn conversation testing for LLM agent evaluations.

Usage:
    from proofagent import Conversation, ConversationTurn, LLMResult

    conv = Conversation(turns=[
        ("What's 2+2?", LLMResult(text="4")),
        ("And times 3?", LLMResult(text="12")),
    ])
    expect(conv).turn_count(2).all_turns_cost_under(0.10)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from proofagent.result import LLMResult


@dataclass
class ConversationTurn:
    """A single turn in a multi-turn conversation."""

    user_message: str
    result: LLMResult


@dataclass
class Conversation:
    """A multi-turn conversation between a user and an LLM.

    Accepts turns as a list of (user_message, LLMResult) tuples for
    convenience, automatically converting them to ConversationTurn objects.
    """

    turns: list[ConversationTurn | tuple[str, LLMResult]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        self.turns = [
            ConversationTurn(user_message=t[0], result=t[1])
            if isinstance(t, tuple)
            else t
            for t in self.turns
        ]

    def turn(self, n: int) -> ConversationTurn:
        """Return the nth turn (supports negative indexing)."""
        return self.turns[n]

    @property
    def total_cost(self) -> float:
        """Sum of cost across all turns."""
        return sum(t.result.cost for t in self.turns)

    @property
    def total_latency(self) -> float:
        """Sum of latency across all turns."""
        return sum(t.result.latency for t in self.turns)
