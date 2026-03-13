"""Core data structures for capturing LLM and agent evaluation results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A single tool/function call made by an agent."""

    name: str
    args: dict = field(default_factory=dict)
    result: str | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "args": self.args, "result": self.result}


@dataclass
class TrajectoryStep:
    """A single step in a multi-step agent trajectory."""

    role: str  # "user", "assistant", "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }


@dataclass
class LLMResult:
    """Captures everything about a single LLM call or multi-step agent run.

    This is the central data object that flows through the entire eval pipeline.
    Assertions, reports, and compliance certificates all operate on LLMResult.
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    cost: float = 0.0
    latency: float = 0.0
    model: str = ""
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "trajectory": [s.to_dict() for s in self.trajectory],
            "cost": self.cost,
            "latency": self.latency,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "metadata": self.metadata,
        }
