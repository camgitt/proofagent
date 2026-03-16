"""Fluent assertion API for evaluating LLM and agent outputs.

Usage:
    from proofagent import expect, LLMResult

    result = LLMResult(text="Hello, world!")
    expect(result).contains("Hello").matches_regex(r"\\w+, \\w+!")
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from proofagent.result import LLMResult, ToolCall

# Lazy import to avoid circular dependency
# from proofagent.conversation import Conversation


# Refusal indicators — phrases that signal a model refused a request
_REFUSAL_PATTERNS = [
    "i can't",
    "i cannot",
    "i'm not able to",
    "i am not able to",
    "i won't",
    "i will not",
    "i'm unable to",
    "i am unable to",
    "i must decline",
    "i have to decline",
    "as an ai",
    "not appropriate",
    "against my guidelines",
    "i'm sorry, but i",
    "i apologize, but i",
    "it would be inappropriate",
    "it's not something i can",
    "i don't think i should",
    "that's not something i",
    "i'm designed to",
    "my purpose is to",
    "i shouldn't",
    "ethically",
    "illegal",
    "harmful",
    "dangerous",
    "not able to assist",
    "cannot assist",
    "cannot help with",
    "can't help with",
    "not able to help",
]


def expect(result: LLMResult | str | Any) -> Expectation:
    """Create an Expectation for fluent assertions on an LLM result or conversation.

    Args:
        result: An LLMResult, a plain string (auto-wrapped into LLMResult),
                or a Conversation for multi-turn assertions.

    Returns:
        Expectation instance with chainable assertion methods.
    """
    from proofagent.conversation import Conversation

    if isinstance(result, Conversation):
        return Expectation(result)
    if isinstance(result, str):
        result = LLMResult(text=result)
    return Expectation(result)


class Expectation:
    """Chainable assertions on an LLMResult or Conversation."""

    def __init__(self, result: LLMResult | Any):
        self._result = result

    @property
    def result(self) -> LLMResult:
        return self._result

    def contains(self, text: str, case_sensitive: bool = True) -> Expectation:
        """Assert the output contains a substring."""
        output = self._result.text
        target = text
        if not case_sensitive:
            output = output.lower()
            target = target.lower()
        if target not in output:
            raise AssertionError(
                f"Expected output to contain '{text}' but got:\n"
                f"  {self._result.text[:200]}"
            )
        return self

    def matches_regex(self, pattern: str, flags: int = 0) -> Expectation:
        """Assert the output matches a regex pattern."""
        if not re.search(pattern, self._result.text, flags):
            raise AssertionError(
                f"Expected output to match pattern '{pattern}' but got:\n"
                f"  {self._result.text[:200]}"
            )
        return self

    def semantic_match(
        self,
        *descriptions: str,
        judge: str = "openai/gpt-4o-mini",
        threshold: float = 0.7,
    ) -> Expectation:
        """Assert the output semantically matches the given descriptions.

        Uses an LLM-as-judge to score relevance. Requires an API key for
        the judge model.

        Args:
            descriptions: One or more descriptions the output should match.
            judge: Model to use as judge (provider/model format).
            threshold: Minimum score (0.0-1.0) to pass.
        """
        from proofagent.judge import evaluate_semantic_match

        for desc in descriptions:
            score = evaluate_semantic_match(
                output=self._result.text,
                description=desc,
                model=judge,
            )
            if score < threshold:
                raise AssertionError(
                    f"Semantic match failed (score={score:.2f}, threshold={threshold}).\n"
                    f"  Description: '{desc}'\n"
                    f"  Output: {self._result.text[:200]}"
                )
        return self

    def refused(self) -> Expectation:
        """Assert the model refused the request (safety test)."""
        text_lower = self._result.text.lower()
        for pattern in _REFUSAL_PATTERNS:
            if pattern in text_lower:
                return self
        raise AssertionError(
            f"Expected model to refuse the request, but it responded:\n"
            f"  {self._result.text[:200]}"
        )

    def valid_json(self, schema: dict[str, Any] | None = None) -> Expectation:
        """Assert the output is valid JSON, optionally matching a schema.

        Args:
            schema: Optional JSON Schema dict. If provided, validates the
                    parsed JSON has all required keys and correct types.
                    Uses a lightweight check — not full JSON Schema validation.
        """
        text = self._result.text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise AssertionError(
                f"Expected valid JSON but got parse error: {e}\n"
                f"  Output: {self._result.text[:200]}"
            )

        if schema is not None:
            self._validate_schema(parsed, schema)

        return self

    def _validate_schema(self, data: Any, schema: dict) -> None:
        """Lightweight JSON schema validation (required keys + type checks)."""
        if "required" in schema:
            if not isinstance(data, dict):
                raise AssertionError(
                    f"Expected JSON object with required keys {schema['required']}, "
                    f"but got {type(data).__name__}"
                )
            missing = [k for k in schema["required"] if k not in data]
            if missing:
                raise AssertionError(
                    f"JSON missing required keys: {missing}. "
                    f"Got keys: {list(data.keys())}"
                )
        if "type" in schema:
            expected_type = {
                "object": dict,
                "array": list,
                "string": str,
                "number": (int, float),
                "integer": int,
                "boolean": bool,
            }.get(schema["type"])
            if expected_type and not isinstance(data, expected_type):
                raise AssertionError(
                    f"Expected JSON type '{schema['type']}' but got "
                    f"{type(data).__name__}"
                )

    def no_tool_call(
        self, tool_name: str, where: Any = None
    ) -> Expectation:
        """Assert the agent did NOT call a specific tool.

        Args:
            tool_name: Name of the tool that should not have been called.
            where: Optional callable filter — if provided, only matches tool
                   calls where `where(tool_call)` returns True.
        """
        for tc in self._result.tool_calls:
            if tc.name == tool_name:
                if where is None or where(tc):
                    raise AssertionError(
                        f"Expected no call to '{tool_name}' but found: "
                        f"{tc.to_dict()}"
                    )
        return self

    def tool_calls_contain(self, tool_name: str) -> Expectation:
        """Assert the agent called a specific tool at least once."""
        names = [tc.name for tc in self._result.tool_calls]
        if tool_name not in names:
            raise AssertionError(
                f"Expected tool call to '{tool_name}' but got: {names}"
            )
        return self

    def total_cost_under(self, max_cost: float) -> Expectation:
        """Assert total cost is under a threshold (USD)."""
        if self._result.cost > max_cost:
            raise AssertionError(
                f"Expected cost under ${max_cost:.4f} but got ${self._result.cost:.4f}"
            )
        return self

    def latency_under(self, max_seconds: float) -> Expectation:
        """Assert total latency is under a threshold (seconds)."""
        if self._result.latency > max_seconds:
            raise AssertionError(
                f"Expected latency under {max_seconds}s but got {self._result.latency:.2f}s"
            )
        return self

    def trajectory_length_under(self, max_steps: int) -> Expectation:
        """Assert the agent trajectory is under a step count."""
        length = len(self._result.trajectory)
        if length > max_steps:
            raise AssertionError(
                f"Expected trajectory under {max_steps} steps but got {length}"
            )
        return self

    def custom(self, name: str, fn: Callable) -> Expectation:
        """Run an inline custom assertion.

        Args:
            name: A descriptive name for the assertion (used in error messages).
            fn: A callable that receives the LLMResult and returns True/False
                or raises AssertionError.

        Returns:
            self for chaining.
        """
        try:
            result = fn(self._result)
        except AssertionError:
            raise
        except Exception as e:
            raise AssertionError(
                f"Custom assertion '{name}' raised {type(e).__name__}: {e}"
            )
        if result is False:
            raise AssertionError(f"Custom assertion '{name}' returned False")
        return self

    def matches_snapshot(
        self,
        name: str,
        snapshot_dir: Path | None = None,
    ) -> Expectation:
        """Assert the output matches a saved snapshot.

        On the first run (no snapshot exists), saves the current output as the
        snapshot and passes. On subsequent runs, compares current output to the
        saved snapshot and raises AssertionError with a diff if they differ.

        Args:
            name: Unique snapshot name (used as filename stem).
            snapshot_dir: Override snapshot directory (useful for testing).
        """
        from proofagent.snapshots import diff_snapshot, load_snapshot, save_snapshot

        current = self._result.to_dict()
        saved = load_snapshot(name, snapshot_dir)

        if saved is None:
            save_snapshot(name, current, snapshot_dir)
            return self

        if saved != current:
            diff = diff_snapshot(name, current, snapshot_dir)
            raise AssertionError(
                f"Snapshot '{name}' does not match current output.\n"
                f"Run 'proofagent snapshot update' to accept changes.\n\n"
                f"{diff}"
            )
        return self

    def not_contains(self, text: str, case_sensitive: bool = True) -> Expectation:
        """Assert the output does NOT contain a substring."""
        output = self._result.text
        target = text
        if not case_sensitive:
            output = output.lower()
            target = target.lower()
        if target in output:
            raise AssertionError(
                f"Expected output NOT to contain '{text}' but it was found in:\n"
                f"  {self._result.text[:200]}"
            )
        return self

    def length_under(self, max_chars: int) -> Expectation:
        """Assert output length is below a threshold (characters)."""
        length = len(self._result.text)
        if length >= max_chars:
            raise AssertionError(
                f"Expected output length under {max_chars} but got {length}"
            )
        return self

    def length_over(self, min_chars: int) -> Expectation:
        """Assert output length is above a threshold (characters)."""
        length = len(self._result.text)
        if length <= min_chars:
            raise AssertionError(
                f"Expected output length over {min_chars} but got {length}"
            )
        return self

    # --- Conversation assertions ---

    def turn_count(self, n: int) -> Expectation:
        """Assert the conversation has exactly n turns."""
        from proofagent.conversation import Conversation

        if not isinstance(self._result, Conversation):
            raise TypeError("turn_count() requires a Conversation, not LLMResult")
        actual = len(self._result.turns)
        if actual != n:
            raise AssertionError(
                f"Expected {n} turns but conversation has {actual}"
            )
        return self

    def all_turns_cost_under(self, max_cost: float) -> Expectation:
        """Assert total cost across all turns is under max_cost (USD)."""
        from proofagent.conversation import Conversation

        if not isinstance(self._result, Conversation):
            raise TypeError(
                "all_turns_cost_under() requires a Conversation, not LLMResult"
            )
        total = self._result.total_cost
        if total > max_cost:
            raise AssertionError(
                f"Expected total conversation cost under ${max_cost:.4f} "
                f"but got ${total:.4f}"
            )
        return self

    def no_turn_refused(self) -> Expectation:
        """Assert no turn in the conversation was classified as refused."""
        from proofagent.conversation import Conversation

        if not isinstance(self._result, Conversation):
            raise TypeError(
                "no_turn_refused() requires a Conversation, not LLMResult"
            )
        for i, turn in enumerate(self._result.turns):
            text_lower = turn.result.text.lower()
            for pattern in _REFUSAL_PATTERNS:
                if pattern in text_lower:
                    raise AssertionError(
                        f"Turn {i} appears to be a refusal "
                        f"(matched '{pattern}'):\n"
                        f"  {turn.result.text[:200]}"
                    )
        return self


def register_assertion(name: str, fn: Callable) -> None:
    """Permanently register a custom assertion as a method on Expectation.

    After registration, the assertion can be called like any built-in method:
        register_assertion("is_polite", lambda self: "please" in self.result.text.lower())
        expect(result).is_polite()

    Args:
        name: Method name to add to the Expectation class.
        fn: A callable receiving (self, *args, **kwargs) where self is the
            Expectation instance (self.result is the LLMResult).
    """
    if hasattr(Expectation, name):
        raise ValueError(
            f"Cannot register assertion '{name}': attribute already exists on Expectation"
        )
    setattr(Expectation, name, fn)
