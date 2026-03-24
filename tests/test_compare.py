"""Tests for model comparison (A vs B testing)."""

import importlib
import sys
from unittest.mock import MagicMock, patch

# proofagent.__init__ re-exports the function "compare", which shadows
# the module name when using @patch.object(_compare_mod, "get_provider").
# We grab the actual module from sys.modules instead.
import proofagent.compare  # noqa: F401 — ensure module is loaded
_compare_mod = sys.modules["proofagent.compare"]
compare_fn = _compare_mod.compare

from proofagent import CompareResult, compare_batch
from proofagent.result import LLMResult



# ── CompareResult dataclass ─────────────────────────────────────────


def test_compare_result_basic():
    r = CompareResult(
        model_a="model-a",
        model_b="model-b",
        result_a=LLMResult(text="Hello from A"),
        result_b=LLMResult(text="Hello from B"),
    )
    assert r.model_a == "model-a"
    assert r.model_b == "model-b"
    assert r.result_a.text == "Hello from A"
    assert r.result_b.text == "Hello from B"
    assert r.winner is None


def test_compare_result_with_winner():
    r = CompareResult(
        model_a="fast",
        model_b="slow",
        result_a=LLMResult(text="A"),
        result_b=LLMResult(text="B"),
        winner="fast",
    )
    assert r.winner == "fast"


def test_compare_result_to_dict():
    r = CompareResult(
        model_a="m-a",
        model_b="m-b",
        result_a=LLMResult(text="A", cost=0.01),
        result_b=LLMResult(text="B", cost=0.02),
        winner="m-a",
        assertion_results={"a": {"check": True}, "b": {"check": False}},
    )
    d = r.to_dict()
    assert d["model_a"] == "m-a"
    assert d["model_b"] == "m-b"
    assert d["result_a"]["text"] == "A"
    assert d["result_b"]["cost"] == 0.02
    assert d["winner"] == "m-a"
    assert d["assertion_results"]["a"]["check"] is True


def test_compare_result_default_assertion_results():
    r = CompareResult(
        model_a="a",
        model_b="b",
        result_a=LLMResult(text=""),
        result_b=LLMResult(text=""),
    )
    assert r.assertion_results == {}


# ── compare() function ──────────────────────────────────────────────


def _make_mock_provider(results_by_model):
    """Create a mock provider that returns pre-set results by model name."""
    provider = MagicMock()
    provider.name.return_value = "mock"

    def _complete(messages, model=None, **kwargs):
        return results_by_model[model]

    provider.complete.side_effect = _complete
    return provider


@patch.object(_compare_mod, "get_provider")
def test_compare_runs_both_models(mock_get_provider):
    result_a = LLMResult(text="Response A", model="alpha")
    result_b = LLMResult(text="Response B", model="beta")
    mock_get_provider.return_value = _make_mock_provider(
        {"alpha": result_a, "beta": result_b}
    )

    cr = compare_fn("Hello", model_a="alpha", model_b="beta")

    assert cr.model_a == "alpha"
    assert cr.model_b == "beta"
    assert cr.result_a.text == "Response A"
    assert cr.result_b.text == "Response B"
    assert cr.winner is None


@patch.object(_compare_mod, "get_provider")
def test_compare_with_assertions_a_wins(mock_get_provider):
    result_a = LLMResult(text="Hello world", model="alpha")
    result_b = LLMResult(text="Goodbye", model="beta")
    mock_get_provider.return_value = _make_mock_provider(
        {"alpha": result_a, "beta": result_b}
    )

    def contains_hello(r):
        return "Hello" in r.text

    cr = compare_fn("Say hi", model_a="alpha", model_b="beta", assertions=[contains_hello])

    assert cr.winner == "alpha"
    assert cr.assertion_results["a"]["contains_hello"] is True
    assert cr.assertion_results["b"]["contains_hello"] is False


@patch.object(_compare_mod, "get_provider")
def test_compare_with_assertions_b_wins(mock_get_provider):
    result_a = LLMResult(text="short", model="alpha")
    result_b = LLMResult(text="a much longer response here", model="beta")
    mock_get_provider.return_value = _make_mock_provider(
        {"alpha": result_a, "beta": result_b}
    )

    def is_long(r):
        return len(r.text) > 10

    cr = compare_fn("Write something", model_a="alpha", model_b="beta", assertions=[is_long])

    assert cr.winner == "beta"


@patch.object(_compare_mod, "get_provider")
def test_compare_with_assertions_tie(mock_get_provider):
    result_a = LLMResult(text="Hello world", model="alpha")
    result_b = LLMResult(text="Hello earth", model="beta")
    mock_get_provider.return_value = _make_mock_provider(
        {"alpha": result_a, "beta": result_b}
    )

    def contains_hello(r):
        return "Hello" in r.text

    cr = compare_fn("Say hi", model_a="alpha", model_b="beta", assertions=[contains_hello])

    assert cr.winner is None  # tie


@patch.object(_compare_mod, "get_provider")
def test_compare_assertion_exception_counts_as_fail(mock_get_provider):
    result_a = LLMResult(text="ok", model="alpha")
    result_b = LLMResult(text="ok", model="beta")
    mock_get_provider.return_value = _make_mock_provider(
        {"alpha": result_a, "beta": result_b}
    )

    call_count = 0

    def flaky(r):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("boom")
        return True

    cr = compare_fn("test", model_a="alpha", model_b="beta", assertions=[flaky])

    # First call (model a) raised, second (model b) passed
    assert cr.winner == "beta"
    assert cr.assertion_results["a"]["flaky"] is False
    assert cr.assertion_results["b"]["flaky"] is True


@patch.object(_compare_mod, "get_provider")
def test_compare_passes_provider_name(mock_get_provider):
    result = LLMResult(text="x")
    mock_get_provider.return_value = _make_mock_provider(
        {"a": result, "b": result}
    )

    compare_fn("hi", model_a="a", model_b="b", provider="openai")

    mock_get_provider.assert_any_call(name="openai")


# ── compare_batch() ─────────────────────────────────────────────────


@patch.object(_compare_mod, "get_provider")
def test_compare_batch(mock_get_provider):
    result_a = LLMResult(text="A", model="alpha")
    result_b = LLMResult(text="B", model="beta")
    mock_get_provider.return_value = _make_mock_provider(
        {"alpha": result_a, "beta": result_b}
    )

    results = compare_batch(
        prompts=["Hello", "World", "Test"],
        model_a="alpha",
        model_b="beta",
    )

    assert len(results) == 3
    assert all(isinstance(r, CompareResult) for r in results)
    assert all(r.model_a == "alpha" for r in results)


@patch.object(_compare_mod, "get_provider")
def test_compare_batch_empty(mock_get_provider):
    results = compare_batch(prompts=[], model_a="a", model_b="b")
    assert results == []
