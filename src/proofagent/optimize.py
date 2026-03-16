"""Cost optimization — find the cheapest model that passes your eval suite."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field


# Unified cost table: per 1M tokens (USD) — updated March 2026
MODEL_COSTS = {
    # OpenAI
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    # Anthropic
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    # Google
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 1.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}


@dataclass
class ModelResult:
    """Score and cost for a single model run."""

    model: str
    score: float
    cost: float
    passed: int = 0
    total: int = 0


@dataclass
class OptimizationResult:
    """Recommendation from the cost optimizer."""

    current_model: str
    current_cost: float
    current_score: float
    recommended_model: str
    recommended_cost: float
    recommended_score: float
    savings_percent: float
    all_results: list[ModelResult] = field(default_factory=list)


def _detect_provider_for_model(model: str) -> str | None:
    """Guess the provider name from a model string."""
    if model.startswith("gpt-") or model.startswith("o1"):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "gemini"
    return None


def run_eval(test_path: str, model: str) -> ModelResult:
    """Run the pytest eval suite for a single model and return the result.

    Uses ``proofagent test`` under the hood so existing fixtures work.
    """
    import json
    import os
    import tempfile

    env = os.environ.copy()
    env["PROOFAGENT_MODEL"] = model

    provider = _detect_provider_for_model(model)
    if provider:
        env["PROOFAGENT_PROVIDER"] = provider

    # Run pytest with JSON report via --tb=line to keep output short.
    # We parse pass/fail counts from the exit code + captured output.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_path,
        "--tb=no",
        "-q",
        f"--junitxml={tmp_path}",
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    # Parse junitxml for pass/fail counts
    passed = 0
    failed = 0
    total_cost = 0.0

    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(tmp_path)
        root = tree.getroot()
        for suite in root.iter("testsuite"):
            tests = int(suite.get("tests", 0))
            failures = int(suite.get("failures", 0))
            errors = int(suite.get("errors", 0))
            passed = tests - failures - errors
            failed = failures + errors
    except Exception:
        # Fallback: parse pytest's short output
        for line in result.stdout.splitlines():
            if "passed" in line or "failed" in line:
                import re

                m_passed = re.search(r"(\d+) passed", line)
                m_failed = re.search(r"(\d+) failed", line)
                if m_passed:
                    passed = int(m_passed.group(1))
                if m_failed:
                    failed = int(m_failed.group(1))
    finally:
        import os as _os

        try:
            _os.unlink(tmp_path)
        except OSError:
            pass

    total = passed + failed
    score = passed / total if total > 0 else 0.0

    # Estimate cost from the model cost table (rough: assume ~1K tokens per test)
    costs = MODEL_COSTS.get(model, {"input": 1.00, "output": 5.00})
    estimated_cost_per_test = (500 * costs["input"] + 200 * costs["output"]) / 1_000_000
    total_cost = estimated_cost_per_test * total

    return ModelResult(
        model=model,
        score=score,
        cost=total_cost,
        passed=passed,
        total=total,
    )


def optimize(
    test_path: str,
    models: list[str],
    min_score: float = 1.0,
) -> OptimizationResult:
    """Run evals against each model and recommend the cheapest one that passes.

    Args:
        test_path: Path to the test directory or file.
        models: List of model names to evaluate.
        min_score: Minimum acceptable score (0.0-1.0). Default 1.0 (all tests pass).

    Returns:
        An OptimizationResult with the recommendation.
    """
    all_results: list[ModelResult] = []

    for model in models:
        mr = run_eval(test_path, model)
        all_results.append(mr)

    # Sort by cost ascending for display
    all_results.sort(key=lambda r: r.cost)

    # The "current" model is the first one in the user's list
    current = all_results[0]
    for r in all_results:
        if r.model == models[0]:
            current = r
            break

    # Find the cheapest model that meets the minimum score
    candidates = [r for r in all_results if r.score >= min_score]

    if candidates:
        recommended = min(candidates, key=lambda r: r.cost)
    else:
        # No model meets the threshold — recommend the highest-scoring one
        recommended = max(all_results, key=lambda r: r.score)

    savings = (
        (1 - recommended.cost / current.cost) * 100
        if current.cost > 0
        else 0.0
    )

    return OptimizationResult(
        current_model=current.model,
        current_cost=current.cost,
        current_score=current.score,
        recommended_model=recommended.model,
        recommended_cost=recommended.cost,
        recommended_score=recommended.score,
        savings_percent=max(savings, 0.0),
        all_results=all_results,
    )
