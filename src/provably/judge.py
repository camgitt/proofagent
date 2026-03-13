"""LLM-as-judge for semantic evaluation."""

from __future__ import annotations

import json
import re

from provably.providers import get_provider

_JUDGE_PROMPT = """You are an evaluation judge. Score how well the AI output matches the given description.

Description: {description}

AI Output:
{output}

Respond with ONLY a JSON object: {{"score": <float between 0.0 and 1.0>, "reason": "<brief explanation>"}}

Scoring guide:
- 1.0: Perfect match — output fully satisfies the description
- 0.7-0.9: Good match — mostly satisfies with minor gaps
- 0.4-0.6: Partial match — some relevant content but significant gaps
- 0.1-0.3: Poor match — barely related
- 0.0: No match — completely unrelated or contradictory"""


def evaluate_semantic_match(
    output: str,
    description: str,
    model: str = "openai/gpt-4o-mini",
) -> float:
    """Use an LLM to score semantic similarity between output and description.

    Args:
        output: The AI-generated text to evaluate.
        description: What the output should convey.
        model: Judge model in "provider/model" format.

    Returns:
        Float score between 0.0 and 1.0.
    """
    # Parse provider/model format
    if "/" in model:
        provider_name, model_name = model.split("/", 1)
    else:
        provider_name = None
        model_name = model

    provider = get_provider(provider_name)

    prompt = _JUDGE_PROMPT.format(
        description=description,
        output=output[:2000],  # truncate to avoid excessive cost
    )

    result = provider.complete(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
    )

    # Parse score from response
    try:
        text = result.text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        return float(data.get("score", 0.0))
    except (json.JSONDecodeError, ValueError, TypeError):
        # Try to extract a number
        match = re.search(r'"score"\s*:\s*([\d.]+)', result.text)
        if match:
            return float(match.group(1))
        return 0.0
