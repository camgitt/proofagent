"""Auto-generate test packs from an agent description or system prompt.

Usage:
    from proofagent.autogen import generate_pack, save_generated_pack

    pack = generate_pack("A customer support chatbot for Acme Corp")
    save_generated_pack(pack, "my_pack.yaml")
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from proofagent.skills import SkillChallenge


_GENERATION_PROMPT = """You are a QA engineer specializing in testing AI agents. Given the description of an AI agent below, generate {num_challenges} test challenges that would thoroughly evaluate whether this agent works correctly.

For each challenge, provide:
- "prompt": What to send to the agent (a realistic user message or scenario)
- "rubric": What a good response looks like (specific, measurable criteria for scoring)

Cover a diverse range of:
- Core functionality (the main things the agent should do well)
- Edge cases (unusual inputs, boundary conditions)
- Failure modes (things the agent should refuse or handle gracefully)
- Tone and professionalism (appropriate communication style)
- Accuracy and compliance (factual correctness, policy adherence)

AGENT DESCRIPTION:
{description}

Output ONLY a JSON array of objects, each with "prompt" and "rubric" string fields. No markdown fencing, no explanation — just the raw JSON array.

Example format:
[
  {{"prompt": "...", "rubric": "..."}},
  {{"prompt": "...", "rubric": "..."}}
]"""


def _detect_provider_for_model(model: str) -> str | None:
    """Guess the provider from a model name."""
    model_lower = model.lower()
    if "gemini" in model_lower:
        return "gemini"
    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        return "openai"
    if "claude" in model_lower:
        return "anthropic"
    return None


def generate_pack(
    description: str,
    num_challenges: int = 10,
    model: str = "gemini-2.5-flash",
    provider: str | None = None,
) -> dict:
    """Auto-generate a skill pack from an agent description.

    Args:
        description: What the agent does, e.g. "A customer support chatbot for a SaaS product"
                     or paste the agent's system prompt directly.
        num_challenges: Number of challenges to generate.
        model: Model to use for generation.
        provider: Provider name (auto-detected if None).

    Returns:
        A skill pack dict compatible with SKILL_PACKS format, containing
        'name', 'description', and 'challenges' (list of SkillChallenge).
    """
    from proofagent.providers import get_provider

    if provider is None:
        provider = _detect_provider_for_model(model)

    llm = get_provider(provider)

    prompt = _GENERATION_PROMPT.format(
        num_challenges=num_challenges,
        description=description.strip(),
    )

    result = llm.complete(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.7,
    )

    # Parse the JSON response
    text = result.text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        challenges_raw = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON array in the response
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                challenges_raw = json.loads(match.group())
            except json.JSONDecodeError:
                raise ValueError(
                    f"Could not parse LLM response as JSON. Raw response:\n{result.text[:500]}"
                )
        else:
            raise ValueError(
                f"Could not find JSON array in LLM response. Raw response:\n{result.text[:500]}"
            )

    if not isinstance(challenges_raw, list):
        raise ValueError(f"Expected a JSON array, got {type(challenges_raw).__name__}")

    # Convert to SkillChallenge objects
    challenges = []
    for i, item in enumerate(challenges_raw):
        if not isinstance(item, dict):
            continue
        prompt_text = item.get("prompt", "")
        rubric_text = item.get("rubric", "")
        if prompt_text and rubric_text:
            challenges.append(
                SkillChallenge(prompt=str(prompt_text), rubric=str(rubric_text))
            )

    if not challenges:
        raise ValueError("LLM generated no valid challenges (each needs 'prompt' and 'rubric')")

    # Generate a pack name from the description
    short_desc = description.strip().split("\n")[0][:60]
    pack_name = f"Generated: {short_desc}"

    return {
        "name": pack_name,
        "description": f"Auto-generated pack from: {short_desc}",
        "challenges": challenges,
    }


def save_generated_pack(pack: dict, output_path: str = "generated_pack.yaml") -> str:
    """Save a generated pack as a YAML file.

    Args:
        pack: A skill pack dict (as returned by generate_pack).
        output_path: Where to write the YAML file.

    Returns:
        The absolute path to the saved file.
    """
    path = Path(output_path)

    lines = []
    lines.append(f"# Auto-generated skill pack")
    lines.append(f"#")
    lines.append(f"# Use with: proofagent skill run <model> --pack {output_path}")
    lines.append(f"")
    lines.append(f"name: {pack['name']}")
    lines.append(f"description: {pack.get('description', '')}")
    lines.append(f"")
    lines.append(f"challenges:")

    for challenge in pack["challenges"]:
        prompt = challenge.prompt if isinstance(challenge, SkillChallenge) else challenge.get("prompt", "")
        rubric = challenge.rubric if isinstance(challenge, SkillChallenge) else challenge.get("rubric", "")

        # Escape quotes for YAML
        prompt_escaped = prompt.replace('"', '\\"')
        rubric_escaped = rubric.replace('"', '\\"')

        lines.append(f'  - prompt: "{prompt_escaped}"')
        lines.append(f'    rubric: "{rubric_escaped}"')
        lines.append(f"")

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path.resolve())
