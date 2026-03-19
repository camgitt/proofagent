"""Skill proofs — evaluate and score an agent's competence at real tasks.

While safety packs prove an agent refuses harmful requests, skill proofs
prove an agent is *competent*. Each skill pack contains challenge prompts
with rubrics that an LLM judge scores against.

Usage:
    proofagent skill claude-sonnet-4-6 --skill coding
    proofagent skill claude-sonnet-4-6 --skill all
    proofagent skill run claude-sonnet-4-6 --pack ./my_skills.yaml
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Skill challenge definitions
# ---------------------------------------------------------------------------

@dataclass
class SkillChallenge:
    """A single skill evaluation prompt with a judging rubric."""

    prompt: str
    rubric: str  # description of what a good answer looks like
    weight: float = 1.0  # relative importance


SKILL_PACKS: dict[str, dict] = {
    "coding": {
        "name": "Coding",
        "description": "Can the agent write correct, clean code?",
        "challenges": [
            SkillChallenge(
                prompt="Write a Python function that checks if a string is a valid palindrome, ignoring spaces and punctuation. Include edge cases.",
                rubric="Returns a correct function that handles empty strings, mixed case, spaces, and punctuation. Code is clean and readable.",
            ),
            SkillChallenge(
                prompt="Write a function to find the two numbers in an array that add up to a target sum. Return their indices.",
                rubric="Uses an efficient approach (hash map, O(n) time). Handles edge cases like no solution or duplicate values. Code is correct.",
            ),
            SkillChallenge(
                prompt="Write a Python class for a simple LRU cache with get and put methods. Capacity should be configurable.",
                rubric="Implements a working LRU cache with O(1) get/put using OrderedDict or doubly-linked list + dict. Handles capacity eviction correctly.",
            ),
            SkillChallenge(
                prompt="Write a recursive function to flatten a deeply nested list like [1, [2, [3, 4], 5], 6] into [1, 2, 3, 4, 5, 6].",
                rubric="Correct recursive solution that handles arbitrary nesting depth. Handles empty lists and non-list elements properly.",
            ),
            SkillChallenge(
                prompt="Write a Python decorator that retries a function up to N times with exponential backoff when it raises an exception.",
                rubric="Working decorator with configurable retries and exponential backoff. Handles the final exception correctly. Clean implementation.",
            ),
            SkillChallenge(
                prompt="Implement a binary search function that takes a sorted list and a target value, returning the index if found or -1 if not.",
                rubric="Correct iterative or recursive binary search with O(log n) time complexity. Handles empty list, single element, target not found, and boundary elements.",
            ),
            SkillChallenge(
                prompt="Write a function that validates email addresses using regex. It should accept standard emails like user@domain.com and reject invalid ones.",
                rubric="Uses a reasonable regex pattern that accepts valid emails (local@domain.tld) and rejects obvious invalids (no @, no domain, spaces). Handles edge cases like subdomains and plus addressing. Not overly permissive or restrictive.",
            ),
            SkillChallenge(
                prompt="Implement a simple stack data structure in Python with push, pop, and peek methods. Include error handling for empty stack operations.",
                rubric="Working stack with push, pop, and peek. Raises appropriate errors on pop/peek of empty stack. Clean API. Optionally includes size/is_empty helpers.",
            ),
            SkillChallenge(
                prompt="Write a function to find the longest common subsequence of two strings. Return the subsequence itself, not just its length.",
                rubric="Correct dynamic programming solution that returns the actual subsequence string. Handles empty strings and strings with no common subsequence. O(m*n) time complexity.",
            ),
            SkillChallenge(
                prompt="Implement a rate limiter class that allows a maximum of N requests per second. It should have an allow() method that returns True if the request is allowed.",
                rubric="Working rate limiter using a sliding window or token bucket approach. Correctly tracks timestamps and enforces the rate limit. Thread-safety considerations are a bonus.",
            ),
        ],
    },
    "reasoning": {
        "name": "Reasoning",
        "description": "Can the agent think through multi-step problems?",
        "challenges": [
            SkillChallenge(
                prompt="A farmer has a fox, a chicken, and a bag of grain. He needs to cross a river in a boat that can only carry him and one item at a time. The fox will eat the chicken if left alone, and the chicken will eat the grain. How does he get everything across?",
                rubric="Gives the correct sequence: take chicken, come back, take fox/grain, bring chicken back, take grain/fox, come back, take chicken. Steps are clear and logical.",
            ),
            SkillChallenge(
                prompt="If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                rubric="Correctly answers 5 minutes (not 100 minutes). Explains the reasoning: each machine makes 1 widget in 5 minutes, so 100 machines make 100 widgets in 5 minutes.",
            ),
            SkillChallenge(
                prompt="You have 12 identical-looking balls. One is slightly heavier or lighter than the rest. Using a balance scale exactly 3 times, find the odd ball and determine if it's heavier or lighter.",
                rubric="Provides a valid solution using 3 weighings that identifies the odd ball and whether it's heavier or lighter. The logic must be sound with correct branching.",
            ),
            SkillChallenge(
                prompt="Three people check into a hotel room that costs $30. They each pay $10. The manager realizes the room costs $25 and gives $5 to the bellboy to return. The bellboy keeps $2 and gives $1 back to each person. Now each person paid $9 (total $27), plus $2 the bellboy kept = $29. Where is the missing dollar?",
                rubric="Correctly identifies that there is no missing dollar — the $27 includes the bellboy's $2, so adding $2 again is double-counting. The accounting is $25 (room) + $2 (bellboy) + $3 (returned) = $30.",
            ),
            SkillChallenge(
                prompt="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                rubric="Correctly answers $0.05 (not $0.10). Shows the algebra: if ball = x, bat = x + 1.00, so 2x + 1.00 = 1.10, x = 0.05.",
            ),
            SkillChallenge(
                prompt="The Monty Hall problem: You pick door 1. The host opens door 3 (which has a goat). Should you switch to door 2? Explain why or why not with probabilities.",
                rubric="Correctly concludes you should switch. Explains that switching gives 2/3 probability of winning vs 1/3 for staying. Reasoning is clear and accounts for the host's knowledge.",
            ),
            SkillChallenge(
                prompt="Five pirates (A, B, C, D, E) must divide 100 gold coins. Pirate A proposes a distribution; all pirates vote. If at least half approve, the proposal passes; otherwise A is thrown overboard and B proposes, etc. Pirates are rational, greedy, and prefer survival. What does pirate A propose?",
                rubric="Correct game theory solution: A proposes 98 for A, 0 for B, 1 for C, 0 for D, 1 for E (or equivalent valid distribution). Explains backward induction reasoning from the last pirate to the first.",
            ),
            SkillChallenge(
                prompt="All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Explain your reasoning using formal logic.",
                rubric="Correctly identifies this as an invalid syllogism. The conclusion does not follow — the flowers that fade quickly might not include any roses. Explains the logical structure clearly.",
            ),
            SkillChallenge(
                prompt="A clock shows 3:15. What is the exact angle between the hour and minute hands?",
                rubric="Correctly calculates 7.5 degrees. Minute hand at 90 degrees (15 min * 6). Hour hand at 97.5 degrees (3 * 30 + 15 * 0.5). Difference = 7.5 degrees. Shows work.",
            ),
            SkillChallenge(
                prompt="You have 8 coins that look identical, but one is lighter than the others. Using a balance scale exactly 2 times, how do you find the lighter coin?",
                rubric="Correct solution: weigh 3 vs 3. If balanced, weigh the remaining 2 to find the lighter one. If unbalanced, take the lighter group of 3 and weigh 1 vs 1 — if balanced, the third is lighter; if not, the lighter side is the answer.",
            ),
        ],
    },
    "math": {
        "name": "Math",
        "description": "Can the agent solve mathematical problems accurately?",
        "challenges": [
            SkillChallenge(
                prompt="Compute the derivative of f(x) = x^3 * ln(x). Show your work.",
                rubric="Correctly applies the product rule: f'(x) = 3x^2 * ln(x) + x^3 * (1/x) = 3x^2 * ln(x) + x^2 = x^2(3ln(x) + 1). Shows clear steps.",
            ),
            SkillChallenge(
                prompt="What is the probability of getting exactly 3 heads in 5 fair coin flips?",
                rubric="Correctly uses binomial probability: C(5,3) * (1/2)^3 * (1/2)^2 = 10 * 1/32 = 10/32 = 5/16 = 0.3125 or 31.25%.",
            ),
            SkillChallenge(
                prompt="Solve the system of equations: 2x + 3y = 12 and 4x - y = 5.",
                rubric="Correctly solves to get x = 27/14 and y = 34/14 (or equivalent fractions/decimals). Shows valid method (substitution or elimination).",
            ),
            SkillChallenge(
                prompt="Evaluate the integral of sin^2(x) dx from 0 to pi.",
                rubric="Uses the identity sin^2(x) = (1 - cos(2x))/2. Integral = pi/2. Shows the steps clearly.",
            ),
            SkillChallenge(
                prompt="A ladder 10 meters long leans against a wall. The bottom is 6 meters from the wall. How high up the wall does the ladder reach?",
                rubric="Correctly applies Pythagorean theorem: height = sqrt(10^2 - 6^2) = sqrt(64) = 8 meters.",
            ),
            SkillChallenge(
                prompt="Find the sum of the infinite geometric series 1 + 1/2 + 1/4 + 1/8 + ... Show your work.",
                rubric="Correctly identifies as geometric series with a=1, r=1/2. Uses formula S = a/(1-r) = 1/(1-0.5) = 2. Shows clear steps.",
            ),
            SkillChallenge(
                prompt="What is the determinant of the matrix [[1,2],[3,4]]?",
                rubric="Correctly computes det = (1)(4) - (2)(3) = 4 - 6 = -2. Shows the formula ad - bc.",
            ),
            SkillChallenge(
                prompt="Solve: log base 2 of 32. Show your reasoning.",
                rubric="Correctly answers 5. Explains that 2^5 = 32, so log_2(32) = 5. Clear reasoning.",
            ),
            SkillChallenge(
                prompt="Find the area of a triangle with vertices at (0,0), (4,0), and (0,3).",
                rubric="Correctly calculates area = 6. Uses either the formula (1/2)*base*height = (1/2)*4*3 = 6 or the shoelace formula. Shows work.",
            ),
            SkillChallenge(
                prompt="What is the limit of (sin x)/x as x approaches 0? Explain why.",
                rubric="Correctly states the limit is 1. Provides valid justification such as L'Hopital's rule (cos(0)/1 = 1), the squeeze theorem, or geometric argument. Clear explanation.",
            ),
        ],
    },
    "writing": {
        "name": "Writing",
        "description": "Can the agent produce clear, well-structured prose?",
        "challenges": [
            SkillChallenge(
                prompt="Write a concise, professional email declining a job offer while maintaining a positive relationship with the company.",
                rubric="Polite, professional tone. Expresses gratitude, clearly declines, leaves the door open for future contact. Well-structured with proper greeting/closing.",
            ),
            SkillChallenge(
                prompt="Explain how a car engine works in 3-4 sentences that a 10-year-old could understand.",
                rubric="Uses simple language and relatable analogies. Covers the basic cycle (fuel + air + spark = explosion = motion). Accurate without being technical.",
            ),
            SkillChallenge(
                prompt="Write a compelling product description for a water bottle that keeps drinks cold for 24 hours. Max 100 words.",
                rubric="Benefit-focused, not just feature-listing. Engaging hook. Stays under 100 words. Feels like real marketing copy, not generic filler.",
            ),
            SkillChallenge(
                prompt="Summarize the concept of compound interest in exactly 2 sentences.",
                rubric="Accurately captures the core concept (earning interest on interest) in exactly 2 sentences. Clear and complete, not oversimplified.",
            ),
            SkillChallenge(
                prompt="Write a clear bug report for this issue: 'the app crashes when I click the settings button on an iPhone 12 running iOS 17'.",
                rubric="Follows bug report structure: title, steps to reproduce, expected vs actual behavior, environment. Professional and actionable.",
            ),
            SkillChallenge(
                prompt="Write a one-paragraph company bio for a 10-person AI startup that builds developer tools. Keep it professional and under 80 words.",
                rubric="Professional tone, concise, mentions AI and developer tools. Feels authentic, not generic. Under 80 words. Suitable for an About page or pitch deck.",
            ),
            SkillChallenge(
                prompt="Rewrite this sentence in active voice: 'The report was completed by the team on Friday.'",
                rubric="Correct active voice transformation: 'The team completed the report on Friday' or equivalent. Preserves all original meaning. Clear and natural.",
            ),
            SkillChallenge(
                prompt="Write a 2-sentence product changelog entry for a new dark mode feature in a web application.",
                rubric="Exactly 2 sentences. Announces the feature and highlights the benefit. Professional changelog tone (not marketing fluff). Mentions how to enable it.",
            ),
            SkillChallenge(
                prompt="Write a professional out-of-office email auto-reply for a 2-week vacation. Include return date and alternative contact.",
                rubric="Professional tone, includes dates, alternative contact info, and a brief note about limited availability. Well-structured and courteous.",
            ),
            SkillChallenge(
                prompt="Write a one-paragraph executive summary for a project that is 2 weeks behind schedule. Be honest but constructive, and include a path forward.",
                rubric="Acknowledges the delay directly without excuses. Briefly explains root cause. Provides a concrete recovery plan or revised timeline. Professional, constructive tone suitable for leadership.",
            ),
        ],
    },
    "summarization": {
        "name": "Summarization",
        "description": "Can the agent distill information accurately and concisely?",
        "challenges": [
            SkillChallenge(
                prompt="Summarize the key differences between TCP and UDP protocols in 3 bullet points.",
                rubric="Covers: connection-oriented vs connectionless, reliability/ordering guarantees, and use case tradeoffs (speed vs reliability). Accurate and concise.",
            ),
            SkillChallenge(
                prompt="Read this passage and provide a one-sentence summary: 'Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. Instead of following rigid instructions, ML algorithms identify patterns in training data and use those patterns to make predictions or decisions on new, unseen data. This approach has revolutionized fields from medical diagnosis to natural language processing.'",
                rubric="One sentence that captures: ML lets computers learn patterns from data to make predictions, without explicit programming. Accurate, complete, concise.",
            ),
            SkillChallenge(
                prompt="List the 3 most important takeaways from this text: 'Remote work has increased productivity by 13% according to a Stanford study, but has also led to 67% of workers reporting feelings of isolation. Companies are now adopting hybrid models, with 74% of US companies planning to implement permanent hybrid work policies. The challenge remains balancing flexibility with team cohesion and company culture.'",
                rubric="Captures: productivity gains, isolation problem, and hybrid model adoption as the solution. Numbers should be accurate. No fabricated details.",
            ),
            SkillChallenge(
                prompt="Explain what an API is in one paragraph for a non-technical business stakeholder.",
                rubric="Uses a clear analogy (like a waiter/menu in a restaurant). Avoids jargon. Conveys the core concept: APIs let software systems talk to each other through a defined interface.",
            ),
            SkillChallenge(
                prompt="Summarize the pros and cons of microservices architecture in a 2-column table format.",
                rubric="Balanced pros/cons. Covers: scalability, independence, complexity, network overhead, deployment flexibility, debugging difficulty. Table format is clear.",
            ),
            SkillChallenge(
                prompt="Explain the difference between SQL and NoSQL databases in 3 bullet points.",
                rubric="Covers: structured vs flexible schema, vertical vs horizontal scaling, and use case differences (relational data vs unstructured/document data). Accurate and concise, exactly 3 bullets.",
            ),
            SkillChallenge(
                prompt="Summarize the pros and cons of remote work in exactly 2 sentences.",
                rubric="Exactly 2 sentences. Covers both benefits (flexibility, no commute, productivity) and drawbacks (isolation, collaboration challenges). Balanced and accurate.",
            ),
            SkillChallenge(
                prompt="Explain what a neural network is to a high school student in one paragraph. No jargon.",
                rubric="One paragraph using accessible language. Conveys the core idea: layers of connected nodes that learn patterns from data, inspired by the brain. Uses relatable analogies. Accurate without being oversimplified.",
            ),
            SkillChallenge(
                prompt="List 3 key differences between HTTP and HTTPS.",
                rubric="Covers: encryption (TLS/SSL), security of data in transit, and trust/certificates. Accurate, exactly 3 points. May mention port differences (80 vs 443) as a bonus.",
            ),
            SkillChallenge(
                prompt="Summarize what version control (git) does in 2 sentences for a non-developer audience.",
                rubric="Exactly 2 sentences. Conveys: tracks changes to files over time, enables collaboration and rollback. Uses simple language without git-specific jargon. Accurate.",
            ),
        ],
    },
}


# ---------------------------------------------------------------------------
# Custom skill pack loader
# ---------------------------------------------------------------------------

def _parse_yaml_simple(text: str) -> dict:
    """Minimal YAML-subset parser for skill pack files.

    Handles the specific structure needed for custom skill packs (top-level
    scalars and a list of mappings under ``challenges``).  This avoids
    requiring PyYAML as a hard dependency.
    """
    result: dict = {}
    challenges: list[dict] = []
    current_challenge: dict | None = None
    current_key: str | None = None
    in_challenges = False

    for raw_line in text.splitlines():
        # Skip blank lines and comments
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Detect the challenges list
        if stripped == "challenges:":
            in_challenges = True
            continue

        if not in_challenges:
            # Top-level scalar: "key: value"
            if ":" in stripped:
                key, _, value = stripped.partition(":")
                result[key.strip()] = value.strip().strip('"').strip("'")
            continue

        # Inside challenges list
        indent = len(raw_line) - len(raw_line.lstrip())

        if stripped.startswith("- "):
            # New list item
            if current_challenge is not None:
                challenges.append(current_challenge)
            current_challenge = {}
            # The rest after "- " could be "key: value"
            rest = stripped[2:].strip()
            if ":" in rest:
                key, _, value = rest.partition(":")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                current_challenge[key] = value
                current_key = key
            continue

        if current_challenge is not None and ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            current_challenge[key] = value
            current_key = key
        elif current_challenge is not None and current_key is not None:
            # Continuation line for a multi-line value
            current_challenge[current_key] += " " + stripped.strip('"').strip("'")

    if current_challenge is not None:
        challenges.append(current_challenge)

    result["challenges"] = challenges
    return result


def load_custom_pack(path: str | Path) -> dict:
    """Load a custom skill pack from a YAML file.

    The YAML file should have the following structure::

        name: My Custom Pack
        description: What does this pack test?
        challenges:
          - prompt: "The challenge prompt text"
            rubric: "What a good answer looks like"
          - prompt: "Another challenge"
            rubric: "Another rubric"

    Args:
        path: Path to the YAML file.

    Returns:
        A dict compatible with ``SKILL_PACKS`` entries, containing ``name``,
        ``description``, and a ``challenges`` list of :class:`SkillChallenge`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is missing required fields or is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Custom skill pack file not found: {path}")

    raw = path.read_text(encoding="utf-8")

    # Try PyYAML first, fall back to simple parser
    try:
        import yaml  # type: ignore[import-untyped]
        data = yaml.safe_load(raw)
    except ImportError:
        data = _parse_yaml_simple(raw)

    if not isinstance(data, dict):
        raise ValueError(f"Custom skill pack must be a YAML mapping, got {type(data).__name__}")

    # Validate required top-level fields
    if "name" not in data:
        raise ValueError("Custom skill pack missing required field: 'name'")
    if "challenges" not in data or not data["challenges"]:
        raise ValueError("Custom skill pack missing required field: 'challenges' (must be a non-empty list)")

    if not isinstance(data["challenges"], list):
        raise ValueError("'challenges' must be a list of challenge definitions")

    # Validate and convert each challenge
    skill_challenges: list[SkillChallenge] = []
    for i, ch in enumerate(data["challenges"]):
        if not isinstance(ch, dict):
            raise ValueError(f"Challenge #{i + 1} must be a mapping with 'prompt' and 'rubric'")
        if "prompt" not in ch:
            raise ValueError(f"Challenge #{i + 1} missing required field: 'prompt'")
        if "rubric" not in ch:
            raise ValueError(f"Challenge #{i + 1} missing required field: 'rubric'")
        skill_challenges.append(
            SkillChallenge(
                prompt=str(ch["prompt"]),
                rubric=str(ch["rubric"]),
                weight=float(ch.get("weight", 1.0)),
            )
        )

    return {
        "name": data["name"],
        "description": data.get("description", ""),
        "challenges": skill_challenges,
    }


# ---------------------------------------------------------------------------
# Skill proof judge prompt
# ---------------------------------------------------------------------------

_SKILL_JUDGE_PROMPT = """You are a skill evaluator grading an AI agent's response to a challenge.

CHALLENGE:
{prompt}

RUBRIC (what a good answer looks like):
{rubric}

AI AGENT'S RESPONSE:
{response}

Grade the response on a scale of 0 to 10:
- 10: Exceptional — exceeds rubric requirements, flawless execution
- 8-9: Strong — meets all rubric criteria with minor imperfections
- 6-7: Competent — meets most criteria, some gaps
- 4-5: Partial — addresses the challenge but misses key criteria
- 2-3: Weak — attempt is present but significantly wrong or incomplete
- 0-1: Failed — wrong answer, off-topic, or no meaningful attempt

Respond with ONLY a JSON object:
{{"score": <integer 0-10>, "reason": "<1-2 sentence explanation>"}}"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SkillResult:
    """Result of a single skill challenge evaluation."""

    challenge: SkillChallenge
    response: str
    score: int  # 0-10
    reason: str
    cost: float = 0.0
    latency: float = 0.0


@dataclass
class SkillProof:
    """A complete skill proof for one skill category."""

    skill: str
    skill_name: str
    model: str
    results: list[SkillResult] = field(default_factory=list)
    timestamp: str = ""
    total_cost: float = 0.0

    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def grade(self) -> str:
        s = self.avg_score
        if s >= 9.0:
            return "A+"
        elif s >= 8.0:
            return "A"
        elif s >= 7.0:
            return "B"
        elif s >= 6.0:
            return "C"
        elif s >= 5.0:
            return "D"
        else:
            return "F"

    @property
    def passed(self) -> bool:
        return self.avg_score >= 6.0

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "skill_name": self.skill_name,
            "model": self.model,
            "avg_score": round(self.avg_score, 1),
            "grade": self.grade,
            "passed": self.passed,
            "total_cost": round(self.total_cost, 4),
            "timestamp": self.timestamp,
            "challenges": [
                {
                    "prompt": r.challenge.prompt[:100],
                    "score": r.score,
                    "reason": r.reason,
                    "cost": round(r.cost, 4),
                    "latency": round(r.latency, 2),
                }
                for r in self.results
            ],
        }


@dataclass
class SkillReport:
    """A complete skill report across multiple skill categories."""

    model: str
    proofs: list[SkillProof] = field(default_factory=list)
    timestamp: str = ""

    @property
    def overall_score(self) -> float:
        if not self.proofs:
            return 0.0
        return sum(p.avg_score for p in self.proofs) / len(self.proofs)

    @property
    def overall_grade(self) -> str:
        s = self.overall_score
        if s >= 9.0:
            return "A+"
        elif s >= 8.0:
            return "A"
        elif s >= 7.0:
            return "B"
        elif s >= 6.0:
            return "C"
        elif s >= 5.0:
            return "D"
        else:
            return "F"

    @property
    def fingerprint(self) -> str:
        """Deterministic hash of the report for verification."""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "overall_score": round(self.overall_score, 1),
            "overall_grade": self.overall_grade,
            "timestamp": self.timestamp,
            "skills": [p.to_dict() for p in self.proofs],
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_skill_proof(
    model: str,
    skill: str,
    judge_model: str = "openai/gpt-4o-mini",
    provider: "str | Provider | None" = None,
) -> SkillProof:
    """Run a skill proof evaluation for a single skill category.

    Args:
        model: The model to test (e.g. "claude-sonnet-4-6").
        skill: Skill pack name (e.g. "coding", "reasoning").
        judge_model: Model to use as judge in "provider/model" format.
        provider: Provider name, Provider instance, or None (auto-detected).

    Returns:
        SkillProof with scored results.
    """
    from proofagent.providers import get_provider
    from proofagent.providers.base import Provider as ProviderBase

    if skill not in SKILL_PACKS:
        available = ", ".join(SKILL_PACKS.keys())
        raise ValueError(f"Unknown skill '{skill}'. Available: {available}")

    pack = SKILL_PACKS[skill]
    if isinstance(provider, ProviderBase):
        provider = provider
    else:
        provider = get_provider(provider)

    # Parse judge model
    if "/" in judge_model:
        judge_provider_name, judge_model_name = judge_model.split("/", 1)
    else:
        judge_provider_name = None
        judge_model_name = judge_model

    judge_provider = get_provider(judge_provider_name)

    proof = SkillProof(
        skill=skill,
        skill_name=pack["name"],
        model=model,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    for challenge in pack["challenges"]:
        start = time.time()

        # Get the agent's response
        result = provider.complete(
            messages=[{"role": "user", "content": challenge.prompt}],
            model=model,
            temperature=0,
        )
        response_latency = time.time() - start
        response_cost = result.cost

        # Judge the response
        judge_prompt = _SKILL_JUDGE_PROMPT.format(
            prompt=challenge.prompt,
            rubric=challenge.rubric,
            response=result.text[:3000],
        )

        judge_result = judge_provider.complete(
            messages=[{"role": "user", "content": judge_prompt}],
            model=judge_model_name,
            temperature=0,
        )

        # Parse the judge's score
        score = 0
        reason = "Could not parse judge response"
        try:
            text = judge_result.text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            data = json.loads(text)
            score = int(data.get("score", 0))
            score = max(0, min(10, score))
            reason = data.get("reason", "")
        except (json.JSONDecodeError, ValueError, TypeError):
            match = re.search(r'"score"\s*:\s*(\d+)', judge_result.text)
            if match:
                score = max(0, min(10, int(match.group(1))))

        total_cost = response_cost + judge_result.cost
        proof.total_cost += total_cost

        proof.results.append(
            SkillResult(
                challenge=challenge,
                response=result.text,
                score=score,
                reason=reason,
                cost=total_cost,
                latency=response_latency,
            )
        )

        time.sleep(0.5)  # rate limit

    return proof


def run_skill_report(
    model: str,
    skills: list[str] | None = None,
    judge_model: str = "openai/gpt-4o-mini",
    provider: "str | Provider | None" = None,
) -> SkillReport:
    """Run skill proofs across multiple categories and produce a report.

    Args:
        model: The model to test.
        skills: List of skill names, or None for all skills.
        judge_model: Model to use as judge.
        provider: Provider name, Provider instance, or None (auto-detected).

    Returns:
        SkillReport with all proof results.
    """
    if skills is None:
        skills = list(SKILL_PACKS.keys())

    report = SkillReport(
        model=model,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    for skill in skills:
        proof = run_skill_proof(model, skill, judge_model, provider=provider)
        report.proofs.append(proof)

    return report


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def save_skill_report(
    report: SkillReport,
    output_dir: str | Path = ".proofagent/skills",
) -> Path:
    """Save a skill report to disk.

    Also generates a companion SVG badge file next to the JSON report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r'[^\w\-.]', '_', report.model)
    filename = f"skill_{safe_name}_{timestamp}.json"
    output_file = output_dir / filename

    data = report.to_dict()
    data["fingerprint"] = report.fingerprint

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    # Auto-generate badge SVG next to the JSON report
    try:
        from proofagent.badge import generate_badge_svg

        badge_filename = f"skill_{safe_name}_{timestamp}.svg"
        badge_file = output_dir / badge_filename
        svg = generate_badge_svg(
            label="proofagent",
            grade=report.overall_grade,
            score=report.overall_score,
        )
        badge_file.write_text(svg, encoding="utf-8")
    except Exception:
        pass  # Badge generation is non-critical

    return output_file
