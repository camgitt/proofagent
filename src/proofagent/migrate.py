"""Migrate Promptfoo YAML configs to proofagent Python tests.

Reads a Promptfoo `promptfooconfig.yaml` (or any path) and emits a
proofagent test module. Common assertion types map directly; the rest get
preserved as TODO comments so nothing is silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Maps Promptfoo assertion type → proofagent expectation snippet.
# Each mapper takes the assertion dict and returns a single line of code
# (the chained expect(...) call), or None if it can't translate cleanly.

@dataclass
class MigrationResult:
    out_path: Path
    test_count: int
    handled: dict[str, int]      # assertion type → count
    unsupported: dict[str, int]  # assertion type → count
    skipped_tests: int


def _py_str(s: Any) -> str:
    """Quote a string for safe Python source emission."""
    if not isinstance(s, str):
        s = str(s)
    return repr(s)


def _slug(s: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return (s or "case")[:max_len]


def _translate_assertion(a: dict) -> tuple[str | None, str]:
    """Translate one Promptfoo assertion dict.

    Returns (code, kind):
      code: chained .method(...) snippet without leading dot, or None
      kind: a label for stats ("contains", "regex", "unsupported:javascript", ...)
    """
    atype = (a.get("type") or "").lower()
    val = a.get("value")
    threshold = a.get("threshold")

    if atype == "contains":
        return f"contains({_py_str(val)})", "contains"
    if atype == "not-contains":
        return f"not_contains({_py_str(val)})", "not_contains"
    if atype == "icontains":
        return f"contains({_py_str(val)}, case_sensitive=False)", "contains"
    if atype == "equals":
        # Promptfoo equals is exact match; closest in proofagent is contains
        # with a TODO note that exact match should be tightened.
        return f"contains({_py_str(val)})  # was equals — tighten if needed", "equals_as_contains"
    if atype == "regex":
        return f"matches_regex({_py_str(val)})", "regex"
    if atype in ("contains-json", "is-json"):
        return "valid_json()", "valid_json"
    if atype == "similar":
        thr = float(threshold) if threshold is not None else 0.75
        return f"semantic_match({_py_str(val)}, threshold={thr})", "semantic_match"
    if atype == "contains-any":
        items = val if isinstance(val, list) else [val]
        return f"custom('contains_any', lambda r: any(s in r.text for s in {items!r}))", "contains_any"
    if atype == "contains-all":
        items = val if isinstance(val, list) else [val]
        return f"custom('contains_all', lambda r: all(s in r.text for s in {items!r}))", "contains_all"
    if atype == "cost":
        if threshold is not None:
            return f"total_cost_under({float(threshold)})", "cost"
    if atype == "latency":
        if threshold is not None:
            return f"latency_under({float(threshold)})", "latency"

    # Unsupported — return None plus a labeled kind so we can report it
    return None, f"unsupported:{atype or 'unknown'}"


_JINJA_VAR_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def _render_jinja(template: str, vars_: dict) -> str:
    """Substitute Promptfoo-style {{var}} placeholders. Unknown vars stay literal."""
    def repl(m: re.Match) -> str:
        v = vars_.get(m.group(1))
        return str(v) if v is not None else m.group(0)
    return _JINJA_VAR_RE.sub(repl, template)


def _render_test(idx: int, test: dict, default_assert: list[dict] | None = None, prompt_template: str | None = None) -> tuple[str, dict[str, int], dict[str, int]]:
    """Render one Python test function from one Promptfoo test entry."""
    handled: dict[str, int] = {}
    unsupported: dict[str, int] = {}
    desc = test.get("description") or test.get("name")
    vars_ = test.get("vars") or {}
    asserts = list(test.get("assert") or [])
    if default_assert:
        asserts = list(default_assert) + asserts

    name_seed = desc or "_".join(f"{k}_{v}" for k, v in vars_.items() if isinstance(v, (str, int))) or f"case_{idx}"
    fn_name = f"test_{_slug(name_seed)}_{idx}"

    # Per-test override beats top-level prompt; otherwise fall back to first var
    template = test.get("prompt") or prompt_template
    if template:
        prompt = _render_jinja(template, vars_) if vars_ else template
    elif vars_:
        # No template at all but we have vars — use the first scalar var as the prompt
        first_scalar = next((v for v in vars_.values() if isinstance(v, (str, int, float))), None)
        prompt = str(first_scalar) if first_scalar is not None else "{input}"
    else:
        prompt = "{input}"

    lines = [f"def {fn_name}(proofagent_run):"]
    if desc:
        lines.append(f'    """{desc}"""')
    lines.append(f"    result = proofagent_run({_py_str(prompt)})")

    if not asserts:
        lines.append("    # NOTE: no assertions in source; add at least one")
        lines.append("    expect(result).not_refused()")
        handled["not_refused"] = handled.get("not_refused", 0) + 1
    else:
        chain_parts: list[str] = []
        comments: list[str] = []
        for a in asserts:
            code, kind = _translate_assertion(a)
            if code is not None:
                chain_parts.append(code)
                handled[kind] = handled.get(kind, 0) + 1
            else:
                comments.append(f"    # TODO: port {kind!r} — value: {a.get('value')!r}")
                unsupported[kind] = unsupported.get(kind, 0) + 1
        if chain_parts:
            chained = "expect(result)." + ".".join(chain_parts)
            lines.append(f"    {chained}")
        else:
            lines.append("    # TODO: all assertions in this case were unsupported; see comments above")
        lines.extend(comments)

    return "\n".join(lines) + "\n", handled, unsupported


def migrate_promptfoo(config_path: Path, out_path: Path | None = None) -> MigrationResult:
    """Convert a Promptfoo config file to a proofagent test module."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required for migrate-from-promptfoo. Run: pip install pyyaml") from e

    config = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Promptfoo config at {config_path} did not parse as a mapping")

    tests = config.get("tests") or []
    default_test = config.get("defaultTest") or {}
    default_assert = default_test.get("assert") if isinstance(default_test, dict) else None

    # Top-level prompts — Promptfoo allows a list; we use the first as the
    # default template (per-test prompts override this if present).
    prompts = config.get("prompts") or []
    prompt_template: str | None = None
    if prompts:
        first = prompts[0]
        if isinstance(first, str):
            # Strip 'file://' prefix; otherwise treat as inline template
            prompt_template = first[7:] if first.startswith("file://") else first
        elif isinstance(first, dict):
            prompt_template = first.get("raw") or first.get("content")

    out_path = out_path or Path("tests") / "test_migrated_from_promptfoo.py"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        '"""Migrated from Promptfoo config.\n\n'
        f'Source: {config_path}\n'
        'Review TODOs — assertions like llm-rubric and javascript can\'t auto-port.\n'
        '"""\n\n'
        'from proofagent import expect\n\n'
    )

    rendered: list[str] = []
    handled_total: dict[str, int] = {}
    unsupported_total: dict[str, int] = {}
    skipped = 0

    for i, test in enumerate(tests):
        if not isinstance(test, dict):
            skipped += 1
            continue
        body, handled, unsupported = _render_test(i, test, default_assert, prompt_template)
        rendered.append(body)
        for k, v in handled.items():
            handled_total[k] = handled_total.get(k, 0) + v
        for k, v in unsupported.items():
            unsupported_total[k] = unsupported_total.get(k, 0) + v

    out_path.write_text(header + "\n".join(rendered))

    return MigrationResult(
        out_path=out_path,
        test_count=len(rendered),
        handled=handled_total,
        unsupported=unsupported_total,
        skipped_tests=skipped,
    )
