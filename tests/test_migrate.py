"""Tests for the Promptfoo → proofagent migration."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from proofagent.cli import cli
from proofagent.migrate import (
    _render_jinja,
    _slug,
    _translate_assertion,
    migrate_promptfoo,
)


# ─── helpers ──────────────────────────────────────────────────────────────


def test_slug_basic():
    assert _slug("Hello World") == "hello_world"
    assert _slug("Refuses harmful requests!") == "refuses_harmful_requests"


def test_slug_truncates():
    long = "a" * 100
    assert len(_slug(long)) == 40


def test_slug_empty_falls_back():
    assert _slug("") == "case"
    assert _slug("!!!") == "case"


def test_render_jinja_basic_substitution():
    assert _render_jinja("Hello {{name}}", {"name": "world"}) == "Hello world"


def test_render_jinja_handles_whitespace():
    assert _render_jinja("Hi {{ name }}", {"name": "Cam"}) == "Hi Cam"


def test_render_jinja_unknown_var_stays_literal():
    out = _render_jinja("Hi {{name}}, {{unknown}}", {"name": "Cam"})
    assert out == "Hi Cam, {{unknown}}"


def test_render_jinja_no_vars_returns_template():
    assert _render_jinja("static text", {}) == "static text"


# ─── assertion translation ───────────────────────────────────────────────


def test_translate_contains():
    code, kind = _translate_assertion({"type": "contains", "value": "hello"})
    assert code == "contains('hello')"
    assert kind == "contains"


def test_translate_not_contains():
    code, kind = _translate_assertion({"type": "not-contains", "value": "step 1"})
    assert "not_contains" in code
    assert kind == "not_contains"


def test_translate_icontains_case_insensitive():
    code, _ = _translate_assertion({"type": "icontains", "value": "Hi"})
    assert "case_sensitive=False" in code


def test_translate_regex():
    code, kind = _translate_assertion({"type": "regex", "value": r"\d+"})
    assert "matches_regex" in code
    assert kind == "regex"


def test_translate_contains_json():
    code, kind = _translate_assertion({"type": "contains-json"})
    assert code == "valid_json()"
    assert kind == "valid_json"


def test_translate_similar_with_threshold():
    code, kind = _translate_assertion(
        {"type": "similar", "value": "hello there", "threshold": 0.8}
    )
    assert "semantic_match" in code
    assert "threshold=0.8" in code
    assert kind == "semantic_match"


def test_translate_similar_default_threshold():
    code, _ = _translate_assertion({"type": "similar", "value": "hi"})
    assert "threshold=0.75" in code


def test_translate_contains_any_list():
    code, kind = _translate_assertion(
        {"type": "contains-any", "value": ["hola", "hi"]}
    )
    assert "any" in code
    assert kind == "contains_any"


def test_translate_cost_with_threshold():
    code, kind = _translate_assertion({"type": "cost", "threshold": 0.05})
    assert code == "total_cost_under(0.05)"
    assert kind == "cost"


def test_translate_latency():
    code, kind = _translate_assertion({"type": "latency", "threshold": 2.0})
    assert "latency_under" in code
    assert kind == "latency"


def test_translate_unsupported_llm_rubric():
    code, kind = _translate_assertion({"type": "llm-rubric", "value": "be helpful"})
    assert code is None
    assert kind == "unsupported:llm-rubric"


def test_translate_unsupported_javascript():
    code, kind = _translate_assertion({"type": "javascript", "value": "1+1"})
    assert code is None
    assert kind == "unsupported:javascript"


def test_translate_unknown_type():
    code, kind = _translate_assertion({"type": "wat", "value": "?"})
    assert code is None
    assert kind == "unsupported:wat"


# ─── end-to-end migration ────────────────────────────────────────────────


def _write_config(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "promptfooconfig.yaml"
    p.write_text(content)
    return p


def test_migrate_minimal_config(tmp_path):
    cfg = _write_config(
        tmp_path,
        """
prompts:
  - "Answer: {{q}}"
tests:
  - description: simple contains
    vars:
      q: "hello"
    assert:
      - type: contains
        value: "world"
""",
    )
    out = tmp_path / "test_out.py"
    result = migrate_promptfoo(cfg, out)
    assert result.test_count == 1
    assert result.handled.get("contains") == 1
    assert result.unsupported == {}
    body = out.read_text()
    assert "from proofagent import expect" in body
    assert "proofagent_run('Answer: hello')" in body
    assert ".contains('world')" in body


def test_migrate_inherits_default_assert(tmp_path):
    cfg = _write_config(
        tmp_path,
        """
prompts:
  - "{{q}}"
defaultTest:
  assert:
    - type: cost
      threshold: 0.05
tests:
  - vars:
      q: "a"
    assert:
      - type: contains
        value: "x"
  - vars:
      q: "b"
    assert:
      - type: contains
        value: "y"
""",
    )
    out = tmp_path / "test_out.py"
    result = migrate_promptfoo(cfg, out)
    assert result.test_count == 2
    # Cost should appear for BOTH tests (inherited from defaultTest)
    assert result.handled["cost"] == 2
    body = out.read_text()
    assert body.count("total_cost_under(0.05)") == 2


def test_migrate_preserves_unsupported_as_todos(tmp_path):
    cfg = _write_config(
        tmp_path,
        """
prompts:
  - "{{q}}"
tests:
  - vars:
      q: "test"
    assert:
      - type: llm-rubric
        value: "be helpful"
      - type: javascript
        value: "true"
""",
    )
    out = tmp_path / "test_out.py"
    result = migrate_promptfoo(cfg, out)
    assert result.unsupported.get("unsupported:llm-rubric") == 1
    assert result.unsupported.get("unsupported:javascript") == 1
    body = out.read_text()
    assert "TODO" in body
    assert "llm-rubric" in body
    assert "javascript" in body


def test_migrate_test_without_assertions_gets_safe_default(tmp_path):
    cfg = _write_config(
        tmp_path,
        """
prompts:
  - "{{q}}"
tests:
  - description: lazy test
    vars:
      q: "hi"
""",
    )
    out = tmp_path / "test_out.py"
    result = migrate_promptfoo(cfg, out)
    assert result.test_count == 1
    body = out.read_text()
    # Falls back to a not_refused() so the file is at least valid
    assert "not_refused()" in body


def test_migrate_no_prompt_uses_first_var(tmp_path):
    cfg = _write_config(
        tmp_path,
        """
tests:
  - vars:
      input: "raw prompt"
    assert:
      - type: contains
        value: "x"
""",
    )
    out = tmp_path / "test_out.py"
    migrate_promptfoo(cfg, out)
    body = out.read_text()
    assert "proofagent_run('raw prompt')" in body


def test_migrate_skips_non_dict_test_entries(tmp_path):
    cfg = _write_config(
        tmp_path,
        """
prompts:
  - "{{q}}"
tests:
  - vars:
      q: "valid"
    assert:
      - type: contains
        value: "x"
  - "this is not a dict"
""",
    )
    out = tmp_path / "test_out.py"
    result = migrate_promptfoo(cfg, out)
    assert result.test_count == 1
    assert result.skipped_tests == 1


def test_migrate_strips_file_prefix_from_prompts(tmp_path):
    cfg = _write_config(
        tmp_path,
        """
prompts:
  - "file://prompts/main.txt"
tests:
  - description: file prompt
    vars:
      q: "hi"
    assert:
      - type: contains
        value: "x"
""",
    )
    out = tmp_path / "test_out.py"
    migrate_promptfoo(cfg, out)
    body = out.read_text()
    # The 'file://' prefix should be stripped (we treat the path literally for now)
    assert "file://" not in body


def test_migrate_invalid_yaml_raises(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("- not a mapping\n- still not")
    with pytest.raises(ValueError):
        migrate_promptfoo(cfg)


def test_migrate_handles_missing_pyyaml(tmp_path, monkeypatch):
    """Friendly ImportError if PyYAML isn't installed."""
    import sys
    monkeypatch.setitem(sys.modules, "yaml", None)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("tests: []")
    with pytest.raises(ImportError, match="PyYAML"):
        migrate_promptfoo(cfg)


# ─── CLI integration ─────────────────────────────────────────────────────


def test_cli_migrate_command_exists():
    runner = CliRunner()
    result = runner.invoke(cli, ["migrate-from-promptfoo", "--help"])
    assert result.exit_code == 0
    assert "Promptfoo" in result.output


def test_cli_migrate_full_run(tmp_path):
    """Full CLI flow on a sample config."""
    cfg = _write_config(
        tmp_path,
        """
prompts:
  - "{{q}}"
tests:
  - description: smoke
    vars:
      q: "hello"
    assert:
      - type: contains
        value: "world"
""",
    )
    out = tmp_path / "test_migrated.py"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["migrate-from-promptfoo", str(cfg), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "Wrote 1 tests" in result.output
    assert out.exists()
    assert "from proofagent import expect" in out.read_text()
