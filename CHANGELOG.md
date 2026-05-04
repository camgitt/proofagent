# Changelog

## 0.9.0 (2026-05-03)

- **Promptfoo migration** — convert Promptfoo YAML configs to proofagent Python tests in one command:
  - `proofagent migrate-from-promptfoo path/to/promptfooconfig.yaml [--out tests/test_x.py]`
  - Auto-ports: contains, not-contains, icontains, regex, contains-json, similar (with threshold), contains-any, contains-all, cost, latency
  - Preserves unsupported assertion types (llm-rubric, javascript, custom JS) as `# TODO` comments rather than silently dropping
  - Inherits `defaultTest` assertions, supports per-test overrides, substitutes Jinja-style `{{var}}` placeholders
  - 31 unit tests covering the migration logic + CLI integration
- Context: OpenAI acquired Promptfoo on 2026-03-09. This command exists for users who want to migrate to a fully independent eval tool.

## 0.8.0 (2026-03-18)

- **Skill proofs** — evaluate agent competence in coding, reasoning, math, writing, summarization
  - `proofagent skill run <model> --skill all` — run skill evaluations with LLM-as-judge scoring
  - `proofagent skill submit <path>` — submit proof to leaderboard
  - `.skill_score()` assertion for pytest
  - SHA-256 fingerprinted reports
- **Custom skill packs** — define your own challenges via YAML (`--pack ./my_skills.yaml`)
- **Skills leaderboard page** — tab-based view, URL submission, copyable CLI walkthrough
- **Blog** — 4 articles including Promptfoo migration guide
- Add `not_refused()` assertion (inverse of `refused()`)
- Add `--provider` flag to `scan` and `skill run` commands
- Add cross-provider model comparison (`--provider-a`, `--provider-b`)
- Fix `refused()` false positives — broad patterns replaced with contextual phrases
- Fix XSS in dashboard and leaderboard (HTML escaping)
- Fix path traversal via model names in file paths
- Fix cost fallback warnings for unknown models
- Fix Ollama auto-detect to verify server is running
- Fix `test` command to use `sys.executable` instead of `python`
- Add `robots.txt` and `sitemap.xml`
- Add community infrastructure (CONTRIBUTING.md, issue templates, CODE_OF_CONDUCT.md)
- Remove certification language throughout — renamed to "evaluation reports"
- Remove fabricated model scores from leaderboard
- Remove TM symbols (no registration filed)
- Normalize navigation across all site pages
- Update comparison table and FAQ for Promptfoo's OpenAI acquisition
- Bump status to Beta

## 0.7.2 (2026-03-16)

- Add `proofagent doctor` command — checks Python, pytest, API keys, provider SDKs
- Fix: detect GEMINI_API_KEY in addition to GOOGLE_API_KEY
- Fix: add pytest as core dependency (was only in dev extras)
- Fix: clear error message when no API key is set instead of cryptic import error

## 0.7.0 (2026-03-16)

- Add multi-turn conversation testing (`Conversation` class, `turn_count`, `all_turns_cost_under`, `no_turn_refused`)
- Add regression snapshots (`matches_snapshot` assertion, `proofagent snapshot` CLI)
- Add model drift detection (`proofagent drift` — compare runs, catch regressions)
- Add cost optimizer (`proofagent optimize` — find cheapest model that passes your evals)
- Add 3 new prompt packs: bias (10), accuracy (10), hallucination (10)
- Add HTML compliance report generation (`proofagent report --format html`)
- Add GitHub Actions reusable action (`action.yml`)
- Add "Tested with proofagent" badge SVG

## 0.6.1 (2026-03-16)

- Fix PyPI description to accurately reflect current features
- Fix dashboard footer showing stale version number
- Add API provider compliance disclaimer to `proofagent scan`
- Point documentation URL to proofagent.dev
- Add py.typed marker for PEP 561 type checking support

## 0.6.0 (2026-03-16)

- Add `proofagent init` — interactive wizard that creates and runs your first test
- Add `proofagent scan` — run built-in safety pack against any model, get instant score
- Add built-in safety prompt pack (10 adversarial prompts)
- Fix Anthropic model alias mapping (claude-sonnet-4-6 -> claude-sonnet-4-20250514)
- Update default models: gpt-4.1-mini, gemini-2.5-flash

## 0.5.2 (2026-03-15)

- Fix model alias mapping for all Anthropic models
- Update OpenAI and Gemini cost tables with latest models

## 0.5.1 (2026-03-15)

- Add cost tracking to all providers
- Fix provider auto-detection from environment variables

## 0.5.0 (2026-03-14)

- Initial public release
- 16 assertions, 8 providers
- pytest plugin with `proofagent_run` fixture
- CLI: test, report, gate, dashboard, compare commands
- LLM-as-judge for semantic matching
- Dataset loaders (CSV, JSONL)
- Zero telemetry, zero cloud dependency
