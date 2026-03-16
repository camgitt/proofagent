# Changelog

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
