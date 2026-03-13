"""Configuration loader — reads provably.json or [tool.provably] from pyproject.toml."""

from __future__ import annotations

import json
from pathlib import Path

_DEFAULTS = {
    "provider": None,  # auto-detect
    "model": None,  # provider default
    "judge_model": "openai/gpt-4o-mini",
    "results_dir": ".provably/results",
    "min_score": 0.85,
}


def load_config(project_dir: Path | None = None) -> dict:
    """Load provably config from provably.json or pyproject.toml."""
    project_dir = project_dir or Path.cwd()
    config = dict(_DEFAULTS)

    # Try provably.json first
    config_file = project_dir / "provably.json"
    if config_file.exists():
        with open(config_file) as f:
            config.update(json.load(f))
        return config

    # Try pyproject.toml [tool.provably]
    pyproject = project_dir / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return config
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        provably_config = data.get("tool", {}).get("provably", {})
        config.update(provably_config)

    return config
