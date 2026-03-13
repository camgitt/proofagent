"""Pytest plugin for provably — auto-registered via entry point."""

from __future__ import annotations

from provably.markers import MARKERS


def pytest_configure(config):
    """Register custom markers."""
    for name, description in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")


# Register fixtures by importing them — pytest discovers them from here
from provably.fixtures import provably_config, provably_provider, provably_run  # noqa: E402, F401


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print provably summary after test run."""
    stats = terminalreporter.stats
    passed = len(stats.get("passed", []))
    failed = len(stats.get("failed", []))
    total = passed + failed

    if total == 0:
        return

    score = passed / total if total > 0 else 0
    terminalreporter.write_sep("=", "provably summary")
    terminalreporter.write_line(f"  Pass rate: {score:.0%} ({passed}/{total})")

    if failed > 0:
        terminalreporter.write_line(f"  Failed: {failed}")
        for report in stats.get("failed", []):
            terminalreporter.write_line(f"    FAIL {report.nodeid}")
