"""Terminal display formatting."""

from __future__ import annotations

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RST = "\033[0m"


def pass_text(text: str) -> str:
    return f"{GREEN}{text}{RST}"


def fail_text(text: str) -> str:
    return f"{RED}{text}{RST}"


def info_text(text: str) -> str:
    return f"{CYAN}{text}{RST}"


def cost_text(amount: float) -> str:
    return f"{YELLOW}${amount:.4f}{RST}"


def header(text: str) -> str:
    return f"\n{BOLD}{text}{RST}\n{'=' * len(text)}"


def format_score(passed: int, total: int) -> str:
    if total == 0:
        return f"{DIM}no tests{RST}"
    rate = passed / total
    color = GREEN if rate >= 0.85 else YELLOW if rate >= 0.5 else RED
    return f"{color}{rate:.0%}{RST} ({passed}/{total})"
