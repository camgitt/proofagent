"""Custom pytest markers for provably tests."""

MARKERS = {
    "provably": "Mark test as a provably evaluation test",
    "safety": "Mark test as a safety/red-team evaluation",
    "agent": "Mark test as a multi-step agent evaluation",
    "cost": "Mark test that tracks cost assertions",
}
