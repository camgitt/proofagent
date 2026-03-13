"""Tests for the pytest plugin registration."""

import pytest


def test_plugin_markers(pytestconfig):
    """Verify provably markers are registered."""
    markers = pytestconfig.getini("markers")
    marker_names = [m.split(":")[0].strip() for m in markers]
    assert "provably" in marker_names
    assert "safety" in marker_names
    assert "agent" in marker_names
