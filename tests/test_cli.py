"""Smoke tests for the CLI."""

from click.testing import CliRunner

from proofagent.cli import cli


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.7.2" in result.output


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "pytest for AI agents" in result.output


def test_report_no_results():
    runner = CliRunner()
    result = runner.invoke(cli, ["report", "--input", "/tmp/nonexistent"])
    assert "No results found" in result.output
