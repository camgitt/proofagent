"""Provably CLI — test, report, and gate commands."""

from __future__ import annotations

import sys

import click

from provably.__version__ import __version__


@click.group()
@click.version_option(__version__, prog_name="provably")
def cli():
    """Provably — pytest for AI agents."""
    pass


@cli.command()
@click.argument("path", default="tests/")
@click.option("--model", default=None, help="Override model for all tests")
@click.option("--provider", default=None, help="Override provider")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option("-k", default=None, help="Only run tests matching expression")
def test(path, model, provider, verbose, k):
    """Run provably eval tests."""
    import subprocess

    cmd = ["python", "-m", "pytest", path, "--tb=short"]
    if verbose:
        cmd.append("-v")
    if k:
        cmd.extend(["-k", k])

    # Pass overrides via environment
    import os

    env = os.environ.copy()
    if model:
        env["PROVABLY_MODEL"] = model
    if provider:
        env["PROVABLY_PROVIDER"] = provider

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


@cli.command()
@click.option(
    "--input", "input_path", default=".provably/results", help="Results directory"
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["terminal", "json"]),
    default="terminal",
)
def report(input_path, fmt):
    """Show latest eval results."""
    from provably.report import load_latest_results, print_summary

    data = load_latest_results(input_path)
    if data is None:
        click.echo("No results found. Run 'provably test' first.")
        sys.exit(1)

    if fmt == "json":
        import json

        click.echo(json.dumps(data, indent=2))
    else:
        print_summary(data)


@cli.command()
@click.option("--min-score", default=0.85, help="Minimum pass rate (0.0-1.0)")
@click.option("--max-cost", default=None, type=float, help="Maximum total cost (USD)")
@click.option("--block-on-fail", is_flag=True, help="Exit code 1 if gate fails")
@click.option(
    "--input", "input_path", default=".provably/results", help="Results directory"
)
def gate(min_score, max_cost, block_on_fail, input_path):
    """CI quality gate — check if eval results meet thresholds."""
    from provably import display
    from provably.report import load_latest_results

    data = load_latest_results(input_path)
    if data is None:
        click.echo("No results found. Run 'provably test' first.")
        sys.exit(1)

    summary = data.get("summary", {})
    score = summary.get("score", 0)
    total_cost = summary.get("total_cost", 0)
    passed = True

    click.echo(display.header("Provably Gate"))
    click.echo(
        f"  Score: {display.format_score(summary.get('passed', 0), summary.get('total', 0))}"
    )

    if score < min_score:
        click.echo(
            display.fail_text(f"  FAIL: Score {score:.0%} < {min_score:.0%}")
        )
        passed = False
    else:
        click.echo(
            display.pass_text(f"  PASS: Score {score:.0%} >= {min_score:.0%}")
        )

    if max_cost is not None:
        if total_cost > max_cost:
            click.echo(
                display.fail_text(
                    f"  FAIL: Cost ${total_cost:.4f} > ${max_cost:.4f}"
                )
            )
            passed = False
        else:
            click.echo(
                display.pass_text(
                    f"  PASS: Cost ${total_cost:.4f} <= ${max_cost:.4f}"
                )
            )

    if not passed and block_on_fail:
        sys.exit(1)
