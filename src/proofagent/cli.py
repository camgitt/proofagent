"""proofagent CLI — test, report, and gate commands."""

from __future__ import annotations

import sys

import click

from proofagent.__version__ import __version__


@click.group()
@click.version_option(__version__, prog_name="proofagent")
def cli():
    """proofagent — pytest for AI agents."""
    pass


@cli.command()
def init():
    """Create a starter test file and run it. The fastest way to get started."""
    import os
    import subprocess

    click.echo()
    click.echo("  \033[1mproofagent init\033[0m")
    click.echo()

    # Check if test file already exists
    filename = "test_my_agent.py"
    if os.path.exists(filename):
        click.echo(f"  {filename} already exists. Running it...")
        click.echo()
        result = subprocess.run(
            [sys.executable, "-m", "pytest", filename, "-v", "--tb=short"]
        )
        sys.exit(result.returncode)

    # Ask what they want to test
    click.echo("  What do you want to test?")
    click.echo()
    click.echo("    \033[36m1\033[0m  Safety — does the model refuse harmful requests?")
    click.echo("    \033[36m2\033[0m  Accuracy — does the model give correct answers?")
    click.echo("    \033[36m3\033[0m  Tool use — does the agent call the right tools?")
    click.echo("    \033[36m4\033[0m  All of the above")
    click.echo()

    choice = click.prompt("  Pick one", type=click.IntRange(1, 4), default=4)
    click.echo()

    # Build the test file
    imports = "from proofagent import expect, LLMResult"
    tests = []

    if choice in (1, 4):
        tests.append('''
def test_refusal():
    """Model should refuse harmful requests."""
    result = LLMResult(text="I can't help with that. This request is harmful.")
    expect(result).refused()
''')

    if choice in (2, 4):
        tests.append('''
def test_correct_answer():
    """Model should give the right answer."""
    result = LLMResult(text="The answer is 4.")
    expect(result).contains("4")
''')

    if choice in (3, 4):
        imports = "from proofagent import expect, LLMResult, ToolCall"
        tests.append('''
def test_tool_usage():
    """Agent should check limits before trading."""
    result = LLMResult(
        text="Bought 10 AAPL",
        tool_calls=[
            ToolCall(name="check_limit", args={}),
            ToolCall(name="execute_trade", args={}),
        ],
    )
    expect(result).tool_calls_contain("check_limit")
    expect(result).tool_calls_contain("execute_trade")
''')

    content = imports + "\n" + "\n".join(tests)

    with open(filename, "w") as f:
        f.write(content)

    click.echo(f"  Created \033[1m{filename}\033[0m")
    click.echo()

    # Run it
    if click.confirm("  Run it now?", default=True):
        click.echo()
        result = subprocess.run(
            [sys.executable, "-m", "pytest", filename, "-v", "--tb=short"]
        )

        if result.returncode == 0:
            click.echo()
            click.echo("  \033[32mAll passing.\033[0m Edit the file to add your own tests.")
            click.echo("  Docs: https://proofagent.dev/quickstart")
            click.echo()

        sys.exit(result.returncode)


@cli.command()
@click.argument("model")
@click.option("--pack", "pack_name", default="safety", help="Prompt pack to run (default: safety)")
def scan(model, pack_name):
    """Run a built-in prompt pack against a model and see the results.

    Example: proofagent scan claude-sonnet-4-6
    """
    import time

    from proofagent.expect import Expectation, _REFUSAL_PATTERNS
    from proofagent.packs import get_pack
    from proofagent.providers import get_provider
    from proofagent.result import LLMResult

    pack = get_pack(pack_name)

    click.echo()
    click.echo(f"  \033[1mproofagent scan\033[0m — {pack['name']} Pack")
    click.echo(f"  Model: {model}")
    click.echo(f"  {len(pack['prompts'])} prompts")
    click.echo()

    try:
        provider = get_provider()
    except Exception as e:
        click.echo(f"  \033[31mError:\033[0m {e}")
        click.echo("  Set an API key: export ANTHROPIC_API_KEY=... (or OPENAI_API_KEY, GOOGLE_API_KEY)")
        sys.exit(1)

    passed = 0
    failed = 0
    total_cost = 0.0

    for i, prompt in enumerate(pack["prompts"], 1):
        try:
            result = provider.complete(
                messages=[{"role": "user", "content": prompt}],
                model=model,
            )
            total_cost += result.cost

            # Check the assertion
            try:
                exp = Expectation(result)
                getattr(exp, pack["assertion"])()
                passed += 1
                click.echo(f"  \033[32m✓\033[0m [{i}/{len(pack['prompts'])}] {prompt[:60]}")
            except AssertionError:
                failed += 1
                click.echo(f"  \033[31m✗\033[0m [{i}/{len(pack['prompts'])}] {prompt[:60]}")
                click.echo(f"    \033[2m→ {result.text[:80]}\033[0m")

            time.sleep(0.5)  # rate limit

        except Exception as e:
            failed += 1
            click.echo(f"  \033[31m✗\033[0m [{i}/{len(pack['prompts'])}] Error: {e}")

    total = passed + failed
    score = passed / total if total else 0
    grade = "A+" if score == 1.0 else "A" if score >= 0.9 else "B" if score >= 0.8 else "C" if score >= 0.7 else "F"

    click.echo()
    click.echo(f"  Score: {passed}/{total} ({score:.0%}) — Grade: {grade}")
    click.echo(f"  Cost: ${total_cost:.4f}")
    click.echo()

    if failed > 0:
        sys.exit(1)


@cli.command()
@click.argument("path", default="tests/")
@click.option("--model", default=None, help="Override model for all tests")
@click.option("--provider", default=None, help="Override provider")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option("-k", default=None, help="Only run tests matching expression")
def test(path, model, provider, verbose, k):
    """Run proofagent eval tests."""
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
        env["PROOFAGENT_MODEL"] = model
    if provider:
        env["PROOFAGENT_PROVIDER"] = provider

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


@cli.command()
@click.option(
    "--input", "input_path", default=".proofagent/results", help="Results directory"
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["terminal", "json"]),
    default="terminal",
)
def report(input_path, fmt):
    """Show latest eval results."""
    from proofagent.report import load_latest_results, print_summary

    data = load_latest_results(input_path)
    if data is None:
        click.echo("No results found. Run 'proofagent test' first.")
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
    "--input", "input_path", default=".proofagent/results", help="Results directory"
)
def gate(min_score, max_cost, block_on_fail, input_path):
    """CI quality gate — check if eval results meet thresholds."""
    from proofagent import display
    from proofagent.report import load_latest_results

    data = load_latest_results(input_path)
    if data is None:
        click.echo("No results found. Run 'proofagent test' first.")
        sys.exit(1)

    summary = data.get("summary", {})
    score = summary.get("score", 0)
    total_cost = summary.get("total_cost", 0)
    passed = True

    click.echo(display.header("proofagent Gate"))
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


@cli.command()
@click.option("--test", "test_path", default=None, help="Run tests first, then show results")
@click.option("--port", default=7175, help="Port to serve on")
@click.option("--no-browser", is_flag=True, help="Don't auto-open browser")
def dashboard(test_path, port, no_browser):
    """Open the web dashboard to view eval results."""
    from proofagent.dashboard import serve

    click.echo()
    click.echo("  \033[32mproofagent\033[0m dashboard")
    click.echo()
    serve(test_path=test_path, port=port, no_browser=no_browser)


@cli.command(name="compare")
@click.argument("prompt", required=False)
@click.option("--model-a", required=True, help="First model to compare")
@click.option("--model-b", required=True, help="Second model to compare")
@click.option("--provider", default=None, help="Provider name (auto-detected if omitted)")
@click.option("--system", default=None, help="System message for both models")
def compare_cmd(prompt, model_a, model_b, provider, system):
    """Compare two models on the same prompt (A vs B testing).

    Reads prompt from argument or stdin if not provided.
    """
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        if not prompt:
            click.echo("Error: provide a prompt as an argument or via stdin.")
            sys.exit(1)

    from proofagent.compare import compare

    result = compare(
        prompt=prompt,
        model_a=model_a,
        model_b=model_b,
        provider=provider,
        system=system,
    )

    _print_compare_table(result)


def _print_compare_table(result):
    """Print a side-by-side comparison table to the terminal."""
    width = 40
    sep = "+" + "-" * (width + 2) + "+" + "-" * (width + 2) + "+"

    def _wrap(text: str, w: int) -> list[str]:
        """Wrap text into lines of at most w characters."""
        lines = []
        for paragraph in text.split("\n"):
            while len(paragraph) > w:
                lines.append(paragraph[:w])
                paragraph = paragraph[w:]
            lines.append(paragraph)
        return lines or [""]

    click.echo()
    click.echo(sep)
    click.echo(
        f"| {result.model_a:<{width}} | {result.model_b:<{width}} |"
    )
    click.echo(sep)

    lines_a = _wrap(result.result_a.text, width)
    lines_b = _wrap(result.result_b.text, width)
    max_lines = max(len(lines_a), len(lines_b))

    for i in range(max_lines):
        cell_a = lines_a[i] if i < len(lines_a) else ""
        cell_b = lines_b[i] if i < len(lines_b) else ""
        click.echo(f"| {cell_a:<{width}} | {cell_b:<{width}} |")

    click.echo(sep)

    # Stats row
    click.echo(
        f"| {'Latency: %.2fs' % result.result_a.latency:<{width}} "
        f"| {'Latency: %.2fs' % result.result_b.latency:<{width}} |"
    )
    click.echo(
        f"| {'Cost: $%.4f' % result.result_a.cost:<{width}} "
        f"| {'Cost: $%.4f' % result.result_b.cost:<{width}} |"
    )
    click.echo(
        f"| {'Tokens: %d in / %d out' % (result.result_a.input_tokens, result.result_a.output_tokens):<{width}} "
        f"| {'Tokens: %d in / %d out' % (result.result_b.input_tokens, result.result_b.output_tokens):<{width}} |"
    )
    click.echo(sep)

    if result.winner:
        click.echo(f"\n  Winner: {result.winner}")
    click.echo()
