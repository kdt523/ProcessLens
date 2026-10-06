"""Command-line entry point: ``processlens <command>``."""

from __future__ import annotations

import typer

app = typer.Typer(help="ProcessLens: manufacturing root-cause analysis on UCI SECOM.")


def _not_yet(phase: int) -> None:
    typer.echo(f"Not implemented yet (Phase {phase}).")
    raise typer.Exit(code=1)


@app.command()
def ingest() -> None:
    """Download SECOM and build the processed parquet."""
    _not_yet(1)


@app.command()
def audit() -> None:
    """Generate the data audit report."""
    _not_yet(1)


@app.command()
def train() -> None:
    """Train models, evaluate and log to MLflow."""
    _not_yet(2)


@app.command()
def rootcause() -> None:
    """Rank suspect sensor clusters."""
    _not_yet(3)


@app.command()
def benchmark() -> None:
    """Run the planted-fault benchmark."""
    _not_yet(4)


@app.command("agent-eval")
def agent_eval() -> None:
    """Evaluate the copilot (prompt v1 vs v2)."""
    _not_yet(5)


@app.command()
def drift() -> None:
    """Compute PSI drift between reference and recent windows."""
    _not_yet(6)


if __name__ == "__main__":
    app()
