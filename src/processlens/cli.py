"""Command-line entry point: ``processlens <command>``."""

from __future__ import annotations

import json
import logging

import typer

app = typer.Typer(help="ProcessLens: manufacturing root-cause analysis on UCI SECOM.")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _not_yet(phase: int) -> None:
    typer.echo(f"Not implemented yet (Phase {phase}).")
    raise typer.Exit(code=1)


@app.command()
def ingest() -> None:
    """Download SECOM and build the processed parquet."""
    from processlens.data.ingest import run_ingest

    typer.echo(json.dumps(run_ingest(), indent=2))


@app.command()
def audit() -> None:
    """Generate the data audit report."""
    from processlens.data.audit import run_audit

    summary = run_audit()
    typer.echo(f"Wrote {summary['report']}")


@app.command()
def train() -> None:
    """Train models, evaluate and log to MLflow."""
    from processlens.models.train import run_train

    res = run_train()
    best = res["selected_model"]
    m = res["test"]["metrics"][f"{best}_calibrated"]
    typer.echo(
        f"Selected {best}; test PR-AUC {m['pr_auc']['value']:.3f} "
        f"[{m['pr_auc']['ci_low']:.3f}, {m['pr_auc']['ci_high']:.3f}] "
        f"vs base rate {m['base_rate']['value']:.3f}. Wrote reports/metrics/model.json"
    )


@app.command()
def rootcause(
    start: str | None = typer.Option(None, help="Window start (default: train window)"),
    end: str | None = typer.Option(None, help="Window end (default: train window)"),
    top_k: int | None = typer.Option(None, help="Number of suspect clusters to print"),
) -> None:
    """Rank suspect sensor clusters for a time window; print JSON."""
    from processlens.rootcause.engine import run_rootcause

    res = run_rootcause(start, end, top_k)
    keys = [
        "consensus_rank",
        "representative",
        "n_members",
        "q_value",
        "effect_size",
        "stability_freq",
        "shap_share",
        "evidence",
    ]
    typer.echo(
        json.dumps(
            {
                "window": res["window"],
                "suspects": [{k: s[k] for k in keys} for s in res["suspects"]],
            },
            indent=2,
        )
    )


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
