"""Static figures for the predictive model (test window)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from processlens import viz


def pr_curve_figure(y: np.ndarray, probs: dict[str, np.ndarray], best: str, path: Path) -> Path:
    """PR curves of the selected and reference models against the base rate."""
    fig, ax = viz.new_figure(6.5, 4.5)
    shown = [k for k in probs if k != "dummy" and not k.endswith("_uncalibrated")]
    for color, key in zip(viz.SERIES, shown, strict=False):
        prec, rec, _ = precision_recall_curve(y, probs[key])
        ap = average_precision_score(y, probs[key])
        ax.step(rec, prec, where="post", color=color, linewidth=2, label=f"{key} (AP {ap:.3f})")
    ax.axhline(
        y.mean(),
        color=viz.INK_SECONDARY,
        linestyle="--",
        linewidth=1,
        label=f"base rate {y.mean():.3f}",
    )
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision–recall on the test window", loc="left")
    ax.legend(frameon=False, fontsize=9)
    return viz.save(fig, path)


def calibration_figure(rel: dict[str, dict[str, list[float]]], path: Path) -> Path:
    """Reliability diagram: observed failure rate vs predicted probability."""
    fig, ax = viz.new_figure(5.5, 4.5)
    top = max(max(r["mean_predicted"] + r["observed_rate"]) for r in rel.values())
    ax.plot(
        [0, top],
        [0, top],
        color=viz.MUTED,
        linestyle="--",
        linewidth=1,
        label="perfect calibration",
    )
    for color, (key, r) in zip(viz.SERIES, rel.items(), strict=False):
        ax.plot(
            r["mean_predicted"],
            r["observed_rate"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            label=key,
        )
    ax.set_xlabel("mean predicted probability (quantile bins)")
    ax.set_ylabel("observed failure rate")
    ax.set_title("Calibration on the test window", loc="left")
    ax.legend(frameon=False, fontsize=9)
    return viz.save(fig, path)


def gains_figure(policy: dict[str, Any], path: Path) -> Path:
    """Cumulative gains: share of failures caught vs share of runs inspected."""
    g = policy["gains"]
    fig, ax = viz.new_figure(5.5, 4.5)
    x = np.array(g["inspected_frac"]) * 100
    ax.plot(
        x, np.array(g["caught_frac"]) * 100, color=viz.SERIES[0], linewidth=2, label="model ranking"
    )
    ax.plot(
        [0, 100], [0, 100], color=viz.MUTED, linestyle="--", linewidth=1, label="random inspection"
    )
    for b in policy["budgets"]:
        ax.annotate(
            f"{b['caught_frac']:.0%}",
            (b["budget"] * 100, b["caught_frac"] * 100),
            textcoords="offset points",
            xytext=(4, -12),
            fontsize=8,
            color=viz.INK_SECONDARY,
        )
        ax.plot(b["budget"] * 100, b["caught_frac"] * 100, "o", color=viz.SERIES[0], markersize=6)
    ax.set_xlabel("% of runs inspected (riskiest first)")
    ax.set_ylabel("% of failures caught")
    ax.set_title("Gains on the test window", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    return viz.save(fig, path)


def cost_sensitivity_figure(policy: dict[str, Any], path: Path) -> Path:
    """Plot expected cost per run by cost ratio: model policy vs naive policies."""
    s = policy["sensitivity"]
    ratios = [f"{r['ratio']:g}:1" for r in s]
    xi = np.arange(len(s))
    w = 0.2
    series = [
        ("model policy", "cost_model"),
        ("inspect none", "cost_inspect_none"),
        ("inspect all", "cost_inspect_all"),
        ("random, same budget", "cost_random_same_budget"),
    ]
    fig, ax = viz.new_figure(8, 4)
    for i, (label, key) in enumerate(series):
        ax.bar(
            xi + (i - 1.5) * w,
            [r[key] for r in s],
            width=w - 0.02,
            color=viz.SERIES[i],
            label=label,
        )
    ax.set_xticks(xi, ratios)
    ax.set_xlabel("cost ratio (missed defect : inspection)")
    ax.set_ylabel("expected cost per run")
    ax.set_title("Cost sensitivity on the test window", loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    return viz.save(fig, path)


def make_model_figures(
    y: np.ndarray,
    probs: dict[str, np.ndarray],
    best: str,
    reliability: dict[str, Any],
    policy: dict[str, Any],
    fig_dir: Path,
) -> dict[str, Path]:
    """Write every Phase 2 figure and return their paths."""
    return {
        "pr_curve": pr_curve_figure(y, probs, best, fig_dir / "model_pr_curve.png"),
        "calibration": calibration_figure(reliability, fig_dir / "model_calibration.png"),
        "gains": gains_figure(policy, fig_dir / "model_gains.png"),
        "cost_sensitivity": cost_sensitivity_figure(policy, fig_dir / "model_cost_sensitivity.png"),
    }
