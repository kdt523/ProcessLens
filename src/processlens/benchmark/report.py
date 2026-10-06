"""Benchmark figures, ``reports/benchmark.md`` and the generated ``docs/LIMITS.md``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from processlens import viz
from processlens.benchmark.metrics import check_gates, summarise
from processlens.benchmark.run import METHODS, run_benchmark
from processlens.config import PROJECT_ROOT

SINGLE = ["univariate", "stability", "shap"]
MECH_LABEL = {
    "linear": "linear",
    "threshold": "threshold (> q95)",
    "drift_window": "drift window (30% of time)",
}


def _pct(v: float | None) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0%}"


# ---------------------------------------------------------------- figures


def fig_hit_vs_beta(rows: pd.DataFrame, metric: str, path: Path) -> Path:
    """hit@5 vs effect size, one line per method, one panel per mechanism."""
    mechs = list(dict.fromkeys(rows["mechanism"]))
    fig, axes = plt.subplots(
        1, len(mechs), figsize=(4.2 * len(mechs), 3.8), dpi=110, sharey=True, squeeze=False
    )
    for ax, mech in zip(axes[0], mechs, strict=True):
        viz.style_axes(fig, ax)
        sub = rows[rows["mechanism"] == mech]
        for color, m in zip(viz.SERIES, METHODS, strict=False):
            s = sub[sub["method"] == m].groupby("beta")[metric].mean()
            ax.plot(
                s.index,
                s.to_numpy() * 100,
                color=color,
                marker="o",
                markersize=5,
                linewidth=2.6 if m == "consensus" else 1.6,
                label=m,
            )
        ax.axhline(80, color=viz.MUTED, linestyle="--", linewidth=1)
        ax.set_title(MECH_LABEL.get(mech, mech), loc="left", fontsize=10)
        ax.set_xlabel("planted effect β (log-odds per SD)")
    axes[0][0].set_ylabel(f"{metric} (%)")
    axes[0][0].set_ylim(0, 102)
    axes[0][-1].legend(frameon=False, fontsize=8, loc="lower right")
    return viz.save(fig, path)


def fig_mde(mde: dict[str, dict[str, float | None]], mechs: list[str], path: Path) -> Path:
    """Minimum detectable effect per mechanism and method (missing bar = not reached)."""
    fig, ax = viz.new_figure(7.5, 3.8)
    w = 0.2
    xi = np.arange(len(mechs))
    for i, (color, m) in enumerate(zip(viz.SERIES, METHODS, strict=False)):
        vals = [mde.get(m, {}).get(mech) for mech in mechs]
        pos = xi + (i - 1.5) * w
        ax.bar(pos, [v if v is not None else 0 for v in vals], width=w - 0.02, color=color, label=m)
        for p, v in zip(pos, vals, strict=True):
            if v is None:
                ax.text(p, 0.05, "not\nreached", ha="center", fontsize=6.5, color=viz.INK_SECONDARY)
    ax.set_xticks(xi, [MECH_LABEL.get(m, m) for m in mechs])
    ax.set_ylabel("MDE: smallest β with ≥80% hit@5")
    ax.set_title("Minimum detectable effect (cluster level, lower is better)", loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    return viz.save(fig, path)


def fig_null(nf: dict[str, Any], path: Path) -> Path:
    """False-alarm rates on no-cause scenarios."""
    labels = ["any sensor q < 0.05", "any cluster ≥ moderate", "any cluster strong"]
    keys = ["any_q_below_alpha", "any_moderate_or_strong", "any_strong"]
    kinds = list(nf.get("by_kind", {}))
    fig, ax = viz.new_figure(7, 3.5)
    w = 0.8 / max(len(kinds), 1)
    xi = np.arange(len(keys))
    for i, (color, kind) in enumerate(zip(viz.SERIES, kinds, strict=False)):
        vals = [nf["by_kind"][kind][k] * 100 for k in keys]
        ax.bar(
            xi + (i - (len(kinds) - 1) / 2) * w,
            vals,
            width=w - 0.02,
            color=color,
            label=f"{kind} labels (n={nf['by_kind'][kind]['n']})",
        )
    ax.set_xticks(xi, labels)
    ax.set_ylabel("% of null scenarios")
    ax.set_ylim(0, 100)
    ax.set_title("False alarms when there is no planted cause", loc="left")
    ax.legend(frameon=False, fontsize=8)
    return viz.save(fig, path)


def fig_consensus_gain(rows: pd.DataFrame, metric: str, path: Path) -> Path:
    """Consensus minus the best single method (chosen per cell, in hindsight)."""
    means = rows.groupby(["mechanism", "beta", "method"])[metric].mean().unstack()
    gain = (means["consensus"] - means[SINGLE].max(axis=1)) * 100
    fig, ax = viz.new_figure(7, 3.5)
    for color, (mech, s) in zip(viz.SERIES, gain.groupby(level=0), strict=False):
        s = s.droplevel(0)
        ax.plot(
            s.index,
            s.to_numpy(),
            color=color,
            marker="o",
            markersize=5,
            linewidth=2,
            label=MECH_LABEL.get(mech, mech),
        )
    ax.axhline(0, color=viz.MUTED, linewidth=1)
    ax.set_xlabel("planted effect β")
    ax.set_ylabel("percentage points")
    ax.set_title(f"Consensus vs best single method per cell ({metric})", loc="left")
    ax.legend(frameon=False, fontsize=8)
    return viz.save(fig, path)


# ---------------------------------------------------------------- text


def _mean(rows: pd.DataFrame, metric: str, **where: Any) -> float:
    sub = rows
    for k, v in where.items():
        sub = sub[sub[k] >= v[1]] if isinstance(v, tuple) else sub[sub[k] == v]
    return float(sub[metric].mean()) if len(sub) else float("nan")


def limits_lines(s: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """Plain-English limits, every number taken from the summary."""
    rows = s["_rows"]
    mk = s["mde"]["metric"]
    tgt = s["mde"]["target"]
    mde = s["mde"]["values"].get("consensus", {})
    top_beta = float(rows["beta"].max())
    lines = []
    for mech in dict.fromkeys(rows["mechanism"]):
        best = _mean(rows, mk, method="consensus", mechanism=mech, beta=top_beta)
        if mde.get(mech) is not None:
            lines.append(
                f"**{MECH_LABEL.get(mech, mech).capitalize()} faults:** the consensus ranks the "
                f"planted cluster in the top {cfg['mde_k']} in ≥ {tgt:.0%} of scenarios once the "
                f"effect is at least **β = {mde[mech]:g}** log-odds per SD "
                f"({_pct(best)} at β = {top_beta:g}). Smaller effects are missed more often."
            )
        else:
            lines.append(
                f"**{MECH_LABEL.get(mech, mech).capitalize()} faults:** never reaches "
                f"{tgt:.0%} top-{cfg['mde_k']} recovery on this grid; even at β = {top_beta:g} "
                f"it finds the planted cluster in only {_pct(best)} of scenarios."
            )
    big = {"beta": (">=", 1.0)}
    sens = _mean(rows, "sensor_hit@1", method="consensus", **big)
    clus = _mean(rows, "cluster_hit@1", method="consensus", **big)
    lines.append(
        f"**Correlated sensors cannot be separated.** At β ≥ 1 the consensus puts the planted "
        f"*cluster* first in {_pct(clus)} of scenarios but the exact planted *sensor* first in "
        f"{_pct(sens)}. Treat every member of a suspect cluster as equally suspect."
    )
    nc = s["by_n_causes"].get("consensus", {})
    if "1" in nc and "3" in nc:
        lines.append(
            f"**Several simultaneous causes dilute each other.** Averaged over the grid, "
            f"top-{cfg['mde_k']} recovery is {_pct(nc['1'])} with one planted cause and "
            f"{_pct(nc['3'])} per cause with three."
        )
    nf = s["null_false_alarms"]
    lines.append(
        f"**False alarms with no real cause** ({nf['n']} null scenarios): at least one sensor "
        f"passes q < {cfg['alpha']} in {_pct(nf['any_q_below_alpha'])}, a cluster is graded "
        f"moderate or strong in {_pct(nf['any_moderate_or_strong'])}, and graded strong in "
        f"{_pct(nf['any_strong'])}. A ranking always has a #1; only the evidence grade says "
        "whether it means anything."
    )
    miss = pd.DataFrame(s.get("by_cause_missing_beta_ge_1", []))
    if not miss.empty:
        parts = [
            f"{MECH_LABEL.get(m, m)}: "
            + ", ".join(
                f"{r.cause_missing} missing → {r['mean']:.0%} (n={r['count']})"
                for _, r in g.iterrows()
            )
            for m, g in miss.groupby("mechanism", sort=False)
        ]
        lines.append(
            "**Missing data in the cause sensor** (consensus top-5 recovery at β ≥ 1): "
            + "; ".join(parts)
            + "."
        )
    cons = s["overall_by_method"]["consensus"][mk]
    best_single = max(SINGLE, key=lambda m: s["overall_by_method"][m][mk])
    best_val = s["overall_by_method"][best_single][mk]
    lines.append(
        f"**Consensus vs single methods.** Averaged over all planted scenarios, consensus "
        f"top-{cfg['mde_k']} recovery is {_pct(cons)} vs {_pct(best_val)} for the best "
        f"single method ({best_single}). See the ablation figure per mechanism."
    )
    lines.append(
        "**Benchmark realism.** Planted faults are additive effects on the log-odds of one to "
        "three real sensors. Real defects may involve interactions, lags, sensors that were "
        "never measured, or label noise, so these numbers are an upper bound for comparable "
        "effect sizes, not a guarantee."
    )
    return lines


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


DESIGN = (
    "**Design.** The real SECOM sensor matrix (all runs, usable sensors, real missingness and "
    "correlation) is kept; labels are replaced by synthetic labels generated from known "
    "planted causes. The root-cause engine is run unchanged and scored on whether it "
    "recovers them. The base rate is held at {rate:.1%} by calibrating the intercept.\n\n"
    "- {n_planted} planted scenarios + {n_null} null scenarios ({n} total)\n"
    "- mechanisms: linear `logit = b0 + β·Σz(x)`; threshold `logit = b0 + β·z(1[x > q95])`; "
    "drift window (linear effect inside one contiguous {drift:.0%} stretch of time)\n"
    "- β in log-odds per standard deviation of the driver; 1–3 causes from distinct "
    "clusters\n"
    "- hit@k = share of planted causes found in the top k (sensor level: the exact sensor; "
    "cluster level: its correlated cluster)"
)


def render_benchmark_md(
    s: dict[str, Any], gates: dict[str, Any], cfg: dict[str, Any], smoke: bool
) -> str:
    """Return the benchmark report markdown."""
    ov = pd.DataFrame(s["overall_by_method"]).T.loc[METHODS]
    cols = [c for c in ov.columns if "hit" in c]
    table = _md_table(
        ["method", *cols], [[m, *(f"{ov.loc[m, c]:.2f}" for c in cols)] for m in METHODS]
    )
    mde = s["mde"]["values"]
    mechs = list(dict.fromkeys(s["_rows"]["mechanism"]))
    mde_tab = _md_table(
        ["method", *mechs],
        [
            [
                m,
                *(
                    "not reached" if mde.get(m, {}).get(x) is None else f"{mde[m][x]:g}"
                    for x in mechs
                ),
            ]
            for m in METHODS
        ],
    )
    nf = s["null_false_alarms"]
    rt = s["runtime_seconds"]
    design = DESIGN.format(
        rate=cfg["target_base_rate"],
        n_planted=s["n_planted"],
        n_null=s["n_null"],
        n=s["n_scenarios"],
        drift=cfg["drift_window_frac"],
    )
    verdict = "PASSED" if gates["passed"] else "FAILED"
    sections = [
        "# Planted-fault benchmark",
        "_Generated by `processlens benchmark` from `reports/metrics/benchmark.json`. "
        "Do not edit by hand._" + (" **Smoke configuration.**" if smoke else ""),
        design,
        "## Recovery by method (averaged over all planted scenarios)",
        table,
        "## Recovery vs effect size (ablation)",
        "Cluster level:\n\n![hit vs beta](figures/benchmark_hit5_vs_beta.png)",
        "Sensor level (exact planted sensor):\n\n"
        "![sensor hit vs beta](figures/benchmark_sensor_hit5_vs_beta.png)",
        f"## Minimum detectable effect ({s['mde']['metric']} ≥ {s['mde']['target']:.0%})",
        mde_tab,
        "![mde](figures/benchmark_mde.png)",
        "## False alarms on null scenarios",
        f"{nf['n']} scenarios with no planted cause. Any sensor with q < {cfg['alpha']}: "
        f"{_pct(nf['any_q_below_alpha'])}; any cluster graded moderate or strong: "
        f"{_pct(nf['any_moderate_or_strong'])}; any cluster graded strong: "
        f"{_pct(nf['any_strong'])}.",
        "![null](figures/benchmark_null_false_alarms.png)",
        "## Consensus vs best single method",
        "Positive = consensus beats the best single method for that cell (chosen in "
        "hindsight, so this is a demanding comparison).",
        "![gain](figures/benchmark_consensus_gain.png)",
        "## Quality gate",
        f"Consensus cluster hit@5 at β ≥ {cfg['gates']['large_effect']:g}: "
        f"{gates['consensus_cluster_hit5_large_effect']:.2f} "
        f"(required ≥ {gates['min_required']}); null strong-evidence rate "
        f"{gates['null_any_strong_rate']:.2f} (allowed ≤ {gates['max_allowed']}). "
        f"**{verdict}**.",
        f"## Runtime\n\nMean {rt['mean']:.1f} s per full analysis (p95 {rt['p95']:.1f} s), "
        "measured while scenarios run in parallel on all cores.",
        "## What this means",
        "\n".join(f"- {line}" for line in limits_lines(s, cfg)),
    ]
    return "\n\n".join(sections) + "\n"


def render_limits_md(s: dict[str, Any], cfg: dict[str, Any]) -> str:
    """Return docs/LIMITS.md, generated from the benchmark."""
    return (
        "\n\n".join(
            [
                "# Known limits of ProcessLens",
                "_Generated from the planted-fault benchmark "
                "(`reports/metrics/benchmark.json`) by `processlens benchmark`. "
                "Do not edit by hand._",
                "ProcessLens ranks sensor clusters **associated with** failure. It does not prove "
                "causes. The limits below are measured, not assumed.",
                "\n".join(f"{i}. {line}" for i, line in enumerate(limits_lines(s, cfg), start=1)),
            ]
        )
        + "\n"
    )


def run_and_report(smoke: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Run the benchmark, write metrics JSON, figures, report and (full run) LIMITS.md."""
    results, ctx, _rc, bm = run_benchmark(smoke, root)
    s = summarise(results, ctx.clusters, bm, ctx.x.isna().mean())
    gates = check_gates(s, bm["gates"])
    rows = s["_rows"]
    out = bm["outputs"]
    fig_dir = root / "reports" / "figures"
    if not smoke:
        mechs = list(dict.fromkeys(rows["mechanism"]))
        fig_hit_vs_beta(rows, s["mde"]["metric"], fig_dir / "benchmark_hit5_vs_beta.png")
        fig_hit_vs_beta(rows, "sensor_hit@5", fig_dir / "benchmark_sensor_hit5_vs_beta.png")
        fig_mde(s["mde"]["values"], mechs, fig_dir / "benchmark_mde.png")
        fig_null(s["null_false_alarms"], fig_dir / "benchmark_null_false_alarms.png")
        fig_consensus_gain(rows, s["mde"]["metric"], fig_dir / "benchmark_consensus_gain.png")
    public = {k: v for k, v in s.items() if not k.startswith("_")}
    public["gates"] = gates
    public["smoke"] = smoke
    metrics_path = root / out["metrics"]
    if smoke:
        metrics_path = metrics_path.with_name("benchmark_smoke.json")
    metrics_path.write_text(json.dumps(public, indent=2, default=float), encoding="utf-8")
    if not smoke:
        (root / out["report"]).write_text(
            render_benchmark_md(s, gates, bm, smoke), encoding="utf-8"
        )
        (root / out["limits"]).write_text(render_limits_md(s, bm), encoding="utf-8")
    return public
