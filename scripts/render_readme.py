"""Render README.md and docs/MODEL_CARD.md from reports/metrics/*.json.

Every number in those files comes from here; never edit them by hand.
Usage: uv run python scripts/render_readme.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
REPO = "https://github.com/kdt523/ProcessLens"


def load(name: str) -> dict[str, Any] | None:
    path = METRICS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pct(v: float | None, d: int = 0) -> str:
    return "n/a" if v is None else f"{v * 100:.{d}f}%"


def ci(m: dict[str, float], d: int = 3) -> str:
    return f"{m['value']:.{d}f} [{m['ci_low']:.{d}f}, {m['ci_high']:.{d}f}]"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


# ---------------------------------------------------------------- sections


def data_section(a: dict[str, Any]) -> str:
    s, lab, t = a["shape"], a["label"], a["time"]
    segs = " / ".join(pct(x["fail_rate"], 1) for x in t["fail_rate_by_split_segment"])
    return (
        f"UCI SECOM: **{s['rows']:,} production runs**, **{s['sensors']} sensors**, "
        f"**{lab['failures']} failures ({pct(lab['base_rate'], 2)})**, "
        f"{t['start'][:10]} → {t['end'][:10]}. {len(a['constant_sensors'])} sensors are constant "
        f"and every run has missing values. The failure rate drifts over time "
        f"(train / validation / test windows: {segs}), so all validation is time-ordered."
    )


def model_section(m: dict[str, Any]) -> str:
    best = m["selected_model"]
    t = m["test"]["metrics"]
    ref = m["test"]["paired_vs_reference"]
    rows = []
    for key in [f"{best}_calibrated", ref["reference"], "dummy"]:
        x = t[key]
        rows.append(
            [
                key.replace("_", " "),
                f"{x['base_rate']['value']:.3f}",
                ci(x["pr_auc"]),
                ci(x["roc_auc"]),
                ci(x["brier"]),
                ci(x["ece"]),
            ]
        )
    b20 = next(b for b in m["policy_test"]["budgets"] if b["budget"] == 0.2)
    verdict = (
        "better" if ref["ci_low"] > 0 else "**not significantly better** than the simpler reference"
    )
    return "\n\n".join(
        [
            table(
                [
                    "model (test window, used once)",
                    "base rate",
                    "PR-AUC [95% CI]",
                    "ROC-AUC",
                    "Brier",
                    "ECE",
                ],
                rows,
            ),
            f"- Paired bootstrap PR-AUC difference ({best} − {ref['reference']}): "
            f"{ref['diff']:+.3f} [{ref['ci_low']:+.3f}, {ref['ci_high']:+.3f}]. The selected model "
            f"is {verdict}.",
            f"- Inspecting the riskiest **20%** of test runs catches **{b20['caught']} of "
            f"{b20['failures']} failures ({pct(b20['caught_frac'])})**, vs 20% for random "
            "inspection.",
            f"- At a {m['policy_test']['base']['ratio']:g}:1 missed-defect : inspection cost "
            f"ratio, the cost-optimal policy saves "
            f"{m['policy_test']['base']['savings_vs_best_naive']:+.3f} per run vs the best naive "
            "policy. The signal is weak, so the model rarely justifies an inspection on its own. "
            "Details in [model_report.md](reports/model_report.md).",
        ]
    )


def rootcause_section(r: dict[str, Any]) -> str:
    rows = [
        [
            str(s["consensus_rank"]),
            f"`{s['representative']}`",
            str(s["n_members"]),
            f"{s['q_value']:.2g}",
            f"{s['effect_size']:+.2f}",
            f"{s['stability_freq']:.2f}",
            s["evidence"],
        ]
        for s in r["suspects"][:5]
    ]
    w = r["window"]
    return (
        f"Training window ({w['runs']} runs, {w['failures']} failures), {r['sensors_analysed']} "
        f"sensors in {r['n_clusters']} correlation clusters. Evidence grades: "
        f"{r['evidence_counts']}.\n\n"
        + table(
            [
                "rank",
                "representative",
                "cluster size",
                "q (BH)",
                "effect (Cliff's δ)",
                "stability",
                "evidence",
            ],
            rows,
        )
        + "\n\nThese are **suspects associated with failure**, not proven causes."
    )


def benchmark_section(b: dict[str, Any]) -> str:
    ov = b["overall_by_method"]
    mk = b["mde"]["metric"]
    methods = ["univariate", "stability", "shap", "consensus"]
    rows = [
        [
            m,
            f"{ov[m]['cluster_hit@1']:.2f}",
            f"{ov[m]['cluster_hit@5']:.2f}",
            f"{ov[m]['sensor_hit@5']:.2f}",
        ]
        for m in methods
    ]
    mde = b["mde"]["values"]["consensus"]
    mde_txt = ", ".join(
        f"{k}: " + ("not reached" if v is None else f"β = {v:g}") for k, v in mde.items()
    )
    nf = b["null_false_alarms"]
    return "\n\n".join(
        [
            f"{b['n_planted']} planted-fault scenarios + {b['n_null']} null scenarios on the real "
            "sensor matrix.",
            table(["method", "cluster hit@1", "cluster hit@5", "sensor hit@5"], rows),
            f"- Minimum detectable effect for the consensus ({mk} ≥ {b['mde']['target']:.0%}): "
            f"{mde_txt} (log-odds per SD).",
            f"- No-cause scenarios: any sensor with q < 0.05 in {pct(nf['any_q_below_alpha'])}, "
            f"a cluster graded strong in {pct(nf['any_strong'])}.",
            f"- CI quality gate: consensus cluster hit@5 at large effect "
            f"{b['gates']['consensus_cluster_hit5_large_effect']:.2f} (≥ "
            f"{b['gates']['min_required']}), null strong-rate "
            f"{b['gates']['null_any_strong_rate']:.2f} (≤ {b['gates']['max_allowed']}).",
            "Full report: [benchmark.md](reports/benchmark.md).",
        ]
    )


def agent_section(g: dict[str, Any] | None) -> str:
    if g is None:
        return (
            "_Live copilot evaluation not run yet._ Add `GEMINI_API_KEY` to `.env` and run "
            "`make agent-eval` to fill this table (prompt v1 vs v2 on 40 planted-fault "
            "scenarios). Tests and CI run offline with a scripted model."
        )
    keys = [
        ("hit@3_planted", "hit@3 (planted)"),
        ("numeric_faithfulness", "numeric faithfulness"),
        ("fully_verified_rate", "fully verified"),
        ("abstention_accuracy_null", "abstains on null"),
        ("false_abstention_rate_beta_ge_1", "abstains at β ≥ 1"),
        ("schema_valid_rate", "schema valid"),
        ("latency_p50_s", "latency p50 (s)"),
        ("mean_cost_usd", "cost / report ($)"),
    ]
    rows = []
    for v, d in g["versions"].items():
        s = d["summary"]
        cells = []
        for k, _ in keys:
            val = s.get(k)
            if val is None:
                cells.append("n/a")
            elif k.startswith(("latency", "mean_cost")):
                cells.append(f"{val:.3g}")
            else:
                cells.append(pct(val))
        rows.append([v, *cells])
    return f"{g['n_scenarios']} scenarios, model `{g['model']}`.\n\n" + table(
        ["prompt", *(lbl for _, lbl in keys)], rows
    )


def limits_section() -> str:
    path = ROOT / "docs" / "LIMITS.md"
    if not path.exists():
        return "_Run `make benchmark`._"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln[:1].isdigit()]
    return "\n".join(lines)


# ---------------------------------------------------------------- documents


def render_readme() -> str:
    a, m, r, b = load("audit"), load("model"), load("rootcause"), load("benchmark")
    parts = [
        "# ProcessLens",
        "**Which process parameters are linked to product defects, and how much can we trust "
        "that answer?** A manufacturing root-cause analyzer with a planted-fault benchmark that "
        "measures its own limits, and a LangGraph copilot that reports only numbers it can "
        "verify.",
        "<!-- Generated by scripts/render_readme.py from reports/metrics/*.json. "
        "Do not edit by hand. -->",
        "## The problem",
        "Process engineers investigating a defect face hundreds of correlated sensors, missing "
        "data and a drifting process. ProcessLens is a small, honest version of that problem on "
        "the public UCI SECOM factory dataset: it predicts risky runs, ranks *suspect* sensor "
        "clusters, and measures how far those rankings can be trusted. "
        "Spec: [PROBLEM_SPEC.md](docs/PROBLEM_SPEC.md).",
        "## Approach",
        "1. **Data contract + audit:** pandera schema, missingness and drift audit "
        "([data_audit.md](reports/data_audit.md)).\n"
        "2. **Leakage-safe prediction:** time-ordered split, train-only `Pipeline`, rolling-"
        "origin CV, Platt calibration, bootstrap CIs, cost-based inspection policy, MLflow.\n"
        "3. **Root-cause ranking:** Mann–Whitney + BH-FDR, L1 stability selection, LightGBM "
        "SHAP, Spearman clusters, consensus with evidence grades ([METHODS.md](docs/METHODS.md)).\n"
        "4. **Planted-fault benchmark:** inject known causes into the real sensor matrix; "
        "measure hit@k, minimum detectable effect and null false alarms → "
        "[LIMITS.md](docs/LIMITS.md).\n"
        "5. **Guarded copilot:** LangGraph + Gemini Flash over typed tools; a verifier traces "
        "every number to a tool call; abstains on weak evidence; prompt v1 vs v2 eval.\n"
        "6. **Serving:** FastAPI, Streamlit dashboard, PSI drift monitoring, CI quality gates.",
        "Architecture diagram: [ARCHITECTURE.md](docs/ARCHITECTURE.md).",
        "## Results",
        "### Data",
        data_section(a) if a else "_Run `make audit`._",
        "### Defect prediction",
        model_section(m) if m else "_Run `make train`._",
        "### Top suspects (training window)",
        rootcause_section(r) if r else "_Run `make rootcause`._",
        "### How much to trust the ranking: planted-fault benchmark",
        benchmark_section(b) if b else "_Run `make benchmark`._",
        "### Copilot evaluation",
        agent_section(load("agent")),
        "## Known limits (generated from the benchmark)",
        limits_section(),
        "## How to run",
        "```bash\nmake setup        # uv sync + pre-commit\nmake all          # ingest → audit "
        "→ train → rootcause → benchmark → drift → render README\nmake api          # FastAPI "
        "on :8000 (docs at /docs)\nmake app          # Streamlit dashboard\nmake test lint\n"
        "make agent-eval   # live copilot eval (needs GEMINI_API_KEY in .env)\n"
        "make mlflow-ui    # experiment tracking\n```",
        "Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). Every number above is "
        "reproduced by `make all` from a clean clone.",
        "## Repo map",
        "```\nconfigs/         every tunable value (seeds, thresholds, costs, model names)\n"
        "src/processlens/ data · features · models · rootcause · benchmark · monitoring · "
        "agent · api\napp/             Streamlit dashboard\nreports/         generated metrics "
        "(JSON), reports and figures\ndocs/            spec, methods, limits, model card, "
        "architecture\nevals/           copilot scenarios and recorded LLM responses\n"
        "tests/           contract, leakage, methods, benchmark, verifier, agent, API, app\n```",
        "## What I'd do next",
        "- Validate suspects with process engineers and log the outcome of each investigation, "
        "turning the copilot into a feedback loop.\n"
        "- Replace the static parquet with a plant-historian feed and run drift checks on a "
        "schedule.\n"
        "- Add interaction and lagged-effect mechanisms to the benchmark, where the current "
        "methods are expected to be weakest.\n"
        "- Pre-register an improved consensus (e.g. univariate-weighted) and test it on fresh "
        "benchmark seeds.",
        f"---\nModel card: [MODEL_CARD.md](docs/MODEL_CARD.md) · Repository: {REPO}",
    ]
    return "\n\n".join(parts) + "\n"


def render_model_card() -> str:
    m, b, a = load("model"), load("benchmark"), load("audit")
    if not (m and b and a):
        return "# Model card\n\n_Run `make all` first._\n"
    best = m["selected_model"]
    t = m["test"]["metrics"][f"{best}_calibrated"]
    sp = m["split"]
    return (
        "\n\n".join(
            [
                "# Model card: ProcessLens defect-risk model and root-cause engine",
                "_Generated by `scripts/render_readme.py` from `reports/metrics/*.json`._",
                "## Intended use",
                "- **Risk score:** rank production runs for inspection under a fixed budget.\n"
                "- **Root-cause engine:** rank *suspect* sensor clusters for process engineers to "
                "investigate.\n"
                "- **Not for:** automatic rejection of product, or claims that a sensor causes "
                "defects.",
                "## Data",
                f"UCI SECOM, {a['shape']['rows']} runs × {a['shape']['sensors']} anonymised sensors, "
                f"failure base rate {pct(a['label']['base_rate'], 2)}. Time-ordered windows: "
                + "; ".join(
                    f"{k} {v['start'][:10]} → {v['end'][:10]} ({v['rows']} runs, "
                    f"{v['failures']} failures)"
                    for k, v in sp.items()
                )
                + ".",
                "## Model",
                f"{best.replace('_', ' ')} selected by rolling-origin CV inside the training window; "
                f"{m['calibration']} calibration fit on validation; preprocessing (drop, median "
                "imputation + missing indicators, scaling for linear models) fit on training rows only.",
                "## Performance (test window, evaluated once, 95% bootstrap CIs)",
                table(
                    ["metric", "value"],
                    [
                        ["base rate", f"{t['base_rate']['value']:.3f}"],
                        ["PR-AUC", ci(t["pr_auc"])],
                        ["ROC-AUC", ci(t["roc_auc"])],
                        ["Brier", ci(t["brier"])],
                        ["ECE", ci(t["ece"])],
                    ],
                ),
                "## Root-cause engine reliability (planted-fault benchmark)",
                benchmark_section(b),
                "## Limits",
                limits_section(),
                "## Ethical and operational risks",
                "- **False alarms cost inspection time.** On no-cause scenarios the engine still "
                f"flags a q < 0.05 sensor in {pct(b['null_false_alarms']['any_q_below_alpha'])} of "
                "cases; act on the evidence grade, not the rank.\n"
                "- **Not causal.** Rankings are associations in observational data; confirm with "
                "engineering knowledge or controlled trials.\n"
                "- **Drift.** The failure rate and many sensors shift over time "
                "(see `reports/metrics/drift.json`); re-validate before use on new periods.\n"
                "- **Weak predictive signal.** The risk score is only modestly better than the base "
                "rate; it supports prioritisation, not decisions about individual runs.\n"
                "- **LLM copilot.** Reports are restricted to verified tool outputs, but wording can "
                "still mislead; the verifier blocks unverified numbers and causal phrasing.",
            ]
        )
        + "\n"
    )


def main() -> None:
    (ROOT / "README.md").write_text(render_readme(), encoding="utf-8")
    (ROOT / "docs" / "MODEL_CARD.md").write_text(render_model_card(), encoding="utf-8")
    print("Wrote README.md and docs/MODEL_CARD.md")


if __name__ == "__main__":
    main()
