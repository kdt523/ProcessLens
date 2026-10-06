"""Model & limits: PR curve with CI band, calibration, gains, cost-sensitivity slider."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import MUTED, SERIES, pct, require, style
from sklearn.metrics import precision_recall_curve

from processlens.models.policy import evaluate_policy, gains_curve

m = require("model", "make train")
best = m["selected_model"]
t = m["test"]["metrics"][f"{best}_calibrated"]
y = np.array(m["test_predictions"]["y"])
p = np.array(m["test_predictions"]["p_calibrated"])
pv = m["test"]["paired_vs_reference"]

with st.container(horizontal=True):
    pr = t["pr_auc"]
    st.metric(
        "Test PR-AUC",
        f"{pr['value']:.3f}",
        border=True,
        help=f"95% bootstrap CI {pr['ci_low']:.3f} – {pr['ci_high']:.3f}",
    )
    st.metric("Base rate (no-skill PR-AUC)", f"{t['base_rate']['value']:.3f}", border=True)
    st.metric("ROC-AUC", f"{t['roc_auc']['value']:.3f}", border=True)
    st.metric("ECE (calibrated)", f"{t['ece']['value']:.3f}", border=True)
st.caption(
    f"Selected **{best}** by rolling-origin CV; test window used once "
    f"({t['base_rate']['n']} runs, {t['base_rate']['positives']} failures). "
    f"Paired bootstrap vs {pv['reference']}: PR-AUC difference {pv['diff']:+.3f} "
    f"[{pv['ci_low']:+.3f}, {pv['ci_high']:+.3f}] — "
    + ("better." if pv["ci_low"] > 0 else "not distinguishable from the simpler model.")
)


@st.cache_data(max_entries=4)
def pr_band(y: np.ndarray, p: np.ndarray, n: int = 300) -> pd.DataFrame:
    """Bootstrap band for precision at a fixed recall grid (computed from stored predictions)."""
    grid = np.linspace(0, 1, 51)
    rng = np.random.default_rng(0)
    curves = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            prec, rec, _ = precision_recall_curve(y[i], p[i])
            curves.append(np.interp(grid, rec[::-1], np.maximum.accumulate(prec[::-1])))
    c = np.array(curves)
    return pd.DataFrame(
        {"recall": grid, "lo": np.quantile(c, 0.025, 0), "hi": np.quantile(c, 0.975, 0)}
    )


left, right = st.columns(2)
with left, st.container(border=True):
    st.markdown("**Precision–recall (test) with 95% bootstrap band**")
    band = pr_band(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    fig = go.Figure(
        [
            go.Scatter(
                x=band["recall"],
                y=band["hi"],
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=band["recall"],
                y=band["lo"],
                fill="tonexty",
                line={"width": 0},
                fillcolor="rgba(42,120,214,0.15)",
                name="95% band",
                hoverinfo="skip",
            ),
            go.Scatter(
                x=rec, y=prec, line={"color": SERIES[0], "width": 2, "shape": "hv"}, name=best
            ),
            go.Scatter(
                x=[0, 1], y=[y.mean()] * 2, line={"color": MUTED, "dash": "dash"}, name="base rate"
            ),
        ]
    )
    st.plotly_chart(
        style(fig, xaxis_title="recall", yaxis_title="precision", yaxis_range=[0, 1.02]),
        width="stretch",
    )
with right, st.container(border=True):
    st.markdown("**Calibration (test, quantile bins)**")
    fig = go.Figure(
        go.Scatter(x=[0, 0.3], y=[0, 0.3], line={"color": MUTED, "dash": "dash"}, name="perfect")
    )
    for color, (k, r) in zip(SERIES, m["reliability_test"].items(), strict=False):
        fig.add_scatter(
            x=r["mean_predicted"],
            y=r["observed_rate"],
            mode="lines+markers",
            line={"color": color, "width": 2},
            name=k,
        )
    st.plotly_chart(
        style(fig, xaxis_title="mean predicted", yaxis_title="observed rate"), width="stretch"
    )

st.subheader("Inspection policy", anchor=False)
left, right = st.columns([1, 1])
with left, st.container(border=True):
    st.markdown("**Gains: inspect the riskiest x% → catch y% of failures**")
    g = gains_curve(y, p)
    fig = go.Figure(
        [
            go.Scatter(
                x=np.array(g["inspected_frac"]) * 100,
                y=np.array(g["caught_frac"]) * 100,
                line={"color": SERIES[0], "width": 2},
                name="model ranking",
            ),
            go.Scatter(
                x=[0, 100], y=[0, 100], line={"color": MUTED, "dash": "dash"}, name="random"
            ),
        ]
    )
    st.plotly_chart(
        style(fig, xaxis_title="% inspected", yaxis_title="% failures caught"), width="stretch"
    )
with right, st.container(border=True):
    st.markdown("**Cost sensitivity**")
    ratio = st.slider("Cost of a missed defect ÷ cost of an inspection", 2, 50, 10)
    r = evaluate_policy(y, p, 1.0, float(ratio))
    with st.container(horizontal=True):
        st.metric("Threshold p ≥", f"{r['threshold']:.3f}", border=True)
        st.metric("Runs inspected", pct(r["inspected_frac"]), border=True)
        st.metric("Failures caught", pct(r["caught_frac"], 0), border=True)
    costs = pd.DataFrame(
        {
            "policy": ["model", "inspect none", "inspect all", "random, same budget"],
            "expected cost per run": [
                r["cost_model"],
                r["cost_inspect_none"],
                r["cost_inspect_all"],
                r["cost_random_same_budget"],
            ],
        }
    )
    fig = go.Figure(
        go.Bar(
            x=costs["policy"],
            y=costs["expected cost per run"],
            marker_color=[SERIES[0], SERIES[1], SERIES[2], SERIES[3]],
        )
    )
    st.plotly_chart(style(fig, height=260, hovermode="closest"), width="stretch")
    st.caption(
        f"Savings vs the best naive policy: {r['savings_vs_best_naive']:+.3f} per run. "
        "Assumes inspections always find a present defect and costs are constant."
    )
