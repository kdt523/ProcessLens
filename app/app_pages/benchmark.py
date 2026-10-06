"""Benchmark: how much to trust a ranking (hit@k vs effect, MDE, null false alarms)."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import MUTED, SERIES, load_text, pct, require, style

bm = require("benchmark", "make benchmark")
nf = bm["null_false_alarms"]
g = bm["gates"]

with st.container(horizontal=True):
    st.metric("Planted scenarios", bm["n_planted"], border=True)
    st.metric("Null scenarios", bm["n_null"], border=True)
    st.metric("Null: any 'strong' suspect", pct(nf["any_strong"], 0), border=True)
    st.metric("Null: any q < 0.05", pct(nf["any_q_below_alpha"], 0), border=True)
    st.metric("Quality gate", "passed" if g["passed"] else "failed", border=True)

level = st.segmented_control("Level", ["cluster", "sensor"], default="cluster")
k = st.segmented_control("k", [1, 3, 5], default=5)
curve = pd.DataFrame(bm["by_beta"][f"{level}_hit@{k}"])
cols = st.columns(3)
for col, mech in zip(cols, curve["mechanism"].unique(), strict=False):
    with col, st.container(border=True):
        st.markdown(f"**{mech}**")
        fig = go.Figure()
        for color, (method, d) in zip(
            SERIES, curve[curve["mechanism"] == mech].groupby("method", sort=False), strict=False
        ):
            fig.add_scatter(
                x=d["beta"],
                y=d["mean"] * 100,
                mode="lines+markers",
                name=method,
                line={"color": color, "width": 3 if method == "consensus" else 1.5},
                error_y={"type": "data", "array": d["ci"] * 100, "visible": True, "thickness": 1},
            )
        fig.add_hline(y=80, line={"color": MUTED, "dash": "dash"})
        st.plotly_chart(
            style(
                fig,
                height=320,
                xaxis_title="planted β (log-odds per SD)",
                yaxis_title=f"hit@{k} (%)",
                yaxis_range=[0, 105],
            ),
            width="stretch",
        )

left, right = st.columns(2)
with left, st.container(border=True):
    st.markdown(
        f"**Minimum detectable effect** ({bm['mde']['metric']} ≥ {bm['mde']['target']:.0%})"
    )
    mde = pd.DataFrame(bm["mde"]["values"]).T
    st.dataframe(
        mde.map(lambda v: "not reached" if v is None or pd.isna(v) else f"β = {v:g}"),
        width="stretch",
    )
with right, st.container(border=True):
    st.markdown("**False alarms on null scenarios**")
    rows = [
        {
            "null labels": kind,
            "any q < 0.05": pct(v["any_q_below_alpha"], 0),
            "any ≥ moderate": pct(v["any_moderate_or_strong"], 0),
            "any strong": pct(v["any_strong"], 0),
            "n": v["n"],
        }
        for kind, v in nf["by_kind"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with st.expander("Known limits (generated from this benchmark)", icon=":material/rule:"):
    st.markdown(load_text("docs/LIMITS.md") or "Run `make benchmark`.")
