"""Root-cause explorer: suspects, cluster members, pass vs fail distributions, onset."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import SERIES, require, style

rc = require("rootcause", "make rootcause")
w = rc["window"]
st.caption(
    f"Window {w['start'][:10]} → {w['end'][:10]}: {w['runs']} runs, {w['failures']} "
    f"failures; {rc['sensors_analysed']} sensors in {rc['n_clusters']} correlation "
    f"clusters. Evidence: {rc['evidence_counts']}."
)

sus = pd.DataFrame(rc["suspects"])
cols = [
    "consensus_rank",
    "representative",
    "n_members",
    "q_value",
    "effect_size",
    "stability_freq",
    "shap_share",
    "evidence",
]
st.dataframe(
    sus[cols],
    hide_index=True,
    width="stretch",
    column_config={
        "q_value": st.column_config.NumberColumn("q (BH)", format="%.2e"),
        "effect_size": st.column_config.NumberColumn("effect (Cliff's δ)", format="%.3f"),
        "stability_freq": st.column_config.ProgressColumn("stability", min_value=0, max_value=1),
        "shap_share": st.column_config.NumberColumn("SHAP share", format="%.3f"),
    },
)

detailed = [s for s in sus["representative"] if s in rc["temporal"]]
sensor = st.selectbox("Inspect a top suspect", detailed)
row = sus.set_index("representative").loc[sensor]
members = [s for s, c in rc["clusters"].items() if c == int(row["cluster"])]
st.markdown(
    f"**Cluster {int(row['cluster'])}** ({len(members)} sensors): "
    + ", ".join(f"`{s}`" for s in members)
    + ". Any member may be the relevant signal; the data cannot separate them."
)

left, right = st.columns(2)
with left, st.container(border=True):
    st.markdown("**Pass vs fail distribution** (share of each group)")
    d = rc["distributions"][sensor]
    mids = [(a + b) / 2 for a, b in zip(d["edges"][:-1], d["edges"][1:], strict=True)]
    fig = go.Figure()
    for name, color in (("pass", SERIES[0]), ("fail", SERIES[1])):
        total = max(sum(d[name]), 1)
        fig.add_bar(
            x=mids, y=[c / total for c in d[name]], name=name, marker_color=color, opacity=0.75
        )
    st.plotly_chart(
        style(fig, barmode="overlay", xaxis_title=sensor, yaxis_title="share", hovermode="closest"),
        width="stretch",
    )
with right, st.container(border=True):
    st.markdown("**Failure rate by sensor quartile, per month**")
    fr = pd.DataFrame(rc["temporal"][sensor]["fail_rate_by_bin"])
    fig = go.Figure()
    for color, (period, g) in zip(SERIES, fr.groupby("period"), strict=False):
        fig.add_scatter(
            x=g["bin"],
            y=g["fail_rate"] * 100,
            mode="lines+markers",
            name=period,
            line={"color": color, "width": 2},
            customdata=g["runs"],
            hovertemplate="Q%{x}: %{y:.1f}% (%{customdata} runs)",
        )
    st.plotly_chart(
        style(
            fig,
            xaxis_title="sensor quartile (1 = lowest)",
            yaxis_title="failure rate (%)",
            hovermode="closest",
        ),
        width="stretch",
    )

on = rc["temporal"][sensor]["onset"]
if on:
    st.info(
        f"Most likely change in the daily mean on **{on['onset']}**: "
        f"{on['mean_before']:.4g} → {on['mean_after']:.4g} "
        f"({on['shift_sd']:+.2f} SD). A change point is always found; small shifts mean "
        "no meaningful onset.",
        icon=":material/timeline:",
    )
