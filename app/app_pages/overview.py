"""Overview: data health, failure-rate trend, drift alerts."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import SERIES, load_json, pct, require, style

audit = require("audit", "make ingest audit")
lab, shape, miss = audit["label"], audit["shape"], audit["missing"]

with st.container(horizontal=True):
    st.metric("Production runs", f"{shape['rows']:,}", border=True)
    st.metric("Sensors", shape["sensors"], border=True)
    st.metric("Failures", lab["failures"], border=True)
    st.metric("Base failure rate", pct(lab["base_rate"], 2), border=True)
    st.metric("Missing sensor cells", pct(miss["total_cell_frac"], 2), border=True)

weekly = pd.DataFrame(audit["time"]["weekly"])
weekly["fail_rate"] = weekly["failures"] / weekly["runs"]
left, right = st.columns(2)
with left, st.container(border=True):
    st.markdown("**Weekly failure rate**")
    fig = go.Figure(
        go.Scatter(
            x=weekly["week"],
            y=weekly["fail_rate"] * 100,
            mode="lines+markers",
            line={"color": SERIES[0], "width": 2},
            customdata=weekly[["runs", "failures"]],
            hovertemplate="%{y:.1f}% (%{customdata[1]} of %{customdata[0]} runs)<extra></extra>",
        )
    )
    st.plotly_chart(style(fig, yaxis_title="failure rate (%)"), width="stretch")
with right, st.container(border=True):
    st.markdown("**Runs per week**")
    fig = go.Figure(go.Bar(x=weekly["week"], y=weekly["runs"], marker_color=SERIES[0]))
    st.plotly_chart(style(fig, yaxis_title="runs"), width="stretch")

segs = audit["time"]["fail_rate_by_split_segment"]
st.caption(
    "Failure rate in the time-ordered train / validation / test windows: "
    + " / ".join(pct(s["fail_rate"]) for s in segs)
    + ". The process is not stationary, which is why every split is time-ordered."
)

st.subheader("Drift alerts", anchor=False)
drift = load_json("drift")
if drift is None:
    st.info("Run `make drift` to compute PSI drift.", icon=":material/info:")
else:
    c = drift["counts"]
    st.caption(
        f"PSI per sensor: training window ({drift['reference']['start'][:10]} → "
        f"{drift['reference']['end'][:10]}) vs current window "
        f"({drift['current']['start'][:10]} → {drift['current']['end'][:10]}). "
        f"Warn ≥ {drift['thresholds']['warn']}, alert ≥ {drift['thresholds']['alert']} "
        f"or a missing-rate jump ≥ {drift['thresholds']['missing_rate_delta_alert']:.0%}."
    )
    with st.container(horizontal=True):
        st.metric("Sensors OK", c["ok"], border=True)
        st.metric("Warn", c["warn"], border=True)
        st.metric("Alert", c["alert"], border=True)
    st.dataframe(
        pd.DataFrame(drift["top"]),
        hide_index=True,
        width="stretch",
        column_config={
            "psi": st.column_config.NumberColumn(format="%.3f"),
            "missing_rate_delta": st.column_config.NumberColumn(
                "missing-rate change", format="%.3f"
            ),
        },
    )
