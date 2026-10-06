"""Copilot: verified root-cause reports, each claim shown with its evidence."""

import json
import os
import urllib.error
import urllib.request

import streamlit as st
from common import ROOT, load_json, pct

from processlens.agent.verifier import matches, numeric_leaves

REPORT_DIR = ROOT / "reports" / "agent_reports"
API = os.environ.get("PROCESSLENS_API", "http://127.0.0.1:8000")
TOL = (0.01, 0.001)


def render(payload: dict) -> None:
    """Show a report with a verified badge on every supporting number."""
    rep, ver = payload["report"], payload["verification"]
    calls = {c["id"]: c for c in payload["tool_log"]}
    with st.container(horizontal=True):
        st.metric("Decision", rep["decision"].replace("_", " "), border=True)
        st.metric("Numeric faithfulness", pct(ver["faithfulness"], 0), border=True)
        st.metric("Verifier", "passed" if ver["ok"] else "issues", border=True)
    st.markdown(rep["summary"])
    for f in rep["findings"]:
        with st.container(border=True):
            st.markdown(
                f"**Cluster {f['cluster']}** · `{f['representative_sensor']}` · "
                f"evidence **{f['evidence_strength']}**"
                + (f" · onset {f['onset_date']}" if f.get("onset_date") else "")
            )
            st.markdown(f["interpretation"])
            for n in f["supporting_numbers"]:
                call = calls.get(n["tool_call_id"])
                ok = call is not None and matches(
                    n["value"], numeric_leaves(call["output"]), None, *TOL
                )
                badge = (
                    ":green-badge[:material/check: verified]"
                    if ok
                    else ":red-badge[:material/close: unverified]"
                )
                tool = call["tool"] if call else "unknown call"
                st.markdown(
                    f"{badge} `{n['name']}` = {n['value']:g} · from `{n['tool_call_id']}` ({tool})"
                )
    if rep["recommended_checks"]:
        st.markdown(
            "**Recommended checks**\n" + "\n".join(f"- {c}" for c in rep["recommended_checks"])
        )
    if rep["limits"]:
        st.markdown("**Limits**\n" + "\n".join(f"- {c}" for c in rep["limits"]))
    if ver["problems"]:
        with st.expander(f"Verifier problems ({len(ver['problems'])})"):
            st.markdown("\n".join(f"- {p}" for p in ver["problems"]))
    with st.expander("Tool-call log (everything the LLM saw)"):
        st.json(payload["tool_log"], expanded=False)


agent = load_json("agent")
if agent:
    rows = [{"prompt": v, **d["summary"]} for v, d in agent["versions"].items()]
    st.caption(f"Eval on {agent['n_scenarios']} planted-fault scenarios, model `{agent['model']}`.")
    st.dataframe(rows, hide_index=True, width="stretch")
else:
    st.info(
        "No live copilot evaluation yet. Add `GEMINI_API_KEY` to `.env` and run `make agent-eval`.",
        icon=":material/info:",
    )

tab_saved, tab_live = st.tabs(["Saved eval reports", "Generate for a window (API)"])
with tab_saved:
    files = sorted(REPORT_DIR.glob("*.json")) if REPORT_DIR.exists() else []
    if not files:
        st.write("No saved reports yet.")
    else:
        choice = st.selectbox("Report", files, format_func=lambda p: p.stem)
        payload = json.loads(choice.read_text(encoding="utf-8"))
        sc = payload["scenario"]
        truth = ", ".join(sc["causes"]) or "none (null scenario)"
        st.caption(f"Scenario: {sc['mechanism']}, β = {sc['beta']}, planted: {truth}")
        render(payload)
with tab_live:
    st.caption(f"Calls `{API}/report`. Replay mode needs a stored recording; live needs a key.")
    c1, c2, c3 = st.columns(3)
    start = c1.text_input("Start", "")
    end = c2.text_input("End", "")
    mode = c3.selectbox("Mode", ["replay", "live", "record"])
    if st.button("Generate report", icon=":material/play_arrow:"):
        body = json.dumps({"start": start or None, "end": end or None, "mode": mode}).encode()
        req = urllib.request.Request(
            f"{API}/report", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310 (local API)
                render(json.loads(resp.read()))
        except urllib.error.HTTPError as exc:
            st.error(f"{exc.code}: {json.loads(exc.read()).get('detail', exc.reason)}")
        except urllib.error.URLError as exc:
            st.error(f"API not reachable ({exc.reason}). Start it with `make api`.")
