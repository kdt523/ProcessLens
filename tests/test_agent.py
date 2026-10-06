"""Copilot graph, tools and record/replay; fully offline with a scripted fake LLM."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from processlens.agent.graph import build_graph, load_prompt, run_copilot
from processlens.agent.llm import MissingRecordingError, RecordReplayClient
from processlens.agent.schemas import Finding, Plan, RootCauseReport, SupportingNumber
from processlens.agent.tools import Workspace, as_langchain_tools
from processlens.config import load_config

CAUSE = "sensor_003"


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    r = tmp_path_factory.mktemp("root")
    (r / "reports" / "metrics").mkdir(parents=True)
    (r / "docs").mkdir()
    (r / "reports/metrics/model.json").write_text(
        json.dumps(
            {
                "selected_model": "rf",
                "test": {
                    "metrics": {
                        "rf_calibrated": {
                            "pr_auc": {"value": 0.1, "ci_low": 0.05, "ci_high": 0.2},
                            "base_rate": {"value": 0.054},
                        }
                    }
                },
                "policy_test": {"budgets": [{"budget": 0.2, "caught_frac": 0.41}]},
            }
        )
    )
    (r / "docs/LIMITS.md").write_text(
        "# Limits\n\n1. **Correlated sensors cannot be separated.** cluster first in 48%.\n"
        "2. **False alarms with no real cause** strong in 3%.\n"
    )
    return r


def _workspace(root: Path, signal: float) -> Workspace:
    rng = np.random.default_rng(1)
    n = 500
    x = pd.DataFrame({f"sensor_{i:03d}": rng.normal(size=n) for i in range(1, 11)})
    p = 1 / (1 + np.exp(-(-3 + signal * x[CAUSE])))
    y = (rng.random(n) < p).astype(int)
    ts = pd.Series(pd.date_range("2008-07-19", periods=n, freq="4h"))
    cfg = load_config("rootcause")
    cfg["stability"]["n_subsamples"] = 20
    cfg["shap"]["lightgbm"]["n_estimators"] = 40
    return Workspace.build(x, y, ts, cfg=cfg, root=root)


@pytest.fixture(scope="module")
def strong_ws(root) -> Workspace:
    return _workspace(root, 2.0)


@pytest.fixture(scope="module")
def null_ws(root) -> Workspace:
    return _workspace(root, 0.0)


class ScriptedLLM:
    """Writes reports from the tool log; optionally starts with a wrong number."""

    model = "scripted"

    def __init__(self, ws: Workspace, first_wrong: bool = False) -> None:
        self.ws, self.first_wrong, self.calls = ws, first_wrong, []

    def structured(self, schema, system, user):
        self.calls.append(schema.__name__)
        usage = {"input_tokens": 10, "output_tokens": 5, "seconds": 0.01}
        suspects = next(c for c in self.ws.log if c.tool == "rank_suspects")
        top = suspects.output["suspects"][0]
        if schema is Plan:
            return Plan(
                sensors_to_inspect=[top["representative"], "sensor_bogus"], rationale="top"
            ), usage
        wrong = self.first_wrong and self.calls.count("RootCauseReport") == 1
        report = RootCauseReport(
            summary="One suspect cluster is associated with failures.",
            findings=[
                Finding(
                    cluster=top["cluster"],
                    representative_sensor=top["representative"],
                    evidence_strength=top["evidence"],
                    supporting_numbers=[
                        SupportingNumber(
                            name="effect_size",
                            value=9.99 if wrong else top["effect_size"],
                            tool_call_id=suspects.id,
                        )
                    ],
                    interpretation="Failing runs read higher; any cluster member may matter.",
                )
            ],
            recommended_checks=["Inspect the sensor against maintenance logs."],
            limits=[],
            decision="investigate" if top["evidence"] == "strong" else "monitor",
        )
        return report, usage


def test_tools_log_every_call_with_ids(strong_ws) -> None:
    strong_ws.log = []
    strong_ws.get_data_health()
    s = strong_ws.rank_suspects(3)
    strong_ws.inspect_sensor(s.suspects[0].representative)
    assert [c.id for c in strong_ws.log] == ["call_1", "call_2", "call_3"]
    assert s.suspects[0].representative == CAUSE
    with pytest.raises(ValueError):
        strong_ws.inspect_sensor("sensor_999")


def test_tool_outputs_contain_no_raw_rows(strong_ws) -> None:
    strong_ws.log = []
    detail = strong_ws.inspect_sensor(CAUSE)
    assert len(detail.fail_rate_by_quartile) == 4
    assert len(json.dumps(strong_ws.log_json())) < 2000  # summaries only


def test_langchain_tool_wrappers(strong_ws) -> None:
    tools = {t.name: t for t in as_langchain_tools(strong_ws)}
    assert set(tools) >= {"rank_suspects", "inspect_sensor", "when_did_it_start"}
    assert "sensor_id" in tools["inspect_sensor"].args
    out = tools["get_cluster"].invoke({"sensor_id": CAUSE})
    assert CAUSE in out["members"]


def test_graph_happy_path_verifies(strong_ws) -> None:
    strong_ws.log = []
    llm = ScriptedLLM(strong_ws)
    state = run_copilot(strong_ws, llm, load_config("agent"), "v2")
    assert state["verification"].ok, state["verification"].problems
    assert state["path"][:4] == ["plan", "gather", "draft", "verify"]
    assert "sensor_bogus" not in state["plan"].sensors_to_inspect  # filtered to real suspects
    assert state["report"].decision == "investigate"


def test_graph_revises_after_wrong_number(strong_ws) -> None:
    strong_ws.log = []
    llm = ScriptedLLM(strong_ws, first_wrong=True)
    state = run_copilot(strong_ws, llm, load_config("agent"), "v1")
    assert "revise" in state["path"]
    assert state["revisions"] == 1 and state["verification"].ok


def test_graph_abstains_on_noise(null_ws) -> None:
    null_ws.log = []
    llm = ScriptedLLM(null_ws)
    state = run_copilot(null_ws, llm, load_config("agent"), "v2")
    grades = {
        s["evidence"]
        for s in next(c for c in null_ws.log if c.tool == "rank_suspects").output["suspects"]
    }
    if grades & {"strong", "moderate"}:
        pytest.skip("noise produced a moderate suspect for this seed")
    assert state["abstained"] and state["report"].decision == "insufficient_evidence"
    assert state["verification"].ok, state["verification"].problems
    assert llm.calls == ["Plan"]  # no report drafted by the LLM


def test_graph_compiles_with_expected_nodes() -> None:
    nodes = set(build_graph().get_graph().nodes)
    assert {"plan", "gather", "draft", "verify", "revise", "abstain", "finalize"} <= nodes


def test_prompts_exist_and_differ() -> None:
    v1, v2 = load_prompt("v1"), load_prompt("v2")
    assert "insufficient_evidence" in v2 and "Never" in v2 and len(v2) > len(v1)


def test_record_then_replay(tmp_path, strong_ws) -> None:
    live = ScriptedLLM(strong_ws)
    strong_ws.log = []
    strong_ws.rank_suspects(3)
    rec = RecordReplayClient("record", tmp_path, live)
    plan, _ = rec.structured(Plan, "sys", "user")
    replay = RecordReplayClient("replay", tmp_path, model="scripted")
    again, usage = replay.structured(Plan, "sys", "user")
    assert again == plan and usage["replayed"]
    with pytest.raises(MissingRecordingError):
        replay.structured(Plan, "sys", "different prompt")
