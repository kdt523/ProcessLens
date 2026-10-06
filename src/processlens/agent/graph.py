"""LangGraph copilot: plan → gather → draft → verify → (revise ≤ N) → finalize, or abstain.

The LLM only (a) picks which suspects to inspect and (b) writes/revises the report from the
tool-call log. Tools run in code; the abstain branch is decided by the engine's evidence
grades, not by the LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from processlens.agent.llm import LLMClient
from processlens.agent.schemas import Plan, RootCauseReport
from processlens.agent.tools import Workspace
from processlens.agent.verifier import Verification, verify

PROMPT_DIR = Path(__file__).parent / "prompts"
ACTIONABLE = {"strong", "moderate"}


class CopilotState(TypedDict, total=False):
    """Graph state."""

    ws: Workspace
    llm: LLMClient
    cfg: dict[str, Any]
    prompt: str
    plan: Plan
    report: RootCauseReport
    verification: Verification
    revisions: int
    usage: list[dict[str, Any]]
    path: list[str]
    abstained: bool


def load_prompt(version: str) -> str:
    """Return the system prompt text for ``version`` (``v1`` or ``v2``)."""
    return (PROMPT_DIR / f"{version}.md").read_text(encoding="utf-8")


def _log_payload(ws: Workspace) -> str:
    return json.dumps(ws.log_json(), indent=1, default=str)


def _suspects(ws: Workspace) -> list[dict[str, Any]]:
    calls = [c for c in ws.log if c.tool == "rank_suspects"]
    return calls[-1].output["suspects"] if calls else []


def plan_node(state: CopilotState) -> CopilotState:
    """Run the overview tools, then let the LLM choose which suspects to inspect."""
    ws, cfg = state["ws"], state["cfg"]
    ws.get_data_health()
    ws.rank_suspects(cfg["graph"]["top_k_suspects"])
    ws.get_model_performance()
    ws.get_known_limits()
    allowed = [s["representative"] for s in _suspects(ws)]
    user = (
        "Tool log so far:\n" + _log_payload(ws) + "\n\nChoose which representative "
        f"sensors to inspect in detail (only from: {allowed})."
    )
    plan, usage = state["llm"].structured(Plan, state["prompt"], user)
    chosen = [s for s in plan.sensors_to_inspect if s in allowed][: len(allowed)] or allowed[:3]
    plan = Plan(sensors_to_inspect=chosen, rationale=plan.rationale)
    return {
        "plan": plan,
        "usage": [*state.get("usage", []), usage],
        "path": [*state.get("path", []), "plan"],
    }


def route_after_plan(state: CopilotState) -> str:
    """Abstain when no suspect meets the evidence rules."""
    grades = {s["evidence"] for s in _suspects(state["ws"])}
    return "gather" if grades & ACTIONABLE else "abstain"


def gather_node(state: CopilotState) -> CopilotState:
    """Detail tools for each chosen suspect."""
    ws = state["ws"]
    for s in state["plan"].sensors_to_inspect:
        ws.inspect_sensor(s)
        ws.get_cluster(s)
        ws.when_did_it_start(s)
    return {"path": [*state["path"], "gather"]}


def draft_node(state: CopilotState) -> CopilotState:
    """Write the structured report from the tool log."""
    user = "Tool log:\n" + _log_payload(state["ws"]) + "\n\nWrite the RootCauseReport."
    report, usage = state["llm"].structured(RootCauseReport, state["prompt"], user)
    return {
        "report": report,
        "revisions": 0,
        "usage": [*state["usage"], usage],
        "path": [*state["path"], "draft"],
    }


def verify_node(state: CopilotState) -> CopilotState:
    """Check every number and claim against the log."""
    v = verify(state["report"], state["ws"].log, state["cfg"])
    return {"verification": v, "path": [*state["path"], "verify"]}


def route_after_verify(state: CopilotState) -> str:
    """Revise on failure until the revision budget is spent."""
    if state["verification"].ok:
        return "finalize"
    return "revise" if state["revisions"] < state["cfg"]["graph"]["max_revisions"] else "finalize"


def revise_node(state: CopilotState) -> CopilotState:
    """Ask the LLM to fix exactly the problems the verifier found."""
    feedback = "\n".join(f"- {p}" for p in state["verification"].problems)
    user = (
        "Tool log:\n"
        + _log_payload(state["ws"])
        + "\n\nYour previous report:\n"
        + state["report"].model_dump_json(indent=1)
        + "\n\nThe verifier rejected it for these reasons:\n"
        + feedback
        + "\n\nReturn a corrected RootCauseReport. Remove any number you cannot copy "
        "exactly from a tool output."
    )
    report, usage = state["llm"].structured(RootCauseReport, state["prompt"], user)
    return {
        "report": report,
        "revisions": state["revisions"] + 1,
        "usage": [*state["usage"], usage],
        "path": [*state["path"], "revise"],
    }


def abstain_node(state: CopilotState) -> CopilotState:
    """Deterministic insufficient-evidence report built from tool outputs only."""
    ws = state["ws"]
    limits = next(c.output["limits"] for c in ws.log if c.tool == "get_known_limits")
    counts = next(c.output["evidence_counts"] for c in ws.log if c.tool == "rank_suspects")
    report = RootCauseReport(
        summary="No process parameter is singled out in this window: every suspect cluster "
        "has weak evidence under the evidence rules, so the ranking may reflect noise.",
        findings=[],
        recommended_checks=[
            "Collect more failing runs or widen the analysis window before acting.",
            "Review process logs for events not captured by the sensors.",
        ],
        limits=[ln for ln in limits if "False alarms" in ln or "Correlated" in ln][:2],
        decision="insufficient_evidence",
    )
    v = verify(report, ws.log, state["cfg"])
    return {
        "report": report,
        "verification": v,
        "abstained": True,
        "path": [*state["path"], f"abstain(weak={counts.get('weak', 0)})"],
    }


def finalize_node(state: CopilotState) -> CopilotState:
    """Mark the end of the run."""
    return {"abstained": state.get("abstained", False), "path": [*state["path"], "finalize"]}


def build_graph() -> Any:
    """Compile the copilot graph."""
    g = StateGraph(CopilotState)
    for name, fn in [
        ("plan", plan_node),
        ("gather", gather_node),
        ("draft", draft_node),
        ("verify", verify_node),
        ("revise", revise_node),
        ("abstain", abstain_node),
        ("finalize", finalize_node),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", route_after_plan, ["gather", "abstain"])
    g.add_edge("gather", "draft")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", route_after_verify, ["revise", "finalize"])
    g.add_edge("revise", "verify")
    g.add_edge("abstain", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def run_copilot(
    ws: Workspace, llm: LLMClient, cfg: dict[str, Any], prompt_version: str = "v2"
) -> CopilotState:
    """Run the graph once and return the final state."""
    out: CopilotState = build_graph().invoke(
        {
            "ws": ws,
            "llm": llm,
            "cfg": cfg,
            "prompt": load_prompt(prompt_version),
            "usage": [],
            "path": [],
            "revisions": 0,
        }
    )
    return out
