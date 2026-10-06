"""Copilot eval harness: planted-fault scenarios with known answers, prompt v1 vs v2."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from processlens.agent.graph import run_copilot
from processlens.agent.llm import LLMClient, make_client
from processlens.agent.tools import Workspace
from processlens.benchmark.run import Context, build_context
from processlens.benchmark.synth import Scenario, generate_labels, pick_causes
from processlens.config import PROJECT_ROOT, load_config
from processlens.data.contract import load_processed
from processlens.data.ingest import TIMESTAMP

log = logging.getLogger(__name__)
LOW_EFFECT = 0.5


def eval_scenarios(ctx: Context, cfg: dict[str, Any], bm: dict[str, Any]) -> list[Scenario]:
    """30 single-cause planted scenarios across mechanisms and effects, plus 10 nulls."""
    rng = np.random.default_rng(cfg["seed"] + 1)  # distinct from benchmark seeds
    combos = [(m, b) for m in bm["full"]["mechanisms"] for b in bm["full"]["effect_sizes"]]
    out: list[Scenario] = []
    while len(out) < cfg["eval"]["n_planted"]:
        mech, beta = combos[len(out) % len(combos)]
        seed = int(rng.integers(0, 2**31 - 1))
        causes = pick_causes(ctx.pool, ctx.clusters, 1, np.random.default_rng(seed))
        out.append(Scenario(mech, 1, float(beta), seed, None, causes))
    kinds = bm["null_kinds"]
    for i in range(cfg["eval"]["n_null"]):
        out.append(Scenario("null", 0, 0.0, int(rng.integers(0, 2**31 - 1)), kinds[i % 2]))
    return out


def _workspace(sc: Scenario, ctx: Context, ts: Any, bm: dict[str, Any], cache: Path) -> Workspace:
    path = cache / f"{sc.key}.pkl"
    if path.exists():
        with path.open("rb") as fh:
            ws: Workspace = pickle.load(fh)
        ws.log = []
        return ws
    y = generate_labels(ctx.x, sc, bm, ctx.real_labels)
    ws = Workspace.build(ctx.x, y, ts, ctx.clusters)
    with path.open("wb") as fh:
        pickle.dump(ws, fh)
    return ws


def score_run(
    sc: Scenario, state: dict[str, Any] | None, clusters: Any, error: str | None
) -> dict[str, Any]:
    """Per-scenario metrics for one prompt version."""
    row: dict[str, Any] = {
        "key": sc.key,
        "mechanism": sc.mechanism,
        "beta": sc.beta,
        "causes": list(sc.causes),
        "schema_valid": state is not None,
        "error": error,
    }
    if state is None:
        return row
    rep, ver = state["report"], state["verification"]
    found = [f.cluster for f in rep.findings[:3]]
    truth = {int(clusters[c]) for c in sc.causes}
    usage = state.get("usage", [])
    row.update(
        {
            "decision": rep.decision,
            "abstained": rep.decision == "insufficient_evidence",
            "hit@3": bool(truth & set(found)) if truth else None,
            "faithfulness": ver.faithfulness,
            "verified": ver.ok,
            "n_problems": len(ver.problems),
            "revisions": state.get("revisions", 0),
            "seconds": float(sum(u.get("seconds", 0.0) for u in usage)),
            "input_tokens": int(sum(u.get("input_tokens", 0) for u in usage)),
            "output_tokens": int(sum(u.get("output_tokens", 0) for u in usage)),
            "path": state.get("path", []),
        }
    )
    return row


def aggregate(rows: list[dict[str, Any]], pricing: dict[str, float]) -> dict[str, Any]:
    """Summary metrics for one prompt version."""
    ok = [r for r in rows if r["schema_valid"]]
    planted = [r for r in ok if r["mechanism"] != "null"]
    null = [r for r in ok if r["mechanism"] == "null"]
    low = [r for r in planted if r["beta"] <= LOW_EFFECT]
    secs = [r["seconds"] for r in ok]

    def mean(xs: list[Any]) -> float | None:
        return float(np.mean(xs)) if xs else None

    tokens_in = mean([r["input_tokens"] for r in ok])
    tokens_out = mean([r["output_tokens"] for r in ok])
    cost = (
        None
        if tokens_in is None or tokens_out is None
        else (tokens_in * pricing["input"] + tokens_out * pricing["output"]) / 1e6
    )
    return {
        "n": len(rows),
        "schema_valid_rate": len(ok) / len(rows) if rows else None,
        "hit@3_planted": mean([r["hit@3"] for r in planted]),
        "hit@3_planted_beta_ge_1": mean([r["hit@3"] for r in planted if r["beta"] >= 1.0]),
        "numeric_faithfulness": mean([r["faithfulness"] for r in ok]),
        "fully_verified_rate": mean([r["verified"] for r in ok]),
        "abstention_accuracy_null": mean([r["abstained"] for r in null]),
        "abstention_rate_low_effect": mean([r["abstained"] for r in low]),
        "false_abstention_rate_beta_ge_1": mean(
            [r["abstained"] for r in planted if r["beta"] >= 1.0]
        ),
        "mean_revisions": mean([r["revisions"] for r in ok]),
        "latency_p50_s": float(np.quantile(secs, 0.5)) if secs else None,
        "latency_p95_s": float(np.quantile(secs, 0.95)) if secs else None,
        "mean_input_tokens": tokens_in,
        "mean_output_tokens": tokens_out,
        "mean_cost_usd": cost,
    }


def run_eval(
    mode: str = "record", root: Path = PROJECT_ROOT, llm: LLMClient | None = None
) -> dict[str, Any]:
    """Run every scenario for every prompt version; write agent.json and sample reports."""
    cfg, bm, rc = load_config("agent"), load_config("benchmark"), load_config("rootcause")
    ctx = build_context(rc, bm, root)
    ts = load_processed(load_config("data"), root)[TIMESTAMP]
    scenarios = eval_scenarios(ctx, cfg, bm)
    (root / "evals").mkdir(exist_ok=True)
    (root / "evals" / "scenarios.jsonl").write_text(
        "\n".join(json.dumps(asdict(s)) for s in scenarios) + "\n", encoding="utf-8"
    )
    cache = root / ".cache" / "agent_ws"
    cache.mkdir(parents=True, exist_ok=True)
    Parallel(n_jobs=-1)(delayed(_workspace)(s, ctx, ts, bm, cache) for s in scenarios)
    llm = llm or make_client(cfg, mode, root)
    out_dir = root / "reports" / "agent_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"model": llm.model, "mode": mode, "versions": {}}
    for version in cfg["prompt_versions"]:
        rows = []
        for sc in scenarios:
            ws = _workspace(sc, ctx, ts, bm, cache)
            try:
                state, err = dict[str, Any](run_copilot(ws, llm, cfg, version)), None
            except Exception as exc:  # unparseable output, API error, missing recording
                state, err = None, f"{type(exc).__name__}: {exc}"
                log.warning("%s %s failed: %s", version, sc.key, err)
            rows.append(score_run(sc, state, ctx.clusters, err))
            if state is not None:
                (out_dir / f"{version}_{sc.key}.json").write_text(
                    json.dumps(
                        {
                            "scenario": asdict(sc),
                            "report": state["report"].model_dump(),
                            "verification": state["verification"].model_dump(),
                            "tool_log": ws.log_json(),
                            "path": state.get("path", []),
                        },
                        indent=1,
                        default=str,
                    ),
                    encoding="utf-8",
                )
        results["versions"][version] = {
            "summary": aggregate(rows, cfg["pricing_usd_per_million_tokens"]),
            "rows": rows,
        }
    results["n_scenarios"] = len(scenarios)
    path = root / "reports" / "metrics" / "agent.json"
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results
