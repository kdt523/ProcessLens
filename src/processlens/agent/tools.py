"""Copilot tools: typed, pure functions over analysis outputs. The LLM never sees raw rows.

A ``Workspace`` holds one analysis window (sensor matrix, labels, timestamps and the
root-cause engine result). Every tool call is appended to ``Workspace.log`` with an id
that report numbers must cite.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from processlens.agent.schemas import (
    ClusterInfo,
    DataHealth,
    KnownLimits,
    ModelPerformance,
    Onset,
    SensorDetail,
    Suspect,
    SuspectList,
    ToolCall,
)
from processlens.config import PROJECT_ROOT, load_config
from processlens.rootcause.engine import rank_suspects
from processlens.rootcause.temporal import onset as onset_fn


def _num(v: Any) -> float | None:
    """Convert to a finite float or None."""
    if v is None:
        return None
    f = float(v)
    return None if math.isnan(f) or math.isinf(f) else f


@dataclass
class Workspace:
    """One analysis window plus the engine result and a tool-call log."""

    x: pd.DataFrame
    y: np.ndarray
    ts: pd.Series
    result: dict[str, Any]
    cfg: dict[str, Any]
    root: Path = PROJECT_ROOT
    log: list[ToolCall] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        x: pd.DataFrame,
        y: np.ndarray,
        ts: pd.Series,
        clusters: pd.Series | None = None,
        cfg: dict[str, Any] | None = None,
        root: Path = PROJECT_ROOT,
    ) -> Workspace:
        """Run the root-cause engine on the window and wrap the result."""
        cfg = cfg or load_config("rootcause")
        res = rank_suspects(x, np.asarray(y), cfg, clusters)
        return cls(
            x.reset_index(drop=True), np.asarray(y), ts.reset_index(drop=True), res, cfg, root
        )

    # ------------------------------------------------------------ logging

    def _record(self, tool: str, args: dict[str, Any], out: BaseModel) -> BaseModel:
        self.log.append(
            ToolCall(
                id=f"call_{len(self.log) + 1}",
                tool=tool,
                args=args,
                output=json.loads(out.model_dump_json()),
            )
        )
        return out

    # ------------------------------------------------------------ tools

    def get_data_health(self) -> DataHealth:
        """Window size, failures, base rate and missing-cell share."""
        out = DataHealth(
            window_start=str(self.ts.iloc[0])[:10],
            window_end=str(self.ts.iloc[-1])[:10],
            runs=len(self.y),
            failures=int(self.y.sum()),
            base_rate=float(self.y.mean()),
            sensors_analysed=self.x.shape[1],
            missing_cell_share=float(self.x.isna().to_numpy().mean()),
        )
        return self._record("get_data_health", {}, out)  # type: ignore[return-value]

    def rank_suspects(self, top_k: int = 5) -> SuspectList:
        """Top-k consensus suspect clusters with evidence grades."""
        cons = self.result["consensus"]
        suspects = [
            Suspect(
                consensus_rank=int(r.consensus_rank),
                cluster=int(r.cluster),
                representative=str(r.representative),
                n_members=int(r.n_members),
                q_value=_num(r.q_value),
                effect_size=_num(r.effect_size),
                stability_freq=float(r.stability_freq),
                shap_share=float(r.shap_share),
                evidence=r.evidence,
            )
            for r in cons.head(top_k).itertuples()
        ]
        counts = {k: int(v) for k, v in cons["evidence"].value_counts().items()}
        out = SuspectList(suspects=suspects, evidence_counts=counts)
        return self._record("rank_suspects", {"top_k": top_k}, out)  # type: ignore[return-value]

    def _check_sensor(self, sensor_id: str) -> None:
        if sensor_id not in self.x.columns:
            raise ValueError(f"Unknown sensor {sensor_id!r}")

    def inspect_sensor(self, sensor_id: str) -> SensorDetail:
        """Statistics for one sensor: q, effect, stability, SHAP, medians, fail rate by quartile."""
        self._check_sensor(sensor_id)
        row = self.result["sensors"].loc[sensor_id]
        v = self.x[sensor_id]
        ok = v.notna().to_numpy()
        quart = pd.qcut(v[ok].rank(method="first"), 4, labels=False)
        rates = pd.Series(self.y[ok]).groupby(quart.to_numpy()).mean()
        fail = self.y == 1
        out = SensorDetail(
            sensor=sensor_id,
            cluster=int(self.result["clusters"][sensor_id]),
            q_value=_num(row["q"]),
            effect_size=_num(row["effect"]),
            stability_freq=float(row["stability"]),
            shap_share=float(row["shap_share"]),
            missing_share=float(1 - ok.mean()),
            median_pass=_num(v[ok & ~fail].median()),
            median_fail=_num(v[ok & fail].median()),
            fail_rate_by_quartile=[float(r) for r in rates.to_numpy()],
        )
        return self._record("inspect_sensor", {"sensor_id": sensor_id}, out)  # type: ignore[return-value]

    def get_cluster(self, sensor_id: str) -> ClusterInfo:
        """Members of the sensor's correlation cluster."""
        self._check_sensor(sensor_id)
        cl = self.result["clusters"]
        cid = int(cl[sensor_id])
        out = ClusterInfo(sensor=sensor_id, cluster=cid, members=list(cl[cl == cid].index))
        return self._record("get_cluster", {"sensor_id": sensor_id}, out)  # type: ignore[return-value]

    def when_did_it_start(self, sensor_id: str) -> Onset:
        """Most likely change point in the sensor's daily mean."""
        self._check_sensor(sensor_id)
        t = self.cfg["temporal"]
        res = onset_fn(self.x[sensor_id], self.ts, t["onset_freq"], t["onset_min_size"]) or {}
        out = Onset(
            sensor=sensor_id,
            onset=res.get("onset"),
            mean_before=_num(res.get("mean_before")),
            mean_after=_num(res.get("mean_after")),
            shift_sd=_num(res.get("shift_sd")),
        )
        return self._record("when_did_it_start", {"sensor_id": sensor_id}, out)  # type: ignore[return-value]

    def get_model_performance(self) -> ModelPerformance:
        """Test-window performance of the Phase 2 defect model."""
        m = json.loads((self.root / "reports/metrics/model.json").read_text(encoding="utf-8"))
        best = m["selected_model"]
        t = m["test"]["metrics"][f"{best}_calibrated"]
        budget = {b["budget"]: b["caught_frac"] for b in m["policy_test"]["budgets"]}
        out = ModelPerformance(
            model=best,
            pr_auc=t["pr_auc"]["value"],
            pr_auc_ci_low=t["pr_auc"]["ci_low"],
            pr_auc_ci_high=t["pr_auc"]["ci_high"],
            base_rate=t["base_rate"]["value"],
            caught_share_at_20pct_budget=budget.get(0.2),
        )
        return self._record("get_model_performance", {}, out)  # type: ignore[return-value]

    def get_known_limits(self) -> KnownLimits:
        """Numbered limits from docs/LIMITS.md (generated by the benchmark)."""
        path = self.root / "docs" / "LIMITS.md"
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        limits = [ln.split(". ", 1)[1] for ln in lines if ln[:1].isdigit() and ". " in ln]
        return self._record("get_known_limits", {}, KnownLimits(limits=limits))  # type: ignore[return-value]

    def log_json(self) -> list[dict[str, Any]]:
        """Return the tool-call log as plain dicts (what the LLM is shown)."""
        return [c.model_dump() for c in self.log]


def _plain(fn: Callable[..., Any]) -> Callable[..., dict[str, Any]]:
    @functools.wraps(fn)
    def wrapper(**kwargs: Any) -> dict[str, Any]:
        return fn(**kwargs).model_dump()

    return wrapper


def as_langchain_tools(ws: Workspace) -> list[StructuredTool]:
    """Expose the workspace tools as LangChain ``StructuredTool``s (args typed from signatures)."""
    fns: list[Callable[..., Any]] = [
        ws.get_data_health,
        ws.rank_suspects,
        ws.inspect_sensor,
        ws.get_cluster,
        ws.when_did_it_start,
        ws.get_model_performance,
        ws.get_known_limits,
    ]
    return [StructuredTool.from_function(func=_plain(f)) for f in fns]
