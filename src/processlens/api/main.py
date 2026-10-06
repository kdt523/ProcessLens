"""ProcessLens FastAPI service: scoring, root-cause ranking, copilot reports, drift."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pandera.errors as pe
from fastapi import FastAPI, HTTPException

from processlens.api.schemas import (
    Contributor,
    Health,
    ReportRequest,
    ReportResponse,
    RootCauseResponse,
    RunScore,
    ScoreRequest,
    ScoreResponse,
    WindowRequest,
)
from processlens.config import PROJECT_ROOT
from processlens.data.contract import sensor_schema

app = FastAPI(
    title="ProcessLens",
    version="0.1.0",
    description="Suspect process parameters associated with defects (not causes).",
)
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
REPORTS = ["model", "rootcause", "benchmark", "drift", "agent"]


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the calibrated model bundle written by ``processlens train``."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `make train`")
    bundle: dict[str, Any] = joblib.load(path)
    return bundle


def _sensor_of(feature: str) -> str:
    return feature.removeprefix("missingindicator_")


def contributions(pipeline: Any, x: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Per-feature contributions to each run's score (TreeSHAP for trees, coef·x for linear)."""
    pre, model = pipeline[:-1], pipeline[-1]
    z = pre.transform(x)
    names = [str(n) for n in pre.get_feature_names_out()]
    if hasattr(model, "coef_"):
        return np.asarray(z) * model.coef_.ravel(), names
    import shap

    est = getattr(model, "model_", model)  # unwrap EarlyStoppedLGBM
    vals = shap.TreeExplainer(est).shap_values(np.asarray(z))
    vals = np.asarray(vals[1] if isinstance(vals, list) else vals)
    return (vals[..., 1] if vals.ndim == 3 else vals), names


@app.get("/health", response_model=Health)
def health() -> Health:
    """Report whether the model and analysis reports are available."""
    try:
        name: str | None = load_model()["name"]
    except FileNotFoundError:
        name = None
    avail = [r for r in REPORTS if (PROJECT_ROOT / f"reports/metrics/{r}.json").exists()]
    return Health(
        status="ok" if name else "degraded",
        model_loaded=name is not None,
        model_name=name,
        reports_available=avail,
    )


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    """Calibrated failure risk per run plus its top contributing sensors (SHAP)."""
    try:
        bundle = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    features: list[str] = bundle["features"]
    raw = pd.DataFrame(req.runs).astype("float64")
    try:
        sensor_schema(features).validate(raw, lazy=True)
    except pe.SchemaErrors as exc:
        raise HTTPException(422, f"Input violates the data contract: {exc.failure_cases}") from exc
    x = raw.reindex(columns=features)
    risk = bundle["model"].predict_proba(x)[:, 1]
    contrib, names = contributions(bundle["pipeline"], x)
    scores = []
    for i in range(len(x)):
        top = np.argsort(-np.abs(contrib[i]))[: req.top_n_contributors]
        scores.append(
            RunScore(
                risk=float(risk[i]),
                top_contributors=[
                    Contributor(
                        feature=names[j],
                        sensor=_sensor_of(names[j]),
                        value=None
                        if pd.isna(x.iloc[i][_sensor_of(names[j])])
                        else float(x.iloc[i][_sensor_of(names[j])]),
                        contribution=float(contrib[i, j]),
                    )
                    for j in top
                ],
            )
        )
    return ScoreResponse(model=bundle["name"], scores=scores)


@app.post("/rootcause", response_model=RootCauseResponse)
def rootcause(req: WindowRequest) -> RootCauseResponse:
    """Ranked suspect sensor clusters for a time window."""
    from processlens.rootcause.engine import run_rootcause

    try:
        res = run_rootcause(req.start, req.end, req.top_k, write=False)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return RootCauseResponse(
        note=res["note"],
        window=res["window"],
        evidence_counts=res["evidence_counts"],
        suspects=res["suspects"],
    )


@app.post("/report", response_model=ReportResponse)
def report(req: ReportRequest) -> ReportResponse:
    """Copilot report for a window (replay mode needs a stored recording; live needs a key)."""
    from processlens.agent.graph import run_copilot
    from processlens.agent.llm import MissingRecordingError, make_client
    from processlens.agent.tools import Workspace
    from processlens.config import load_config
    from processlens.data.contract import load_processed, sensor_columns
    from processlens.data.ingest import LABEL, TIMESTAMP
    from processlens.rootcause.engine import resolve_window, usable_sensors

    df = load_processed()
    lo, hi = resolve_window(df, req.start, req.end)
    w = df[(df[TIMESTAMP] >= lo) & (df[TIMESTAMP] <= hi)].reset_index(drop=True)
    x_all = w[sensor_columns(w)]
    x = x_all[usable_sensors(x_all, load_config("rootcause")["max_missing_frac"])]
    cfg = load_config("agent")
    try:
        llm = make_client(cfg, req.mode)
        ws = Workspace.build(x, w[LABEL].to_numpy(), w[TIMESTAMP])
        state = run_copilot(ws, llm, cfg, req.prompt_version)
    except (MissingRecordingError, RuntimeError) as exc:
        raise HTTPException(503, str(exc)) from exc
    return ReportResponse(
        report=state["report"].model_dump(),
        verification=state["verification"].model_dump(),
        tool_log=ws.log_json(),
        path=state.get("path", []),
    )


@app.get("/drift")
def drift(start: str | None = None, end: str | None = None) -> dict[str, Any]:
    """PSI drift between the training window and ``[start, end]`` (default: test window)."""
    from processlens.monitoring.drift import run_drift

    return run_drift(start, end, write=False)
