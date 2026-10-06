"""Request/response models for the ProcessLens API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Health(BaseModel):
    """Service status."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str | None
    reports_available: list[str]


class ScoreRequest(BaseModel):
    """Runs to score: one mapping of sensor id → value per run (missing sensors = NaN)."""

    runs: list[dict[str, float | None]] = Field(min_length=1, max_length=1000)
    top_n_contributors: int = Field(5, ge=1, le=20)


class Contributor(BaseModel):
    """One feature's SHAP contribution to a run's risk (model log-odds/probability units)."""

    feature: str
    sensor: str
    value: float | None
    contribution: float


class RunScore(BaseModel):
    """Calibrated failure probability plus top contributing sensors."""

    risk: float
    top_contributors: list[Contributor]


class ScoreResponse(BaseModel):
    """Scores for every submitted run."""

    model: str
    scores: list[RunScore]


class WindowRequest(BaseModel):
    """Analysis window (defaults to the training window)."""

    start: str | None = None
    end: str | None = None
    top_k: int = Field(10, ge=1, le=50)


class RootCauseResponse(BaseModel):
    """Ranked suspect clusters for the window."""

    note: str
    window: dict[str, Any]
    evidence_counts: dict[str, int]
    suspects: list[dict[str, Any]]


class ReportRequest(BaseModel):
    """Copilot report for a window."""

    start: str | None = None
    end: str | None = None
    prompt_version: Literal["v1", "v2"] = "v2"
    mode: Literal["replay", "record", "live"] = "replay"


class ReportResponse(BaseModel):
    """Verified copilot report."""

    report: dict[str, Any]
    verification: dict[str, Any]
    tool_log: list[dict[str, Any]]
    path: list[str]
