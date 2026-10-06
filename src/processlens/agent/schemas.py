"""Typed tool outputs and the structured root-cause report."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Evidence = Literal["strong", "moderate", "weak"]
Decision = Literal["investigate", "monitor", "insufficient_evidence"]


# ---------------------------------------------------------------- tool outputs


class DataHealth(BaseModel):
    """Window summary: size, failures and data quality."""

    window_start: str
    window_end: str
    runs: int
    failures: int
    base_rate: float
    sensors_analysed: int
    missing_cell_share: float


class Suspect(BaseModel):
    """One ranked suspect cluster from the consensus engine."""

    consensus_rank: int
    cluster: int
    representative: str
    n_members: int
    q_value: float | None
    effect_size: float | None
    stability_freq: float
    shap_share: float
    evidence: Evidence


class SuspectList(BaseModel):
    """Top-k consensus suspects plus how many clusters were graded at each level."""

    suspects: list[Suspect]
    evidence_counts: dict[str, int]


class SensorDetail(BaseModel):
    """Per-sensor statistics for one suspect."""

    sensor: str
    cluster: int
    q_value: float | None
    effect_size: float | None
    stability_freq: float
    shap_share: float
    missing_share: float
    median_pass: float | None
    median_fail: float | None
    fail_rate_by_quartile: list[float]


class ClusterInfo(BaseModel):
    """Members of a sensor's correlation cluster."""

    sensor: str
    cluster: int
    members: list[str]


class Onset(BaseModel):
    """Most likely change point in the sensor's daily mean."""

    sensor: str
    onset: str | None
    mean_before: float | None
    mean_after: float | None
    shift_sd: float | None


class ModelPerformance(BaseModel):
    """Phase 2 test-window performance of the defect model."""

    model: str
    pr_auc: float
    pr_auc_ci_low: float
    pr_auc_ci_high: float
    base_rate: float
    caught_share_at_20pct_budget: float | None


class KnownLimits(BaseModel):
    """Plain-English limits from the planted-fault benchmark (docs/LIMITS.md)."""

    limits: list[str]


class ToolCall(BaseModel):
    """One logged tool call; ``id`` is what report numbers must cite."""

    id: str
    tool: str
    args: dict[str, Any]
    output: dict[str, Any]


# ---------------------------------------------------------------- LLM outputs


class Plan(BaseModel):
    """Which suspects to inspect in detail."""

    sensors_to_inspect: list[str] = Field(
        description="Representative sensor ids from rank_suspects to inspect (at most top_k)."
    )
    rationale: str


class SupportingNumber(BaseModel):
    """A number quoted in the report, with the tool call that produced it."""

    name: str = Field(description="What the number is, e.g. 'q_value' or 'effect_size'.")
    value: float = Field(description="Copied exactly from the cited tool output.")
    tool_call_id: str = Field(description="id of the tool call whose output contains value.")


class Finding(BaseModel):
    """One suspect cluster in the report."""

    cluster: int
    representative_sensor: str
    evidence_strength: Evidence
    supporting_numbers: list[SupportingNumber]
    onset_date: str | None = None
    interpretation: str = Field(description="Plain-English meaning; never claim causation.")


class RootCauseReport(BaseModel):
    """Engineer-facing root-cause report written only from tool outputs."""

    summary: str
    findings: list[Finding]
    recommended_checks: list[str]
    limits: list[str]
    decision: Decision
