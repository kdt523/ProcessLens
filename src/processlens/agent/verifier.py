"""Programmatic verifier: every number in the report must come from the tool-call log."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from processlens.agent.schemas import RootCauseReport, ToolCall

NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?")
SENSOR_OR_DATE = re.compile(r"\bsensor_\d+\b|\b\d{4}-\d{2}-\d{2}(?:[T ][\d:]+)?\b|\bcall_\d+\b")
CAUSAL = re.compile(
    r"\b(caused by|causes|causing|is the cause|root cause is|drives failures)\b", re.IGNORECASE
)
NEGATION = re.compile(r"\b(not|no|never|cannot)\b[^.]{0,30}$", re.IGNORECASE)


class Verification(BaseModel):
    """Verifier verdict, faithfulness score and specific feedback for revision."""

    ok: bool
    faithfulness: float
    n_numbers: int
    n_verified: int
    problems: list[str]


def numeric_leaves(obj: Any) -> list[float]:
    """All numbers in a nested structure, including numbers written inside strings."""
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, int | float):
        out.append(float(obj))
    elif isinstance(obj, str):
        out += [v for v, _ in parse_numbers(obj)]
    elif isinstance(obj, dict):
        for v in obj.values():
            out += numeric_leaves(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            out += numeric_leaves(v)
    return out


def parse_numbers(text: str) -> list[tuple[float, int]]:
    """Numbers in free text as (value, decimals); percentages become fractions.

    Sensor ids, ISO dates and tool-call ids are ignored.
    """
    clean = SENSOR_OR_DATE.sub(" ", text)
    out = []
    for m in NUMBER.finditer(clean):
        s = m.group()
        pct = s.endswith("%")
        s = s.rstrip("%")
        dec = len(s.split(".")[1]) if "." in s and "e" not in s.lower() else 0
        val = float(s)
        out.append((val / 100 if pct else val, dec + (2 if pct else 0)))
    return out


def _decimals(v: float) -> int:
    s = repr(float(v))
    return 0 if "e" in s or "." not in s else len(s.split(".")[1].rstrip("0"))


def matches(
    value: float, candidates: Iterable[float], decimals: int | None, rel_tol: float, abs_tol: float
) -> bool:
    """Return True if ``value`` matches a candidate within tolerance or as its rounding."""
    d = _decimals(value) if decimals is None else decimals
    for c in candidates:
        if abs(value - c) <= max(abs_tol, rel_tol * abs(c)):
            return True
        if abs(round(c, d) - value) < 10 ** -(d + 3) and d >= 1:
            return True
    return False


def _free_text(report: RootCauseReport) -> list[tuple[str, str]]:
    items = [("summary", report.summary)]
    items += [
        (f"finding[{i}].interpretation", f.interpretation) for i, f in enumerate(report.findings)
    ]
    items += [(f"recommended_checks[{i}]", c) for i, c in enumerate(report.recommended_checks)]
    items += [(f"limits[{i}]", c) for i, c in enumerate(report.limits)]
    return items


def _causal_claims(text: str) -> list[str]:
    hits = []
    for m in CAUSAL.finditer(text):
        if not NEGATION.search(text[: m.start()]):
            hits.append(m.group())
    return hits


def _engine_evidence(log: list[ToolCall]) -> dict[int, str]:
    grades: dict[int, str] = {}
    for c in log:
        if c.tool == "rank_suspects":
            for s in c.output["suspects"]:
                grades[int(s["cluster"])] = s["evidence"]
    return grades


def verify(report: RootCauseReport, log: list[ToolCall], cfg: dict[str, Any]) -> Verification:
    """Check numbers, citations, evidence grades, decision consistency and causal language."""
    rel, abs_ = cfg["verifier"]["rel_tolerance"], cfg["verifier"]["abs_tolerance"]
    calls = {c.id: c for c in log}
    all_values = numeric_leaves([c.output for c in log]) + numeric_leaves([c.args for c in log])
    all_values += [float(len(report.findings))]
    known_sensors = {s for c in log for s in re.findall(r"sensor_\d+", str(c.output))}
    grades = _engine_evidence(log)
    problems: list[str] = []
    n = ok = 0

    for i, f in enumerate(report.findings):
        if not f.supporting_numbers:
            problems.append(f"finding[{i}] cites no supporting numbers")
        if f.representative_sensor not in known_sensors:
            problems.append(f"finding[{i}] sensor {f.representative_sensor} not in any tool output")
        engine = grades.get(f.cluster)
        if engine is None:
            problems.append(f"finding[{i}] cluster {f.cluster} not returned by rank_suspects")
        elif engine != f.evidence_strength:
            problems.append(
                f"finding[{i}] says evidence '{f.evidence_strength}' but the engine "
                f"graded cluster {f.cluster} '{engine}'"
            )
        for j, sn in enumerate(f.supporting_numbers):
            n += 1
            call = calls.get(sn.tool_call_id)
            if call is None:
                problems.append(
                    f"finding[{i}].supporting_numbers[{j}] cites unknown tool call "
                    f"{sn.tool_call_id!r}"
                )
            elif matches(sn.value, numeric_leaves(call.output), None, rel, abs_):
                ok += 1
            else:
                problems.append(
                    f"finding[{i}].supporting_numbers[{j}] {sn.name}={sn.value} not "
                    f"found in output of {sn.tool_call_id}"
                )

    for where, text in _free_text(report):
        for val, dec in parse_numbers(text):
            n += 1
            if matches(val, all_values, dec, rel, abs_):
                ok += 1
            else:
                problems.append(f"{where}: number {val:g} does not appear in any tool output")
        for claim in _causal_claims(text):
            problems.append(f"{where}: causal language '{claim}'; say 'associated with'")

    if report.decision != "insufficient_evidence" and not report.findings:
        problems.append(f"decision '{report.decision}' but no findings")
    if report.decision == "investigate" and not any(
        f.evidence_strength in ("strong", "moderate") for f in report.findings
    ):
        problems.append("decision 'investigate' requires a strong or moderate finding")

    return Verification(
        ok=not problems,
        faithfulness=ok / n if n else 1.0,
        n_numbers=n,
        n_verified=ok,
        problems=problems,
    )
