"""Population Stability Index (PSI) and missing-rate drift per sensor.

Reference window = the Phase 2 training window. Current window = the test window by
default, or any analyst-chosen window.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from processlens.config import PROJECT_ROOT, load_config
from processlens.data.contract import load_processed, sensor_columns
from processlens.data.ingest import TIMESTAMP
from processlens.data.split import time_split
from processlens.rootcause.engine import usable_sensors

EPS = 1e-4


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI = Σ (c − r)·ln(c / r) over reference-quantile bins (missing values ignored)."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.all(cur == ref[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def level(value: float, warn: float, alert: float) -> str:
    """Map a PSI value to ``ok`` / ``warn`` / ``alert``."""
    if np.isnan(value):
        return "ok"
    return "alert" if value >= alert else "warn" if value >= warn else "ok"


def drift_table(ref: pd.DataFrame, cur: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """PSI, missing-rate change and alert level for every sensor."""
    rows = []
    for c in ref.columns:
        value = psi(ref[c].to_numpy(float), cur[c].to_numpy(float), cfg["psi_bins"])
        dmiss = float(cur[c].isna().mean() - ref[c].isna().mean())
        lvl = level(value, cfg["psi_warn"], cfg["psi_alert"])
        if abs(dmiss) >= cfg["missing_rate_delta_alert"]:
            lvl = "alert"
        rows.append({"sensor": c, "psi": value, "missing_rate_delta": dmiss, "level": lvl})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")


def run_drift(
    start: str | None = None,
    end: str | None = None,
    root: Path = PROJECT_ROOT,
    write: bool = True,
) -> dict[str, Any]:
    """Compare the training window with ``[start, end]`` (default: the test window)."""
    cfg = load_config("monitoring")
    split = load_config("model")["split"]
    df = load_processed(load_config("data"), root)
    sp = time_split(df, split["train_frac"], split["val_frac"])
    ref = df.iloc[sp.train]
    if start or end:
        ts = df[TIMESTAMP]
        lo = pd.Timestamp(start) if start else ts.min()
        hi = pd.Timestamp(end) if end else ts.max()
        cur = df[(ts >= lo) & (ts <= hi)]
    else:
        cur = df.iloc[sp.test]
    sensors = usable_sensors(ref[sensor_columns(ref)], load_config("rootcause")["max_missing_frac"])
    table = drift_table(ref[sensors], cur[sensors], cfg)
    counts = table["level"].value_counts().to_dict()
    result = {
        "reference": {
            "start": str(ref[TIMESTAMP].iloc[0]),
            "end": str(ref[TIMESTAMP].iloc[-1]),
            "runs": len(ref),
        },
        "current": {
            "start": str(cur[TIMESTAMP].iloc[0]),
            "end": str(cur[TIMESTAMP].iloc[-1]),
            "runs": len(cur),
        },
        "thresholds": {
            "warn": cfg["psi_warn"],
            "alert": cfg["psi_alert"],
            "missing_rate_delta_alert": cfg["missing_rate_delta_alert"],
        },
        "sensors": len(table),
        "counts": {k: int(counts.get(k, 0)) for k in ("ok", "warn", "alert")},
        "top": json.loads(table.head(25).to_json(orient="records")),
    }
    if write:
        out = root / "reports" / "metrics" / "drift.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
