"""Temporal views of a suspect: failure rate by sensor bin over time, and onset of a shift."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import ruptures as rpt


def fail_rate_by_bin(
    values: pd.Series, y: np.ndarray, ts: pd.Series, n_bins: int, freq: str
) -> list[dict[str, Any]]:
    """Failure rate per (period, sensor-quantile bin); bins are quantiles of the window."""
    df = pd.DataFrame({"v": values.to_numpy(), "y": np.asarray(y), "ts": ts.to_numpy()})
    df = df.dropna(subset=["v"])
    if df["v"].nunique() < 2:
        return []
    df["bin"] = pd.qcut(df["v"].rank(method="first"), n_bins, labels=False)
    df["period"] = df["ts"].dt.to_period(freq.rstrip("S")).astype(str)
    g = df.groupby(["period", "bin"])["y"].agg(["mean", "count"]).reset_index()
    return [
        {
            "period": r.period,
            "bin": int(r.bin) + 1,
            "fail_rate": float(r["mean"]),
            "runs": int(r["count"]),
        }
        for _, r in g.iterrows()
    ]


def onset(values: pd.Series, ts: pd.Series, freq: str, min_size: int = 3) -> dict[str, Any] | None:
    """Single most likely change point in the sensor's ``freq`` mean (ruptures binary segmentation).

    Returns the first period after the change, the means before/after and the shift in
    units of the sensor's standard deviation, or None if the series is too short.
    """
    s = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(ts)).resample(freq).mean().dropna()
    if len(s) < 2 * min_size:
        return None
    algo = rpt.Binseg(model="l2", min_size=min_size).fit(s.to_numpy().reshape(-1, 1))
    k = algo.predict(n_bkps=1)[0]
    if k >= len(s):
        return None
    before, after = float(s.iloc[:k].mean()), float(s.iloc[k:].mean())
    sd = float(np.nanstd(values.to_numpy()))
    return {
        "onset": s.index[k].date().isoformat(),
        "mean_before": before,
        "mean_after": after,
        "shift_sd": (after - before) / sd if sd > 0 else float("nan"),
        "n_periods": int(len(s)),
    }
