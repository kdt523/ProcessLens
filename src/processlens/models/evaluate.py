"""Classification metrics with bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

Metric = Callable[[np.ndarray, np.ndarray], float]


def recall_at_precision(y: np.ndarray, p: np.ndarray, min_precision: float) -> float:
    """Highest recall achievable at precision ≥ ``min_precision`` (0 if never reached)."""
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= min_precision
    return float(rec[ok].max()) if ok.any() else 0.0


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin ECE: weighted mean |observed rate − mean predicted| per bin."""
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def metric_fns(cfg: dict[str, Any]) -> dict[str, Metric]:
    """Return the named metric functions used everywhere."""
    ev = cfg["evaluation"]
    return {
        "pr_auc": lambda y, p: float(average_precision_score(y, p)),
        "roc_auc": lambda y, p: float(roc_auc_score(y, p)),
        "recall_at_precision": lambda y, p: recall_at_precision(y, p, ev["min_precision"]),
        "brier": lambda y, p: float(brier_score_loss(y, p)),
        "ece": lambda y, p: expected_calibration_error(y, p, ev["ece_bins"]),
    }


def _both_classes(y: np.ndarray) -> bool:
    return 0 < y.sum() < len(y)


def bootstrap_indices(y: np.ndarray, n_iter: int, seed: int) -> list[np.ndarray]:
    """Resample row indices with replacement, skipping draws that lack a class."""
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    while len(out) < n_iter:
        idx = rng.integers(0, len(y), len(y))
        if _both_classes(y[idx]):
            out.append(idx)
    return out


def _ci(values: np.ndarray, level: float) -> tuple[float, float]:
    lo, hi = np.quantile(values, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def evaluate_with_ci(
    y: np.ndarray, p: np.ndarray, cfg: dict[str, Any], seed: int
) -> dict[str, dict[str, float]]:
    """Point estimate and percentile bootstrap CI for every metric, plus the base rate."""
    ev = cfg["evaluation"]
    y, p = np.asarray(y), np.asarray(p)
    idx = bootstrap_indices(y, ev["bootstrap_iters"], seed)
    out: dict[str, dict[str, float]] = {}
    for name, fn in metric_fns(cfg).items():
        boots = np.array([fn(y[i], p[i]) for i in idx])
        lo, hi = _ci(boots, ev["ci_level"])
        out[name] = {"value": fn(y, p), "ci_low": lo, "ci_high": hi}
    out["base_rate"] = {"value": float(y.mean()), "n": int(len(y)), "positives": int(y.sum())}
    return out


def paired_bootstrap_diff(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric: Metric,
    n_iter: int,
    level: float,
    seed: int,
) -> dict[str, float]:
    """Bootstrap the difference ``metric(a) − metric(b)`` on the same resampled rows."""
    y = np.asarray(y)
    diffs = np.array(
        [metric(y[i], p_a[i]) - metric(y[i], p_b[i]) for i in bootstrap_indices(y, n_iter, seed)]
    )
    lo, hi = _ci(diffs, level)
    return {
        "diff": float(metric(y, p_a) - metric(y, p_b)),
        "ci_low": lo,
        "ci_high": hi,
        "p_a_not_better": float((diffs <= 0).mean()),
    }
