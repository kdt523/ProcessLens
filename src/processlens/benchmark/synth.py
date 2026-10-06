"""Planted-fault scenario generator: real sensor matrix, synthetic labels from known causes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

MECHANISMS = ("linear", "threshold", "drift_window")


@dataclass(frozen=True)
class Scenario:
    """One benchmark cell replicate. ``mechanism == 'null'`` means no cause."""

    mechanism: str
    n_causes: int
    beta: float
    seed: int
    null_kind: str | None = None
    causes: tuple[str, ...] = field(default=())

    @property
    def key(self) -> str:
        """Stable short hash identifying the scenario (used for caching)."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def cause_pool(x: pd.DataFrame, max_missing: float, min_distinct: int) -> list[str]:
    """Sensors eligible as planted causes: few missing values and enough distinct values."""
    ok = (x.isna().mean() < max_missing) & (x.nunique(dropna=True) >= min_distinct)
    return list(x.columns[ok.to_numpy()])


def pick_causes(
    pool: list[str], clusters: pd.Series, n: int, rng: np.random.Generator
) -> tuple[str, ...]:
    """Pick ``n`` cause sensors from distinct clusters."""
    order = rng.permutation(pool)
    chosen: list[str] = []
    used: set[int] = set()
    for s in order:
        if int(clusters[s]) not in used:
            chosen.append(str(s))
            used.add(int(clusters[s]))
        if len(chosen) == n:
            return tuple(chosen)
    raise ValueError("Not enough distinct clusters in the cause pool")


def standardise(v: np.ndarray) -> np.ndarray:
    """Z-score with missing values set to 0 (the mean)."""
    mu, sd = np.nanmean(v), np.nanstd(v)
    z = (v - mu) / sd if sd > 0 else np.zeros_like(v)
    return np.nan_to_num(z, nan=0.0)


def signal(
    x: pd.DataFrame,
    causes: tuple[str, ...],
    mechanism: str,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sum over causes of the per-SD driver for ``mechanism`` (before multiplying by beta)."""
    total = np.zeros(len(x))
    window = np.ones(len(x), dtype=bool)
    if mechanism == "drift_window":
        width = int(round(cfg["drift_window_frac"] * len(x)))
        start = int(rng.integers(0, len(x) - width + 1))
        window = np.zeros(len(x), dtype=bool)
        window[start : start + width] = True
    for c in causes:
        v = x[c].to_numpy(dtype=float)
        if mechanism == "threshold":
            above = (v > np.nanquantile(v, cfg["threshold_quantile"])).astype(float)
            total += standardise(above)
        elif mechanism in ("linear", "drift_window"):
            total += standardise(v) * window
        else:
            raise ValueError(f"Unknown mechanism {mechanism!r}")
    return total


def calibrate_intercept(eta: np.ndarray, target: float, tol: float = 1e-6) -> float:
    """Find b0 so that mean(sigmoid(b0 + eta)) equals ``target`` (bisection)."""
    lo, hi = -30.0, 30.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(mid + eta)))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def generate_labels(
    x: pd.DataFrame,
    scenario: Scenario,
    cfg: dict[str, Any],
    real_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Draw synthetic 0/1 labels for ``scenario`` (seeded, reproducible)."""
    rng = np.random.default_rng(scenario.seed)
    target = cfg["target_base_rate"]
    if scenario.mechanism == "null":
        if scenario.null_kind == "permuted":
            if real_labels is None:
                raise ValueError("permuted null needs real labels")
            return rng.permutation(np.asarray(real_labels))
        return (rng.random(len(x)) < target).astype(int)
    eta = scenario.beta * signal(x, scenario.causes, scenario.mechanism, cfg, rng)
    b0 = calibrate_intercept(eta, target)
    p = 1 / (1 + np.exp(-(b0 + eta)))
    return (rng.random(len(x)) < p).astype(int)


def build_scenarios(
    grid: dict[str, Any], cfg: dict[str, Any], pool: list[str], clusters: pd.Series
) -> list[Scenario]:
    """Expand a grid (``full`` or ``smoke``) into concrete, seeded scenarios."""
    master = np.random.default_rng(cfg["seed"])
    out: list[Scenario] = []
    for mech in grid["mechanisms"]:
        for n in grid["n_causes"]:
            for beta in grid["effect_sizes"]:
                for _ in range(grid["n_seeds"]):
                    seed = int(master.integers(0, 2**31 - 1))
                    causes = pick_causes(pool, clusters, n, np.random.default_rng(seed))
                    out.append(Scenario(mech, n, float(beta), seed, None, causes))
    kinds = cfg["null_kinds"]
    for i in range(grid["n_null"]):
        seed = int(master.integers(0, 2**31 - 1))
        out.append(Scenario("null", 0, 0.0, seed, kinds[i % len(kinds)]))
    return out
