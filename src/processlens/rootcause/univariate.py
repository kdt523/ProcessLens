"""Per-sensor Mann–Whitney U test (fail vs pass) with rank-biserial effect size and BH q."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from processlens.data.audit import informative_missingness
from processlens.stats import bh_fdr


def mann_whitney_table(x: pd.DataFrame, y: np.ndarray, min_per_group: int = 3) -> pd.DataFrame:
    """Return sensor, p, q, effect (rank-biserial, + means higher in failures), group sizes.

    Rank-biserial r = 2·U_fail / (n_fail·n_pass) − 1, identical to Cliff's delta.
    Missing values are dropped per sensor.
    """
    fail = np.asarray(y) == 1
    rows = []
    for c in x.columns:
        v = x[c].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        a, b = v[ok & fail], v[ok & ~fail]
        if len(a) < min_per_group or len(b) < min_per_group or np.ptp(v[ok]) == 0:
            rows.append(
                {"sensor": c, "p": np.nan, "effect": np.nan, "n_fail": len(a), "n_pass": len(b)}
            )
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
        rows.append(
            {
                "sensor": c,
                "p": float(p),
                "effect": float(2 * u / (len(a) * len(b)) - 1),
                "n_fail": len(a),
                "n_pass": len(b),
            }
        )
    out = pd.DataFrame(rows)
    out["q"] = bh_fdr(out["p"])
    return out.set_index("sensor")


def missingness_table(x: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Fisher test of missingness vs failure per sensor (from the Phase 1 audit)."""
    res = informative_missingness(x, pd.Series(np.asarray(y), index=x.index))
    if res.empty:
        return pd.DataFrame(columns=["missing_p", "missing_q", "missing_odds_ratio"])
    return res.set_index("sensor").rename(
        columns={"p": "missing_p", "q": "missing_q", "odds_ratio": "missing_odds_ratio"}
    )[["missing_p", "missing_q", "missing_odds_ratio"]]
