"""Small statistical helpers shared across modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def bh_fdr(pvalues: np.ndarray | pd.Series) -> np.ndarray:
    """Return Benjamini–Hochberg q-values; NaN p-values stay NaN."""
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def spearman_matrix(x: pd.DataFrame) -> pd.DataFrame:
    """Approximate pairwise Spearman correlation with missing values.

    Each column is ranked once over its observed values, then pairwise-complete
    Pearson correlation is computed on the ranks. This differs from exact
    pairwise Spearman (which re-ranks within each pair's common rows) only when
    missingness patterns differ, and is ~50x faster on SECOM.
    """
    return x.rank().corr(method="pearson")


def robust_z(x: pd.Series) -> pd.Series:
    """Return robust z-scores ``(x - median) / (1.4826 * MAD)``; NaN if MAD is 0."""
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    if not mad or np.isnan(mad):
        return pd.Series(np.nan, index=x.index)
    return (x - med) / mad
