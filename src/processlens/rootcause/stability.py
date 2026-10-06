"""L1-logistic stability selection (Meinshausen & Bühlmann, 2010)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

CLIP = 10.0  # robust-scaled values are clipped to ±CLIP for solver stability


def stratified_half(y: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Sample ``frac`` of each class without replacement; return sorted row indices."""
    idx = [
        rng.choice(np.flatnonzero(y == k), int(round(frac * (y == k).sum())), replace=False)
        for k in (0, 1)
    ]
    return np.sort(np.concatenate(idx))


def selection_frequency(
    x: pd.DataFrame, y: np.ndarray, n_subsamples: int, frac: float, c: float, seed: int
) -> pd.Series:
    """Share of subsamples in which each sensor gets a non-zero L1 coefficient.

    The window is median-imputed and robust-scaled once (no out-of-sample prediction
    is made, so there is nothing to leak into), clipped to ±``CLIP``, then an L1
    logistic regression is fit on each stratified subsample.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    z = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(x)
    z = np.clip(RobustScaler().fit_transform(z), -CLIP, CLIP)
    counts = np.zeros(x.shape[1])
    for _ in range(n_subsamples):
        idx = stratified_half(y, frac, rng)
        model = LogisticRegression(
            l1_ratio=1.0, solver="liblinear", C=c, class_weight="balanced"
        ).fit(z[idx], y[idx])
        counts += np.abs(model.coef_.ravel()) > 1e-10
    return pd.Series(counts / n_subsamples, index=x.columns, name="stability")
