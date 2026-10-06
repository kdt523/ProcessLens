"""Mean |SHAP| per sensor from LightGBM fit inside time-ordered CV folds."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from processlens.data.split import rolling_origin_folds


def _shap_matrix(model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP values (log-odds) via LightGBM's ``pred_contrib``; drops the bias column.

    Identical to ``shap.TreeExplainer(model).shap_values(x)`` and much faster.
    """
    return np.asarray(model.predict(x, pred_contrib=True))[:, :-1]


def mean_abs_shap(
    x: pd.DataFrame, y: np.ndarray, n_folds: int, params: dict[str, Any], seed: int
) -> pd.Series:
    """Average over folds of mean |SHAP| on each held-out block (NaNs handled natively)."""
    y = np.asarray(y)
    total = np.zeros(x.shape[1])
    used = 0
    for fit_idx, ev_idx in rolling_origin_folds(len(y), n_folds):
        yf = y[fit_idx]
        if not 0 < yf.sum() < len(yf):
            continue
        pos = yf.sum()
        model = lgb.LGBMClassifier(
            **params,
            scale_pos_weight=(len(yf) - pos) / pos,
            random_state=seed,
            verbose=-1,
            n_jobs=1,
        ).fit(x.iloc[fit_idx], yf)
        total += np.abs(_shap_matrix(model, x.iloc[ev_idx])).mean(axis=0)
        used += 1
    return pd.Series(total / max(used, 1), index=x.columns, name="shap")
