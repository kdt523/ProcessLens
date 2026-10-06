"""Post-hoc probability calibration on the validation window."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator


def calibrate(model: Any, x_val: Any, y_val: np.ndarray, method: str) -> CalibratedClassifierCV:
    """Fit a calibrator on validation rows on top of an already-fitted (frozen) model."""
    return CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(x_val, y_val)


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int) -> dict[str, list[float]]:
    """Observed failure rate vs mean predicted probability in quantile bins."""
    obs, pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return {"mean_predicted": pred.tolist(), "observed_rate": obs.tolist()}
