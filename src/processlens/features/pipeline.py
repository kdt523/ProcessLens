"""Leakage-safe sklearn pipelines: every statistic is learned in ``fit`` on training rows."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

MODEL_NAMES = ["dummy", "logreg_l2", "logreg_l1", "random_forest", "lightgbm"]
LINEAR = {"logreg_l2", "logreg_l1"}


class DropSensors(BaseEstimator, TransformerMixin):
    """Drop columns that are too often missing or have zero variance in the fit data."""

    def __init__(self, max_missing_frac: float = 0.5) -> None:
        self.max_missing_frac = max_missing_frac

    def fit(self, x: pd.DataFrame, y: Any = None) -> DropSensors:
        """Decide which columns to keep using only ``x`` (the training rows)."""
        x = pd.DataFrame(x)
        miss = x.isna().mean()
        nunique = x.nunique(dropna=True)
        keep = (miss <= self.max_missing_frac) & (nunique > 1)
        self.feature_names_in_ = np.asarray(x.columns, dtype=object)
        self.keep_ = list(x.columns[keep.to_numpy()])
        self.dropped_ = list(x.columns[~keep.to_numpy()])
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return only the kept columns."""
        return pd.DataFrame(x)[self.keep_]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the kept column names."""
        return np.asarray(self.keep_, dtype=object)


class EarlyStoppedLGBM(ClassifierMixin, BaseEstimator):
    """LightGBM whose tree count is chosen by early stopping on the time-ordered tail.

    ``fit`` holds out the last ``early_stopping_frac`` of the (time-sorted) rows,
    finds the best iteration there, then refits on all rows with that many trees.
    ``scale_pos_weight`` is set from the fit rows' class balance.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        early_stopping_rounds: int = 100,
        early_stopping_frac: float = 0.2,
        random_state: int = 0,
    ) -> None:
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_frac = early_stopping_frac
        self.random_state = random_state

    def _model(self, y: np.ndarray, n_estimators: int | None = None) -> lgb.LGBMClassifier:
        params = dict(self.params or {})
        if n_estimators is not None:
            params["n_estimators"] = n_estimators
        pos = max(int(y.sum()), 1)
        return lgb.LGBMClassifier(
            **params,
            scale_pos_weight=(len(y) - pos) / pos,
            random_state=self.random_state,
            verbose=-1,
            n_jobs=1,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> EarlyStoppedLGBM:
        """Pick the tree count on the time-ordered tail, then refit on everything."""
        x, y = np.asarray(x), np.asarray(y)
        cut = int(round(len(y) * (1 - self.early_stopping_frac)))
        best = None
        if 0 < y[cut:].sum() < len(y) - cut and y[:cut].sum() > 0:
            probe = self._model(y[:cut]).fit(
                x[:cut],
                y[:cut],
                eval_X=x[cut:],
                eval_y=y[cut:],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            )
            best = max(int(probe.best_iteration_ or 1), 1)
        self.best_iteration_ = best
        self.model_ = self._model(y, best).fit(x, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        return np.asarray(self.model_.predict_proba(np.asarray(x)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return class predictions at 0.5."""
        return np.asarray(self.model_.predict(np.asarray(x)))

    @property
    def feature_importances_(self) -> np.ndarray:
        """Expose the fitted model's importances."""
        return self.model_.feature_importances_


def make_estimator(name: str, cfg: dict[str, Any], seed: int) -> BaseEstimator:
    """Return the final estimator for model ``name``."""
    params = dict(cfg["models"].get(name, {}))
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name in LINEAR:
        l1 = name == "logreg_l1"
        return LogisticRegression(
            l1_ratio=1.0 if l1 else 0.0,
            solver="liblinear" if l1 else "lbfgs",
            class_weight="balanced",
            random_state=seed,
            **params,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1, **params
        )
    if name == "lightgbm":
        es_rounds = params.pop("early_stopping_rounds")
        es_frac = params.pop("early_stopping_frac")
        return EarlyStoppedLGBM(params, es_rounds, es_frac, seed)
    raise ValueError(f"Unknown model {name!r}")


def build_pipeline(name: str, cfg: dict[str, Any], seed: int) -> Pipeline:
    """Return drop → impute (+indicators) → [scale] → model, all fit on training rows only."""
    pre = cfg["preprocess"]
    steps: list[tuple[str, Any]] = [
        ("drop", DropSensors(pre["max_missing_frac"])),
        (
            "impute",
            SimpleImputer(
                strategy=pre["impute_strategy"],
                add_indicator=pre["add_indicator"],
                keep_empty_features=False,
            ),
        ),
    ]
    if name in LINEAR:
        steps.append(("scale", RobustScaler()))
    steps.append(("model", make_estimator(name, cfg, seed)))
    return Pipeline(steps)


def feature_names(pipe: Pipeline) -> list[str]:
    """Return model-input feature names (kept sensors + missing indicators)."""
    return [str(n) for n in pipe[:-1].get_feature_names_out()]
