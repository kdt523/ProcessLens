import numpy as np
import pandas as pd
import pytest

from processlens.config import load_config
from processlens.features.pipeline import EarlyStoppedLGBM, build_pipeline
from processlens.models.calibrate import calibrate
from processlens.models.evaluate import (
    evaluate_with_ci,
    expected_calibration_error,
    paired_bootstrap_diff,
    recall_at_precision,
)
from processlens.models.policy import caught_at_budget, evaluate_policy, gains_curve
from processlens.models.train import select_model


@pytest.fixture
def cfg() -> dict:
    c = load_config("model")
    c["evaluation"]["bootstrap_iters"] = 200
    return c


def test_perfect_scores_give_top_metrics(cfg: dict) -> None:
    y = np.array([0] * 90 + [1] * 10)
    p = y * 0.9 + 0.05
    res = evaluate_with_ci(y, p, cfg, 0)
    assert res["pr_auc"]["value"] == pytest.approx(1.0)
    assert res["roc_auc"]["value"] == pytest.approx(1.0)
    assert res["base_rate"]["value"] == pytest.approx(0.1)
    assert res["pr_auc"]["ci_low"] <= res["pr_auc"]["value"] <= res["pr_auc"]["ci_high"]


def test_random_scores_pr_auc_near_base_rate(cfg: dict) -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.07).astype(int)
    res = evaluate_with_ci(y, rng.random(5000), cfg, 0)
    assert abs(res["pr_auc"]["value"] - 0.07) < 0.02


def test_recall_at_precision() -> None:
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1])
    assert recall_at_precision(y, p, 1.0) == pytest.approx(2 / 3)
    assert recall_at_precision(y, p, 0.5) == pytest.approx(1.0)


def test_ece_zero_when_calibrated() -> None:
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.5)) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.full(100, 0.9)) == pytest.approx(0.4)


def test_paired_bootstrap_detects_better_model() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.2).astype(int)
    good = y + rng.normal(0, 0.3, 400)
    bad = rng.random(400)
    from sklearn.metrics import average_precision_score

    res = paired_bootstrap_diff(y, good, bad, average_precision_score, 300, 0.95, 0)
    assert res["ci_low"] > 0
    assert res["p_a_not_better"] < 0.01


def test_gains_and_budget() -> None:
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert caught_at_budget(y, p, 0.3) == pytest.approx(1.0)
    assert caught_at_budget(y, p, 0.1) == pytest.approx(0.5)
    g = gains_curve(y, p, 11)
    assert g["caught_frac"][0] == 0 and g["caught_frac"][-1] == pytest.approx(1.0)


def test_policy_costs_and_bayes_threshold() -> None:
    y = np.array([1, 0, 0, 0])
    p = np.array([0.6, 0.3, 0.05, 0.05])
    r = evaluate_policy(y, p, cost_inspection=1.0, cost_missed=4.0)  # threshold 0.25
    assert r["threshold"] == 0.25
    assert r["inspected_frac"] == 0.5
    assert r["caught_frac"] == 1.0
    assert r["cost_model"] == pytest.approx(0.5)
    assert r["cost_inspect_all"] == 1.0
    assert r["cost_inspect_none"] == pytest.approx(1.0)
    assert r["cost_random_same_budget"] == pytest.approx(0.5 + 4 * 0.25 * 0.5)


def test_select_model_uses_cv_and_skips_dummy() -> None:
    cands = {
        "dummy": {"cv_pr_auc_mean": 0.9, "val": {"pr_auc": {"value": 0.9}}},
        "a": {"cv_pr_auc_mean": 0.2, "val": {"pr_auc": {"value": 0.5}}},
        "b": {"cv_pr_auc_mean": 0.3, "val": {"pr_auc": {"value": 0.1}}},
    }
    assert select_model(cands) == "b"


def test_early_stopped_lgbm_learns_signal() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(600, 4))
    y = (x[:, 0] + 0.3 * rng.normal(size=600) > 1.2).astype(int)
    params = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 7}
    m = EarlyStoppedLGBM(params, early_stopping_rounds=30, early_stopping_frac=0.2).fit(x, y)
    assert m.best_iteration_ is not None and m.best_iteration_ <= 300
    assert m.predict_proba(x).shape == (600, 2)
    assert np.argmax(m.feature_importances_) == 0


def test_calibration_runs_on_frozen_model(cfg: dict) -> None:
    rng = np.random.default_rng(4)
    x = pd.DataFrame(rng.normal(size=(400, 3)), columns=["sensor_001", "sensor_002", "sensor_003"])
    y = (x["sensor_001"] + rng.normal(size=400) > 1.5).astype(int).to_numpy()
    pipe = build_pipeline("logreg_l2", cfg, 0).fit(x.iloc[:200], y[:200])
    coef = pipe.named_steps["model"].coef_.copy()
    cal = calibrate(pipe, x.iloc[200:], y[200:], "sigmoid")
    np.testing.assert_allclose(pipe.named_steps["model"].coef_, coef)  # base model untouched
    p = cal.predict_proba(x.iloc[200:])[:, 1]
    assert abs(p.mean() - y[200:].mean()) < 0.05
