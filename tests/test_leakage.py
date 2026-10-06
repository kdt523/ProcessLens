"""Leakage tests: time order, train-only statistics, train-only feature dropping, no label use."""

import numpy as np
import pandas as pd
import pytest

from processlens.config import load_config
from processlens.data.split import rolling_origin_folds, time_split
from processlens.features.pipeline import build_pipeline, feature_names


@pytest.fixture
def cfg() -> dict:
    return load_config("model")


@pytest.fixture
def data() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 300
    ts = pd.Timestamp("2008-07-19") + pd.to_timedelta(np.sort(rng.integers(0, 10_000, n)), "min")
    x = pd.DataFrame({f"sensor_{i:03d}": rng.normal(i, 1 + i, n) for i in range(1, 7)})
    x.loc[rng.random(n) < 0.1, "sensor_002"] = np.nan
    y = (rng.random(n) < 0.15).astype("int64")
    return pd.concat([pd.DataFrame({"timestamp": ts, "label": y}), x], axis=1)


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return df.filter(like="sensor_"), df["label"].to_numpy()


def test_split_is_strictly_time_ordered(data: pd.DataFrame) -> None:
    sp = time_split(data, 0.6, 0.2)
    ts = data["timestamp"]
    assert ts.iloc[sp.train].max() < ts.iloc[sp.val].min()
    assert ts.iloc[sp.val].max() < ts.iloc[sp.test].min()
    assert ts.iloc[sp.train].max() < ts.iloc[sp.test].min()
    assert len(sp.train) + len(sp.val) + len(sp.test) == len(data)


def test_split_never_breaks_timestamp_ties() -> None:
    ts = pd.to_datetime(["2008-01-01"] * 5 + ["2008-01-02"] * 5 + ["2008-01-03"] * 5)
    sp = time_split(pd.DataFrame({"timestamp": ts}), 0.3, 0.3)
    assert ts[sp.train].max() < ts[sp.val].min()


def test_split_rejects_unsorted(data: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        time_split(data.iloc[::-1], 0.6, 0.2)


def test_rolling_folds_only_look_back() -> None:
    folds = list(rolling_origin_folds(100, 4))
    assert len(folds) == 4
    for fit_idx, ev_idx in folds:
        assert fit_idx.max() < ev_idx.min()
        assert fit_idx.min() == 0


@pytest.mark.parametrize("model", ["logreg_l2", "random_forest"])
def test_imputer_and_scaler_use_train_rows_only(data: pd.DataFrame, cfg: dict, model: str) -> None:
    sp = time_split(data, 0.6, 0.2)
    x, y = _xy(data)
    xtr = x.iloc[sp.train]
    pipe = build_pipeline(model, cfg, 0).fit(xtr, y[sp.train])
    kept = pipe.named_steps["drop"].keep_
    np.testing.assert_allclose(pipe.named_steps["impute"].statistics_, xtr[kept].median())
    if model == "logreg_l2":
        imputed = pipe[:2].transform(xtr)
        np.testing.assert_allclose(pipe.named_steps["scale"].center_, np.median(imputed, axis=0))


def test_statistics_do_not_change_when_test_rows_change(data: pd.DataFrame, cfg: dict) -> None:
    sp = time_split(data, 0.6, 0.2)
    x, y = _xy(data)
    a = build_pipeline("logreg_l2", cfg, 0).fit(x.iloc[sp.train], y[sp.train])
    x2 = x.copy()
    x2.iloc[sp.test] = 1e6  # corrupt future rows
    b = build_pipeline("logreg_l2", cfg, 0).fit(x2.iloc[sp.train], y[sp.train])
    np.testing.assert_allclose(
        a.named_steps["impute"].statistics_, b.named_steps["impute"].statistics_
    )
    np.testing.assert_allclose(a.named_steps["model"].coef_, b.named_steps["model"].coef_)


def test_column_all_nan_in_train_is_dropped_even_if_present_in_test(
    data: pd.DataFrame, cfg: dict
) -> None:
    sp = time_split(data, 0.6, 0.2)
    x, y = _xy(data)
    x["sensor_099"] = np.nan
    x.loc[x.index[sp.test], "sensor_099"] = 1.0  # only observed in the future
    pipe = build_pipeline("random_forest", cfg, 0).fit(x.iloc[sp.train], y[sp.train])
    assert "sensor_099" in pipe.named_steps["drop"].dropped_
    assert not any("sensor_099" in f for f in feature_names(pipe))
    assert pipe.predict_proba(x.iloc[sp.test]).shape == (len(sp.test), 2)


def test_column_constant_in_train_is_dropped(data: pd.DataFrame, cfg: dict) -> None:
    sp = time_split(data, 0.6, 0.2)
    x, y = _xy(data)
    x["sensor_098"] = 5.0
    x.loc[x.index[sp.test], "sensor_098"] = np.arange(len(sp.test))
    pipe = build_pipeline("logreg_l1", cfg, 0).fit(x.iloc[sp.train], y[sp.train])
    assert "sensor_098" in pipe.named_steps["drop"].dropped_


def test_features_never_include_label_or_time(data: pd.DataFrame, cfg: dict) -> None:
    x, y = _xy(data)
    pipe = build_pipeline("lightgbm", cfg, 0).fit(x, y)
    names = feature_names(pipe)
    assert all(n.startswith(("sensor_", "missingindicator_sensor_")) for n in names)
    assert not any("label" in n or "timestamp" in n for n in names)


def test_pipeline_rejects_label_derived_input(data: pd.DataFrame, cfg: dict) -> None:
    """Training code builds X only from sensor columns; the contract forbids other columns."""
    from processlens.data.contract import sensor_columns

    assert "label" not in sensor_columns(data)
    assert "timestamp" not in sensor_columns(data)


def test_no_resampling_step_in_pipeline(cfg: dict) -> None:
    for name in ["dummy", "logreg_l2", "logreg_l1", "random_forest", "lightgbm"]:
        steps = [type(s).__name__.lower() for _, s in build_pipeline(name, cfg, 0).steps]
        assert not any("smote" in s or "sampler" in s for s in steps)
