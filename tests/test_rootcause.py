import numpy as np
import pandas as pd
import pytest

from processlens.config import load_config
from processlens.rootcause.aggregate import (
    cluster_table,
    consensus_score,
    evidence_strength,
    method_ranks,
)
from processlens.rootcause.clusters import sensor_clusters
from processlens.rootcause.engine import rank_suspects, usable_sensors
from processlens.rootcause.shap_rank import mean_abs_shap
from processlens.rootcause.stability import selection_frequency, stratified_half
from processlens.rootcause.temporal import fail_rate_by_bin, onset
from processlens.rootcause.univariate import mann_whitney_table

CAUSE, TWIN = "sensor_005", "sensor_006"


@pytest.fixture(scope="module")
def cfg() -> dict:
    c = load_config("rootcause")
    c["stability"]["n_subsamples"] = 30
    c["shap"]["lightgbm"]["n_estimators"] = 60
    return c


@pytest.fixture(scope="module")
def planted() -> tuple[pd.DataFrame, np.ndarray]:
    """20 noise sensors; failure driven by sensor_005; sensor_006 is a noisy copy of it."""
    rng = np.random.default_rng(7)
    n = 600
    x = pd.DataFrame({f"sensor_{i:03d}": rng.normal(size=n) for i in range(1, 21)})
    x[TWIN] = x[CAUSE] + 0.2 * rng.normal(size=n)
    x.loc[rng.random(n) < 0.05, "sensor_010"] = np.nan
    logit = -3.2 + 1.8 * x[CAUSE].to_numpy()
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return x, y


def test_mann_whitney_finds_cause_with_positive_effect(planted) -> None:
    x, y = planted
    t = mann_whitney_table(x, y)
    assert t["p"].idxmin() in {CAUSE, TWIN}
    assert t.loc[CAUSE, "effect"] > 0.3
    assert t.loc[CAUSE, "q"] < 1e-6
    assert (t.drop([CAUSE, TWIN])["q"] > 0.01).mean() > 0.8


def test_rank_biserial_matches_definition() -> None:
    x = pd.DataFrame({"s": [1.0, 2, 3, 4, 5, 6, 7, 8]})
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # failures all higher → effect = +1
    assert mann_whitney_table(x, y).loc["s", "effect"] == pytest.approx(1.0)
    assert mann_whitney_table(x, 1 - y).loc["s", "effect"] == pytest.approx(-1.0)


def test_constant_sensor_gets_nan_not_error() -> None:
    x = pd.DataFrame({"c": [1.0] * 10, "v": np.arange(10.0)})
    y = np.array([0, 1] * 5)
    t = mann_whitney_table(x, y)
    assert np.isnan(t.loc["c", "p"])


def test_stratified_half_keeps_class_balance() -> None:
    y = np.array([1] * 10 + [0] * 90)
    idx = stratified_half(y, 0.5, np.random.default_rng(0))
    assert y[idx].sum() == 5 and len(idx) == 50


def test_stability_selects_cause(planted, cfg) -> None:
    x, y = planted
    freq = selection_frequency(x, y, 30, 0.5, 0.05, 0)
    # L1 splits selection between correlated twins, hence cluster-level reporting
    assert max(freq[CAUSE], freq[TWIN]) >= 0.8
    assert freq.drop([CAUSE, TWIN]).mean() < 0.3


def test_shap_ranks_cause_first(planted, cfg) -> None:
    x, y = planted
    s = mean_abs_shap(x, y, 3, cfg["shap"]["lightgbm"], 0)
    assert s.idxmax() in {CAUSE, TWIN}


def test_clusters_group_twins_only(planted) -> None:
    x, _ = planted
    cl = sensor_clusters(x, 0.8)
    assert cl[CAUSE] == cl[TWIN]
    assert cl.nunique() == x.shape[1] - 1


def test_method_ranks_and_mrr() -> None:
    t = pd.DataFrame(
        {"p": [0.01, 0.5, 0.2], "stability": [0.9, 0.1, 0.1], "shap": [0.3, 0.1, 0.2]},
        index=["a", "b", "c"],
    )
    r = method_ranks(t, ["univariate", "stability", "shap"])
    assert r.loc["a"].tolist() == [1, 1, 1]
    assert r.loc["b", "rank_stability"] == 2.5  # tie shares the average rank
    s = consensus_score(r, "mrr")
    assert s.idxmax() == "a" and s["a"] == pytest.approx(1.0)
    assert consensus_score(r, "borda").idxmax() == "a"


def test_evidence_rules(cfg) -> None:
    rules = cfg["evidence_rules"]
    assert evidence_strength(0.001, 0.3, 0.7, rules) == "strong"
    assert evidence_strength(0.001, 0.3, 0.4, rules) == "moderate"
    assert evidence_strength(0.04, -0.15, 0.35, rules) == "moderate"
    assert evidence_strength(0.2, 0.5, 1.0, rules) == "weak"
    assert evidence_strength(float("nan"), 0.5, 1.0, rules) == "weak"


def test_cluster_table_picks_best_member_as_representative(cfg) -> None:
    sensors = pd.DataFrame(
        {
            "score": [0.2, 0.9, 0.5],
            "q": [0.5, 0.001, 0.2],
            "effect": [0.1, 0.4, -0.6],
            "stability": [0.1, 0.8, 0.2],
            "shap_share": [0.1, 0.5, 0.4],
        },
        index=["a", "b", "c"],
    )
    clusters = pd.Series({"a": 0, "b": 0, "c": 1})
    t = cluster_table(sensors, clusters, cfg["evidence_rules"])
    assert t.loc[0, "representative"] == "b"
    assert t.loc[0, "members"] == ["b", "a"]
    assert t.loc[0, "shap_share"] == pytest.approx(0.6)
    assert t.loc[1, "effect_size"] == pytest.approx(-0.6)
    assert t["consensus_rank"].tolist() == [1, 2]


def test_engine_ranks_planted_cluster_first_with_strong_evidence(planted, cfg) -> None:
    x, y = planted
    res = rank_suspects(x, y, cfg)
    top = res["consensus"].iloc[0]
    assert set(top["members"]) == {CAUSE, TWIN}
    assert top["evidence"] == "strong"
    for m in ["univariate", "stability", "shap"]:
        assert CAUSE in res[m].iloc[0]["members"]
    assert (res["consensus"]["evidence"] == "strong").sum() == 1


def test_engine_on_pure_noise_has_no_strong_evidence(cfg) -> None:
    rng = np.random.default_rng(11)
    x = pd.DataFrame({f"sensor_{i:03d}": rng.normal(size=500) for i in range(1, 16)})
    y = (rng.random(500) < 0.08).astype(int)
    res = rank_suspects(x, y, cfg)
    assert (res["consensus"]["evidence"] == "strong").sum() == 0


def test_usable_sensors() -> None:
    x = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [1.0] * 4, "c": [np.nan, np.nan, np.nan, 1.0]})
    assert usable_sensors(x, 0.5) == ["a"]


def test_onset_detects_planted_shift() -> None:
    ts = pd.Series(pd.date_range("2008-07-01", periods=60 * 4, freq="6h"))
    rng = np.random.default_rng(3)
    v = pd.Series(rng.normal(size=len(ts)))
    v[ts >= "2008-08-10"] += 3.0
    res = onset(v, ts, "D")
    assert res is not None
    assert abs((pd.Timestamp(res["onset"]) - pd.Timestamp("2008-08-10")).days) <= 1
    assert res["shift_sd"] > 1.0


def test_onset_returns_none_for_short_series() -> None:
    ts = pd.Series(pd.date_range("2008-07-01", periods=4, freq="D"))
    assert onset(pd.Series([1.0, 2, 3, 4]), ts, "D") is None


def test_fail_rate_by_bin_shows_gradient() -> None:
    rng = np.random.default_rng(5)
    n = 800
    v = pd.Series(rng.normal(size=n))
    y = (v > 1).astype(int).to_numpy()
    ts = pd.Series(pd.date_range("2008-07-01", periods=n, freq="3h"))
    rows = pd.DataFrame(fail_rate_by_bin(v, y, ts, 4, "MS"))
    by_bin = rows.groupby("bin")["fail_rate"].mean()
    assert by_bin.idxmax() == 4 and by_bin[1] == 0
