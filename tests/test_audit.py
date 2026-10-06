import numpy as np
import pandas as pd

from processlens.data.audit import (
    constant_sensors,
    correlation_summary,
    duplicate_columns,
    fail_rate_by_segment,
    informative_missingness,
    near_constant_sensors,
    outlier_counts,
)
from processlens.data.audit_report import conclusions
from processlens.stats import bh_fdr, robust_z


def test_constant_and_near_constant() -> None:
    x = pd.DataFrame(
        {
            "c": [1.0] * 100,
            "nan": [np.nan] * 100,
            "near": [0.0] * 99 + [1.0],
            "ok": np.arange(100, dtype=float),
        }
    )
    assert constant_sensors(x) == ["c", "nan"]
    assert near_constant_sensors(x, top_freq=0.99, unique_ratio=0.01) == ["near"]


def test_duplicate_columns() -> None:
    x = pd.DataFrame({"a": [1.0, np.nan, 3], "b": [1.0, np.nan, 3], "c": [1.0, 2, 3]})
    assert duplicate_columns(x) == [["a", "b"]]


def test_informative_missingness_finds_planted_signal() -> None:
    rng = np.random.default_rng(1)
    n = 400
    y = pd.Series((rng.random(n) < 0.1).astype(int))
    signal = rng.normal(size=n)
    signal[(y == 1).to_numpy()] = np.nan  # missing exactly when failing
    noise = rng.normal(size=n)
    noise[rng.random(n) < 0.2] = np.nan
    res = informative_missingness(pd.DataFrame({"signal": signal, "noise": noise}), y)
    q = res.set_index("sensor")["q"]
    assert q["signal"] < 1e-6
    assert q["noise"] > 0.05


def test_bh_fdr_keeps_nan_and_is_monotone() -> None:
    q = bh_fdr(np.array([0.01, np.nan, 0.04, 0.03]))
    assert np.isnan(q[1])
    assert np.all(q[~np.isnan(q)] >= np.array([0.01, 0.04, 0.03]))


def test_correlation_summary_clusters_correlated_pair() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=200)
    x = pd.DataFrame({"a": a, "a2": a * 2 + 0.01 * rng.normal(size=200), "b": rng.normal(size=200)})
    res = correlation_summary(x, high=0.9, cluster_rho=0.8)
    assert res["n_clusters"] == 2
    assert res["largest_cluster"] == 2
    assert abs(res["share_pairs_above"] - 1 / 3) < 1e-9


def test_robust_z_and_outliers() -> None:
    s = pd.Series([0.0, 1, -1, 0.5, -0.5, 100])
    assert robust_z(s).abs().idxmax() == 5
    assert outlier_counts(pd.DataFrame({"s": s}), 5.0)["s"] == 1
    assert robust_z(pd.Series([1.0, 1, 1])).isna().all()


def test_fail_rate_by_segment() -> None:
    y = pd.Series([1] * 6 + [0] * 4)
    seg = fail_rate_by_segment(y, [0.6, 0.2, 0.2])
    assert [s["rows"] for s in seg] == [6, 2, 2]
    assert seg[0]["fail_rate"] == 1.0
    assert seg[2]["fail_rate"] == 0.0


def test_conclusions_mention_every_decision() -> None:
    audit = {
        "shape": {"rows": 10},
        "missing": {"threshold": 0.5, "sensors_above_threshold": ["s1"], "rows_with_any": 10},
        "constant_sensors": ["s2"],
        "near_constant_sensors": [],
        "duplicate_groups": [],
        "informative_missingness": {"alpha": 0.05, "significant": []},
        "correlation": {
            "high_threshold": 0.9,
            "share_pairs_above": 0.01,
            "cluster_rho": 0.8,
            "n_sensors": 3,
            "n_clusters": 2,
            "largest_cluster": 2,
        },
        "outliers": {"sensors_with_any": 1, "n_sensors_checked": 3, "z": 5.0},
        "time": {
            "fail_rate_week_min": 0.0,
            "fail_rate_week_max": 0.2,
            "runs_per_week_min": 1,
            "runs_per_week_max": 9,
            "fail_rate_by_split_segment": [{"fail_rate": 0.1}, {"fail_rate": 0.05}],
        },
    }
    text = " ".join(conclusions(audit))
    for word in ["missing", "constant", "indicator", "cluster", "clipping", "Time-ordered"]:
        assert word in text
