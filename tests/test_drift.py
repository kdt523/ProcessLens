import numpy as np
import pandas as pd

from processlens.monitoring.drift import drift_table, level, psi

CFG = {"psi_bins": 10, "psi_warn": 0.1, "psi_alert": 0.25, "missing_rate_delta_alert": 0.1}


def test_psi_zero_for_same_distribution() -> None:
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.02


def test_psi_large_for_shift() -> None:
    rng = np.random.default_rng(1)
    assert psi(rng.normal(size=2000), rng.normal(1.0, 1, 2000)) > 0.25


def test_psi_ignores_nans_and_handles_constant() -> None:
    a = np.array([1.0, 2, 3, np.nan] * 50)
    assert psi(a, a) < 1e-9
    assert psi(np.ones(10), np.ones(10)) == 0.0
    assert np.isnan(psi(np.array([np.nan]), np.array([1.0])))


def test_levels() -> None:
    assert level(0.05, 0.1, 0.25) == "ok"
    assert level(0.15, 0.1, 0.25) == "warn"
    assert level(0.3, 0.1, 0.25) == "alert"


def test_drift_table_flags_shift_and_missing_jump() -> None:
    rng = np.random.default_rng(2)
    ref = pd.DataFrame(
        {
            "stable": rng.normal(size=500),
            "shifted": rng.normal(size=500),
            "holey": rng.normal(size=500),
        }
    )
    cur = pd.DataFrame(
        {
            "stable": rng.normal(size=300),
            "shifted": rng.normal(2, 1, 300),
            "holey": rng.normal(size=300),
        }
    )
    cur.loc[:100, "holey"] = np.nan
    t = drift_table(ref, cur, CFG).set_index("sensor")
    assert t.loc["stable", "level"] == "ok"
    assert t.loc["shifted", "level"] == "alert"
    assert t.loc["holey", "level"] == "alert" and t.loc["holey", "missing_rate_delta"] > 0.3
