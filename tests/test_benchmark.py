import numpy as np
import pandas as pd
import pytest

from processlens.benchmark.metrics import (
    check_gates,
    hit_at_k,
    minimum_detectable_effect,
    null_false_alarms,
    precision_at_k,
    scenario_rows,
)
from processlens.benchmark.synth import (
    Scenario,
    build_scenarios,
    calibrate_intercept,
    cause_pool,
    generate_labels,
    pick_causes,
    signal,
)
from processlens.config import load_config


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config("benchmark")


@pytest.fixture(scope="module")
def x() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({f"sensor_{i:03d}": rng.normal(size=n) for i in range(1, 11)})
    df.loc[rng.random(n) < 0.3, "sensor_009"] = np.nan  # too much missing for the pool
    df["sensor_010"] = rng.integers(0, 3, n).astype(float)  # too few distinct values
    return df


def test_cause_pool_filters(x: pd.DataFrame) -> None:
    pool = cause_pool(x, 0.2, 20)
    assert "sensor_009" not in pool and "sensor_010" not in pool
    assert len(pool) == 8


def test_pick_causes_distinct_clusters() -> None:
    clusters = pd.Series({"a": 0, "b": 0, "c": 1, "d": 2})
    for seed in range(10):
        chosen = pick_causes(list(clusters.index), clusters, 3, np.random.default_rng(seed))
        assert len({clusters[c] for c in chosen}) == 3
    with pytest.raises(ValueError):
        pick_causes(["a", "b"], clusters, 2, np.random.default_rng(0))


def test_intercept_hits_target_base_rate() -> None:
    eta = np.random.default_rng(1).normal(0, 2, 5000)
    b0 = calibrate_intercept(eta, 0.066)
    assert np.mean(1 / (1 + np.exp(-(b0 + eta)))) == pytest.approx(0.066, abs=1e-4)


@pytest.mark.parametrize("mech", ["linear", "threshold", "drift_window"])
def test_planted_labels_keep_base_rate_and_carry_signal(x, cfg, mech) -> None:
    sc = Scenario(mech, 1, 2.0, seed=3, causes=("sensor_001",))
    y = generate_labels(x, sc, cfg)
    assert abs(y.mean() - cfg["target_base_rate"]) < 0.02
    cause = x["sensor_001"].to_numpy()
    other = x["sensor_002"].to_numpy()
    assert cause[y == 1].mean() - cause[y == 0].mean() > 0.3
    assert abs(other[y == 1].mean() - other[y == 0].mean()) < 0.25


def test_threshold_signal_only_above_quantile(x, cfg) -> None:
    s = signal(x, ("sensor_001",), "threshold", cfg, np.random.default_rng(0))
    assert len(np.unique(s)) == 2
    assert (s > 0).mean() == pytest.approx(0.05, abs=0.002)


def test_drift_window_signal_is_zero_outside_window(x, cfg) -> None:
    s = signal(x, ("sensor_001",), "drift_window", cfg, np.random.default_rng(0))
    assert (s != 0).mean() == pytest.approx(cfg["drift_window_frac"], abs=0.01)
    nz = np.flatnonzero(s)
    assert nz.max() - nz.min() + 1 == int(round(cfg["drift_window_frac"] * len(x)))


def test_labels_are_reproducible(x, cfg) -> None:
    sc = Scenario("linear", 1, 1.0, seed=9, causes=("sensor_003",))
    assert (generate_labels(x, sc, cfg) == generate_labels(x, sc, cfg)).all()


def test_null_labels(x, cfg) -> None:
    real = (np.arange(len(x)) % 15 == 0).astype(int)
    perm = generate_labels(x, Scenario("null", 0, 0.0, 1, "permuted"), cfg, real)
    assert perm.sum() == real.sum() and not (perm == real).all()
    bern = generate_labels(x, Scenario("null", 0, 0.0, 1, "bernoulli"), cfg)
    assert abs(bern.mean() - cfg["target_base_rate"]) < 0.02


def test_build_scenarios_counts(x, cfg) -> None:
    pool = cause_pool(x, 0.2, 20)
    clusters = pd.Series(range(len(x.columns)), index=x.columns)
    grid = {
        "mechanisms": ["linear", "threshold"],
        "n_causes": [1, 2],
        "effect_sizes": [0.5, 1.0],
        "n_seeds": 3,
        "n_null": 4,
    }
    sc = build_scenarios(grid, cfg, pool, clusters)
    assert len(sc) == 2 * 2 * 2 * 3 + 4
    assert len({s.key for s in sc}) == len(sc)
    assert sc == build_scenarios(grid, cfg, pool, clusters)  # deterministic


def test_hit_and_precision() -> None:
    assert hit_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert hit_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert hit_at_k(["a", "b", "c"], ["a", "z"], 3) == 0.5
    assert precision_at_k(["a", "b", "c", "d", "e"], ["a", "c"], 5) == pytest.approx(0.4)


def _result(
    mech: str,
    beta: float,
    causes: list[str],
    ranked: list[str],
    cl: list[int],
    strong: int = 0,
    q: int = 0,
) -> dict:
    methods = ["univariate", "stability", "shap", "consensus"]
    return {
        "key": f"{mech}{beta}{causes}",
        "scenario": {
            "mechanism": mech,
            "n_causes": len(causes),
            "beta": beta,
            "causes": causes,
            "null_kind": "bernoulli" if mech == "null" else None,
        },
        "runtime_seconds": 1.0,
        "sensor_rank": dict.fromkeys(methods, ranked),
        "cluster_rank": dict.fromkeys(methods, cl),
        "null_stats": {
            "n_q_below_alpha": q,
            "n_strong": strong,
            "n_moderate_or_strong": strong,
            "min_q": 0.5,
            "top_evidence": "weak",
        },
    }


def test_scenario_rows_cluster_vs_sensor_level() -> None:
    clusters = pd.Series({"a": 0, "b": 0, "c": 1})
    # planted 'a', engine ranks its twin 'b' first: cluster hit, sensor miss at k=1
    rows = scenario_rows([_result("linear", 1.0, ["a"], ["b", "c", "a"], [0, 1])], clusters, [1])
    r = rows[rows["method"] == "consensus"].iloc[0]
    assert r["cluster_hit@1"] == 1.0 and r["sensor_hit@1"] == 0.0


def test_null_false_alarms() -> None:
    res = [_result("null", 0.0, [], [], [], strong=1, q=3), _result("null", 0.0, [], [], [])]
    nf = null_false_alarms(res)
    assert nf["any_strong"] == 0.5 and nf["any_q_below_alpha"] == 0.5


def test_mde_requires_monotone_detection() -> None:
    rows = pd.DataFrame(
        {
            "method": ["consensus"] * 4,
            "mechanism": ["linear"] * 4,
            "beta": [0.5, 1.0, 1.5, 2.0],
            "m": [0.9, 0.5, 0.85, 0.95],
        }
    )
    assert minimum_detectable_effect(rows, "m", 0.8)["consensus"]["linear"] == 1.5
    rows["m"] = 0.1
    assert minimum_detectable_effect(rows, "m", 0.8)["consensus"]["linear"] is None


def test_gates() -> None:
    rows = pd.DataFrame(
        {"method": ["consensus"] * 2, "beta": [2.0, 2.0], "cluster_hit@5": [1.0, 0.5]}
    )
    gates = {"large_effect": 2.0, "min_hit5_large_effect": 0.8, "max_null_false_alarm": 0.2}
    g = check_gates({"_rows": rows, "null_false_alarms": {"any_strong": 0.0}}, gates)
    assert g["consensus_cluster_hit5_large_effect"] == 0.75 and not g["passed"]
