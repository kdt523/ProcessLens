"""Benchmark metrics: hit@k, precision@k, null false alarms, minimum detectable effect."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def hit_at_k(ranked: list[Any], truth: list[Any], k: int) -> float:
    """Share of distinct true items found in the top ``k`` (1/0 for a single cause)."""
    truth_set = set(truth)
    return len(truth_set & set(ranked[:k])) / len(truth_set) if truth_set else float("nan")


def precision_at_k(ranked: list[Any], truth: list[Any], k: int) -> float:
    """Share of the top ``k`` that are true items."""
    return len(set(ranked[:k]) & set(truth)) / k


def scenario_rows(
    results: list[dict[str, Any]], clusters: pd.Series, k_values: list[int]
) -> pd.DataFrame:
    """One row per (planted scenario, method) with sensor- and cluster-level metrics."""
    rows = []
    for r in results:
        sc = r["scenario"]
        if sc["mechanism"] == "null":
            continue
        causes = list(sc["causes"])
        cause_clusters = [int(clusters[c]) for c in causes]
        for m, sensors in r["sensor_rank"].items():
            row = {
                "key": r["key"],
                "mechanism": sc["mechanism"],
                "n_causes": sc["n_causes"],
                "beta": sc["beta"],
                "method": m,
                "runtime": r["runtime_seconds"],
            }
            for k in k_values:
                row[f"sensor_hit@{k}"] = hit_at_k(sensors, causes, k)
                row[f"cluster_hit@{k}"] = hit_at_k(r["cluster_rank"][m], cause_clusters, k)
                row[f"cluster_precision@{k}"] = precision_at_k(
                    r["cluster_rank"][m], cause_clusters, k
                )
            rows.append(row)
    return pd.DataFrame(rows)


def null_false_alarms(results: list[dict[str, Any]]) -> dict[str, Any]:
    """How often a no-cause scenario still produces a 'finding'."""
    nulls = [r for r in results if r["scenario"]["mechanism"] == "null"]
    if not nulls:
        nan = float("nan")
        return {
            "n": 0,
            "any_q_below_alpha": nan,
            "any_strong": nan,
            "any_moderate_or_strong": nan,
            "mean_sensors_q_below_alpha": nan,
            "by_kind": {},
        }
    s = pd.DataFrame([{**r["null_stats"], "kind": r["scenario"]["null_kind"]} for r in nulls])
    summary: dict[str, Any] = {
        "n": int(len(s)),
        "any_q_below_alpha": float((s["n_q_below_alpha"] > 0).mean()),
        "any_strong": float((s["n_strong"] > 0).mean()),
        "any_moderate_or_strong": float((s["n_moderate_or_strong"] > 0).mean()),
        "mean_sensors_q_below_alpha": float(s["n_q_below_alpha"].mean()),
    }
    summary["by_kind"] = {
        str(kind): {
            "n": int(len(g)),
            "any_q_below_alpha": float((g["n_q_below_alpha"] > 0).mean()),
            "any_strong": float((g["n_strong"] > 0).mean()),
            "any_moderate_or_strong": float((g["n_moderate_or_strong"] > 0).mean()),
        }
        for kind, g in s.groupby("kind")
    }
    return summary


def curve(rows: pd.DataFrame, metric: str, by: list[str]) -> pd.DataFrame:
    """Mean of ``metric`` with a normal-approximation 95% CI per group."""
    g = rows.groupby(by)[metric]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["ci"] = 1.96 * out["std"].fillna(0) / np.sqrt(out["count"])
    return out


def minimum_detectable_effect(
    rows: pd.DataFrame, metric: str, target: float
) -> dict[str, dict[str, float | None]]:
    """Smallest beta whose mean ``metric`` ≥ ``target``, per method and mechanism.

    Requires every larger beta to also reach the target (monotone detection).
    """
    out: dict[str, dict[str, float | None]] = {}
    means = rows.groupby(["method", "mechanism", "beta"])[metric].mean()
    for (method, mech), s in means.groupby(level=[0, 1]):
        betas = s.droplevel([0, 1]).sort_index()
        mde = None
        for b in betas.index[::-1]:
            if betas[b] >= target:
                mde = float(b)
            else:
                break
        out.setdefault(str(method), {})[str(mech)] = mde
    return out


def by_cause_missing(
    rows: pd.DataFrame, results: list[dict[str, Any]], missing: pd.Series, metric: str
) -> list[dict[str, Any]]:
    """Consensus ``metric`` by mechanism and missing-share bucket of the planted causes."""
    miss = {
        r["key"]: float(np.mean([missing[c] for c in r["scenario"]["causes"]]))
        for r in results
        if r["scenario"]["mechanism"] != "null"
    }
    df = rows[(rows["method"] == "consensus") & (rows["beta"] >= 1.0)].copy()
    df["cause_missing"] = df["key"].map(miss)
    df["bucket"] = pd.cut(
        df["cause_missing"], [-0.001, 0.0, 0.05, 1.0], labels=["0%", "0–5%", ">5%"]
    )
    g = df.groupby(["mechanism", "bucket"], observed=True)[metric].agg(["mean", "count"])
    return [
        {
            "mechanism": m,
            "cause_missing": str(b),
            "mean": float(r["mean"]),
            "count": int(r["count"]),
        }
        for (m, b), r in g.iterrows()
    ]


def summarise(
    results: list[dict[str, Any]],
    clusters: pd.Series,
    cfg: dict[str, Any],
    missing: pd.Series | None = None,
) -> dict[str, Any]:
    """All benchmark metrics as a JSON-serialisable dict (``_rows`` holds the raw table)."""
    ks = cfg["k_values"]
    rows = scenario_rows(results, clusters, ks)
    mk = f"cluster_hit@{cfg['mde_k']}"
    by_beta = {
        f"{lvl}_hit@{k}": curve(rows, f"{lvl}_hit@{k}", ["method", "mechanism", "beta"]).to_dict(
            orient="records"
        )
        for lvl in ("sensor", "cluster")
        for k in ks
    }
    overall = rows.groupby("method")[[c for c in rows.columns if "@" in c]].mean()
    by_ncauses = rows.groupby(["method", "n_causes"])[mk].mean().unstack().round(4)
    runtimes = [r["runtime_seconds"] for r in results]
    return {
        "n_scenarios": len(results),
        "n_planted": int(rows["key"].nunique()) if len(rows) else 0,
        "n_null": sum(r["scenario"]["mechanism"] == "null" for r in results),
        "overall_by_method": overall.round(4).to_dict(orient="index"),
        "by_beta": by_beta,
        "by_n_causes": {
            m: {str(k): v for k, v in d.items()}
            for m, d in by_ncauses.to_dict(orient="index").items()
        },
        "mde": {
            "metric": mk,
            "target": cfg["mde_hit_target"],
            "values": minimum_detectable_effect(rows, mk, cfg["mde_hit_target"]),
        },
        "null_false_alarms": null_false_alarms(results),
        "by_cause_missing_beta_ge_1": (
            by_cause_missing(rows, results, missing, mk) if missing is not None else []
        ),
        "runtime_seconds": {
            "mean": float(np.mean(runtimes)),
            "p95": float(np.quantile(runtimes, 0.95)),
            "n": len(runtimes),
        },
        "_rows": rows,
    }


def check_gates(summary: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    """Quality gate: consensus cluster hit@5 at a large effect, and null false-alarm rate."""
    rows = summary["_rows"]
    big = rows[(rows["method"] == "consensus") & (rows["beta"] >= gates["large_effect"])]
    hit = float(big["cluster_hit@5"].mean()) if len(big) else float("nan")
    fa = summary["null_false_alarms"].get("any_strong", float("nan"))
    return {
        "consensus_cluster_hit5_large_effect": hit,
        "min_required": gates["min_hit5_large_effect"],
        "null_any_strong_rate": fa,
        "max_allowed": gates["max_null_false_alarm"],
        "passed": bool(
            hit >= gates["min_hit5_large_effect"] and fa <= gates["max_null_false_alarm"]
        ),
    }
