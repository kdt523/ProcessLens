"""Root-cause ranking engine: ranks *suspect* sensor clusters associated with failure.

Rankings are associations in observational data, not causes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from processlens.config import PROJECT_ROOT, load_config
from processlens.data.contract import load_processed, sensor_columns
from processlens.data.ingest import LABEL, TIMESTAMP
from processlens.data.split import time_split
from processlens.rootcause.aggregate import cluster_table, consensus_score, method_ranks
from processlens.rootcause.clusters import sensor_clusters
from processlens.rootcause.shap_rank import mean_abs_shap
from processlens.rootcause.stability import selection_frequency
from processlens.rootcause.temporal import fail_rate_by_bin, onset
from processlens.rootcause.univariate import mann_whitney_table, missingness_table

SINGLE_METHODS = ["univariate", "stability", "shap"]


def usable_sensors(x: pd.DataFrame, max_missing: float) -> list[str]:
    """Sensors with ≤ ``max_missing`` missing and more than one distinct value in ``x``."""
    keep = (x.isna().mean() <= max_missing) & (x.nunique(dropna=True) > 1)
    return list(x.columns[keep.to_numpy()])


def sensor_table(
    x: pd.DataFrame, y: np.ndarray, cfg: dict[str, Any], methods: list[str]
) -> pd.DataFrame:
    """Per-sensor statistics for the requested methods (univariate always computed)."""
    table = mann_whitney_table(x, y)
    if "stability" in methods:
        s = cfg["stability"]
        table["stability"] = selection_frequency(
            x, y, s["n_subsamples"], s["subsample_frac"], s["C"], cfg["seed"]
        )
    else:
        table["stability"] = 0.0
    if "shap" in methods:
        sh = mean_abs_shap(x, y, cfg["shap"]["cv_folds"], cfg["shap"]["lightgbm"], cfg["seed"])
        table["shap"] = sh
        total = sh.sum()
        table["shap_share"] = sh / total if total > 0 else 0.0
    else:
        table["shap"] = table["shap_share"] = 0.0
    return table


def rank_suspects(
    x: pd.DataFrame,
    y: np.ndarray,
    cfg: dict[str, Any],
    clusters: pd.Series | None = None,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    """Run every method once and return per-method and consensus cluster rankings.

    ``x`` must already be restricted to usable sensors. Returns a dict with the
    sensor table and one cluster table per single method plus ``consensus``.
    """
    methods = methods or cfg.get("methods", SINGLE_METHODS)
    clusters = (
        clusters
        if clusters is not None
        else sensor_clusters(x, cfg["clusters"]["min_abs_spearman"])
    )
    table = sensor_table(x, np.asarray(y), cfg, methods)
    ranks = method_ranks(table, methods)
    table = table.join(ranks)
    rules = cfg["evidence_rules"]
    out: dict[str, Any] = {"sensors": table, "clusters": clusters}
    for m in methods:
        t = table.assign(score=1.0 / ranks[f"rank_{m}"])
        out[m] = cluster_table(t, clusters, rules)
    table["score"] = consensus_score(ranks, cfg["aggregate"]["method"])
    out["consensus"] = cluster_table(table, clusters, rules)
    return out


def resolve_window(
    df: pd.DataFrame, start: str | None, end: str | None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the analysis window; defaults to the Phase 2 training window."""
    if start is None or end is None:
        split = load_config("model")["split"]
        sp = time_split(df, split["train_frac"], split["val_frac"])
        default_start, default_end = df[TIMESTAMP].iloc[sp.train[[0, -1]]]
        start = start or str(default_start)
        end = end or str(default_end)
    return pd.Timestamp(start), pd.Timestamp(end)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def run_rootcause(
    start: str | None = None,
    end: str | None = None,
    top_k: int | None = None,
    cfg: dict[str, Any] | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Analyse a time window of the real data and write ``reports/metrics/rootcause.json``."""
    cfg = cfg or load_config("rootcause")
    top_k = top_k or cfg["aggregate"]["top_k"]
    df = load_processed(load_config("data"), root)
    lo, hi = resolve_window(df, start or cfg["window"]["start"], end or cfg["window"]["end"])
    w = df[(df[TIMESTAMP] >= lo) & (df[TIMESTAMP] <= hi)].reset_index(drop=True)
    y = w[LABEL].to_numpy()
    if not 0 < y.sum() < len(y):
        raise ValueError("Window must contain both passing and failing runs")
    x_all = w[sensor_columns(w)]
    x = x_all[usable_sensors(x_all, cfg["max_missing_frac"])]

    t0 = time.perf_counter()
    res = rank_suspects(x, y, cfg)
    runtime = time.perf_counter() - t0
    cons = res["consensus"].head(top_k)
    t = cfg["temporal"]
    temporal = {
        r.representative: {
            "fail_rate_by_bin": fail_rate_by_bin(
                x[r.representative], y, w[TIMESTAMP], t["n_bins"], t["bin_freq"]
            ),
            "onset": onset(x[r.representative], w[TIMESTAMP], t["onset_freq"], t["onset_min_size"]),
        }
        for r in cons.head(t["top_n"]).itertuples()
    }
    miss = missingness_table(x_all, y)
    result = {
        "note": "Suspects are sensor clusters associated with failure in observational data; "
        "they are not proven causes.",
        "window": {
            "start": lo.isoformat(),
            "end": hi.isoformat(),
            "runs": int(len(w)),
            "failures": int(y.sum()),
            "base_rate": float(y.mean()),
        },
        "sensors_analysed": int(x.shape[1]),
        "n_clusters": int(res["clusters"].nunique()),
        "consensus_method": cfg["aggregate"]["method"],
        "runtime_seconds": runtime,
        "evidence_counts": res["consensus"]["evidence"].value_counts().to_dict(),
        "n_sensors_q_below_alpha": int((res["sensors"]["q"] < cfg["fdr_alpha"]).sum()),
        "suspects": _records(cons),
        "per_method_top": {
            m: res[m].head(top_k)["representative"].tolist() for m in SINGLE_METHODS if m in res
        },
        "temporal": temporal,
        "missingness_signal": _records(miss[miss["missing_q"] < cfg["fdr_alpha"]].reset_index()),
        "sensor_stats": _records(
            res["sensors"]
            .reset_index()[["sensor", "p", "q", "effect", "stability", "shap_share", "score"]]
            .sort_values("score", ascending=False)
        ),
        "clusters": {s: int(c) for s, c in res["clusters"].items()},
    }
    out = root / cfg["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return result
