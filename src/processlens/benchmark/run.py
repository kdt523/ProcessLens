"""Run the root-cause engine on every planted-fault scenario (parallel, cached)."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from processlens.benchmark.synth import Scenario, build_scenarios, cause_pool, generate_labels
from processlens.config import PROJECT_ROOT, load_config
from processlens.data.contract import load_processed, sensor_columns
from processlens.data.ingest import LABEL
from processlens.rootcause.clusters import sensor_clusters
from processlens.rootcause.engine import rank_suspects, usable_sensors

log = logging.getLogger(__name__)
METHODS = ["univariate", "stability", "shap", "consensus"]
TOP = 20


@dataclass
class Context:
    """Everything shared by all scenarios."""

    x: pd.DataFrame
    clusters: pd.Series
    pool: list[str]
    real_labels: np.ndarray


def build_context(rc_cfg: dict[str, Any], bm_cfg: dict[str, Any], root: Path) -> Context:
    """Load the real sensor matrix, usable sensors, clusters and the cause pool."""
    df = load_processed(load_config("data"), root)
    x_all = df[sensor_columns(df)]
    x = x_all[usable_sensors(x_all, rc_cfg["max_missing_frac"])]
    clusters = sensor_clusters(x, rc_cfg["clusters"]["min_abs_spearman"])
    pool = cause_pool(x, bm_cfg["cause_pool"]["max_missing"], bm_cfg["cause_pool"]["min_distinct"])
    return Context(x, clusters, pool, df[LABEL].to_numpy())


def _sensor_order(table: pd.DataFrame, method: str) -> list[str]:
    col = "score" if method == "consensus" else f"rank_{method}"
    asc = method != "consensus"
    return list(table.sort_values(col, ascending=asc, kind="stable").index[:TOP])


def run_scenario(
    sc: Scenario, ctx: Context, rc_cfg: dict[str, Any], bm_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Generate labels, run every method once, and keep a compact result."""
    y = generate_labels(ctx.x, sc, bm_cfg, ctx.real_labels)
    t0 = time.perf_counter()
    res = rank_suspects(ctx.x, y, rc_cfg, ctx.clusters)
    runtime = time.perf_counter() - t0
    sensors = res["sensors"]
    cons = res["consensus"]
    return {
        "scenario": asdict(sc),
        "key": sc.key,
        "positives": int(y.sum()),
        "runtime_seconds": runtime,
        "sensor_rank": {m: _sensor_order(sensors, m) for m in METHODS},
        "cluster_rank": {m: [int(c) for c in res[m]["cluster"].head(TOP)] for m in METHODS},
        "null_stats": {
            "n_q_below_alpha": int((sensors["q"] < bm_cfg["alpha"]).sum()),
            "min_q": float(sensors["q"].min()),
            "n_strong": int((cons["evidence"] == "strong").sum()),
            "n_moderate_or_strong": int(cons["evidence"].isin(["strong", "moderate"]).sum()),
            "top_evidence": str(cons["evidence"].iloc[0]),
        },
    }


def _cached(sc: Scenario, cache: Path, ctx: Context, rc: dict, bm: dict) -> dict[str, Any]:
    path = cache / f"{sc.key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    out = run_scenario(sc, ctx, rc, bm)
    path.write_text(json.dumps(out), encoding="utf-8")
    return out


def run_benchmark(
    smoke: bool = False, root: Path = PROJECT_ROOT
) -> tuple[list[dict[str, Any]], Context, dict[str, Any], dict[str, Any]]:
    """Run all scenarios of the full or smoke grid; return results and shared context."""
    bm = load_config("benchmark")
    rc = load_config("rootcause")
    grid = bm["smoke" if smoke else "full"]
    if smoke:
        rc["stability"]["n_subsamples"] = grid["n_subsamples"]
    ctx = build_context(rc, bm, root)
    scenarios = build_scenarios(grid, bm, ctx.pool, ctx.clusters)
    cfg_hash = hashlib.sha256(json.dumps([rc, bm], sort_keys=True).encode()).hexdigest()[:10]
    tag = f"{'smoke' if smoke else 'full'}_{cfg_hash}"
    cache = root / bm["cache_dir"] / tag
    cache.mkdir(parents=True, exist_ok=True)
    log.info("Running %d scenarios (%s) on %d sensors", len(scenarios), tag, ctx.x.shape[1])
    results = Parallel(n_jobs=bm["n_jobs"], verbose=5)(
        delayed(_cached)(sc, cache, ctx, rc, bm) for sc in scenarios
    )
    return list(results), ctx, rc, bm
