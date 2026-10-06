"""Data audit for SECOM: missingness, useless sensors, correlation, time, outliers.

The audit is descriptive and uses the full table. Every cleaning decision it
motivates is re-computed on the training window only inside the Phase 2 pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import fisher_exact

from processlens import viz
from processlens.config import PROJECT_ROOT, load_config
from processlens.data.contract import load_processed, sensor_columns
from processlens.data.ingest import LABEL, TIMESTAMP
from processlens.stats import bh_fdr, robust_z, spearman_matrix

# ---------------------------------------------------------------- computations


def missing_fraction(x: pd.DataFrame) -> pd.Series:
    """Return the share of missing values per column."""
    return x.isna().mean()


def constant_sensors(x: pd.DataFrame) -> list[str]:
    """Return sensors with at most one distinct observed value (or all missing)."""
    return [c for c in x.columns if x[c].nunique(dropna=True) <= 1]


def near_constant_sensors(x: pd.DataFrame, top_freq: float, unique_ratio: float) -> list[str]:
    """Return non-constant sensors dominated by one value or with very few distinct values."""
    out = []
    for c in x.columns:
        obs = x[c].dropna()
        if obs.nunique() <= 1:
            continue
        dominant = obs.value_counts(normalize=True).iloc[0]
        if dominant >= top_freq or obs.nunique() / len(obs) < unique_ratio:
            out.append(c)
    return out


def duplicate_columns(x: pd.DataFrame) -> list[list[str]]:
    """Return groups of columns that are exactly identical (values and NaN positions)."""
    groups: dict[int, list[str]] = {}
    for c in x.columns:
        key = hash(x[c].to_numpy(na_value=np.nan).tobytes())
        groups.setdefault(key, []).append(c)
    dupes = []
    for cols in groups.values():  # confirm hash groups exactly
        while len(cols) > 1:
            head, rest = cols[0], cols[1:]
            same = [c for c in rest if x[head].equals(x[c])]
            if same:
                dupes.append([head, *same])
            cols = [c for c in rest if c not in same]
    return dupes


def informative_missingness(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Fisher's exact test of (missing vs observed) x (fail vs pass) per sensor, with BH q."""
    rows = []
    fail = y.to_numpy() == 1
    for c in x.columns:
        miss = x[c].isna().to_numpy()
        if miss.all() or not miss.any():
            continue
        table = [
            [int((miss & fail).sum()), int((miss & ~fail).sum())],
            [int((~miss & fail).sum()), int((~miss & ~fail).sum())],
        ]
        odds, p = fisher_exact(table)
        rows.append(
            {
                "sensor": c,
                "missing_frac": float(miss.mean()),
                "fail_rate_missing": float(fail[miss].mean()),
                "fail_rate_observed": float(fail[~miss].mean()),
                "odds_ratio": float(odds),
                "p": float(p),
            }
        )
    res = pd.DataFrame(rows)
    if res.empty:
        return res.assign(q=[])
    res["q"] = bh_fdr(res["p"])
    return res.sort_values("p").reset_index(drop=True)


def correlation_summary(x: pd.DataFrame, high: float, cluster_rho: float) -> dict[str, Any]:
    """Share of sensor pairs above ``high`` |Spearman| and a cluster-count preview."""
    corr = spearman_matrix(x).abs()
    vals = corr.to_numpy()
    iu = np.triu_indices_from(vals, k=1)
    pairs = vals[iu]
    pairs = pairs[~np.isnan(pairs)]
    dist = 1.0 - np.nan_to_num(vals, nan=0.0)
    np.fill_diagonal(dist, 0.0)
    labels = fcluster(
        linkage(squareform(dist, checks=False), method="average"),
        t=1.0 - cluster_rho,
        criterion="distance",
    )
    sizes = pd.Series(labels).value_counts()
    return {
        "n_sensors": int(x.shape[1]),
        "n_pairs": int(pairs.size),
        "high_threshold": high,
        "share_pairs_above": float((pairs > high).mean()),
        "cluster_rho": cluster_rho,
        "n_clusters": int(sizes.size),
        "n_singletons": int((sizes == 1).sum()),
        "largest_cluster": int(sizes.max()),
        "cluster_sizes": sizes.sort_values(ascending=False).tolist(),
    }


def time_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly run counts and failure rate."""
    weekly = df.set_index(TIMESTAMP)[LABEL].resample("W").agg(["count", "sum"])
    weekly.columns = ["runs", "failures"]
    weekly["fail_rate"] = weekly["failures"] / weekly["runs"].replace(0, np.nan)
    return weekly


def fail_rate_by_segment(y: pd.Series, fractions: list[float]) -> list[dict[str, Any]]:
    """Failure rate in consecutive time-ordered segments of the given row fractions."""
    edges = np.round(np.cumsum([0.0, *fractions]) * len(y)).astype(int)
    edges[-1] = len(y)
    return [
        {"segment": i, "rows": int(b - a), "fail_rate": float(y.iloc[a:b].mean())}
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True))
    ]


def outlier_counts(x: pd.DataFrame, z: float) -> pd.Series:
    """Count observations with |robust z| > ``z`` per sensor."""
    return pd.Series({c: int((robust_z(x[c]).abs() > z).sum()) for c in x.columns})


# ---------------------------------------------------------------- figures


def _fig_missing_hist(miss: pd.Series, threshold: float, path: Path) -> Path:
    fig, ax = viz.new_figure(7, 3.5)
    ax.hist(miss * 100, bins=40, color=viz.SERIES[0], edgecolor=viz.SURFACE, linewidth=1)
    ax.axvline(threshold * 100, color=viz.INK_SECONDARY, linestyle="--", linewidth=1)
    ax.text(
        threshold * 100 + 1,
        ax.get_ylim()[1] * 0.9,
        f"{threshold:.0%} threshold",
        color=viz.INK_SECONDARY,
        fontsize=9,
    )
    ax.set_xlabel("% missing")
    ax.set_ylabel("sensors")
    ax.set_title("Missing values per sensor", loc="left")
    return viz.save(fig, path)


def _fig_missing_heatmap(df: pd.DataFrame, sensors: list[str], path: Path) -> Path:
    miss = df[sensors].isna()
    order = miss.mean().sort_values(ascending=False).index
    fig, ax = viz.new_figure(9, 4.5)
    ax.imshow(
        miss[order].to_numpy().T,
        aspect="auto",
        cmap=viz.BLUES,
        interpolation="nearest",
        vmin=0,
        vmax=1.6,
    )
    ticks = np.linspace(0, len(df) - 1, 6).astype(int)
    ax.set_xticks(ticks, [df[TIMESTAMP].iloc[i].strftime("%d %b") for i in ticks])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("production runs in time order")
    ax.set_ylabel(f"{len(sensors)} sensors with any missing\n(most missing at top)")
    ax.set_title("Missingness over time (blue = missing)", loc="left")
    return viz.save(fig, path)


def _fig_weekly(weekly: pd.DataFrame, column: str, ylabel: str, title: str, path: Path) -> Path:
    fig, ax = viz.new_figure(8, 3.2)
    vals = weekly[column] * (100 if column == "fail_rate" else 1)
    if column == "fail_rate":
        ax.plot(weekly.index, vals, color=viz.SERIES[0], linewidth=2, marker="o", markersize=5)
    else:
        ax.bar(weekly.index, vals, width=5, color=viz.SERIES[0], edgecolor=viz.SURFACE)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    return viz.save(fig, path)


def _fig_cluster_sizes(sizes: list[int], path: Path) -> Path:
    fig, ax = viz.new_figure(7, 3.2)
    counts = pd.Series(sizes).value_counts().sort_index()
    ax.bar(
        [str(i) for i in counts.index],
        counts.to_numpy(),
        color=viz.SERIES[0],
        edgecolor=viz.SURFACE,
    )
    ax.set_xlabel("cluster size (sensors)")
    ax.set_ylabel("number of clusters")
    ax.set_title("Correlated-sensor clusters (preview)", loc="left")
    return viz.save(fig, path)


def _fig_missingness_signal(res: pd.DataFrame, alpha: float, path: Path) -> Path:
    fig, ax = viz.new_figure(7, 3.5)
    sig = res["q"] < alpha
    ax.scatter(
        res.loc[~sig, "missing_frac"] * 100,
        -np.log10(res.loc[~sig, "p"]),
        s=18,
        color=viz.MUTED,
        label=f"q ≥ {alpha}",
    )
    ax.scatter(
        res.loc[sig, "missing_frac"] * 100,
        -np.log10(res.loc[sig, "p"]),
        s=24,
        color=viz.SERIES[0],
        label=f"q < {alpha}",
    )
    ax.set_xlabel("% missing")
    ax.set_ylabel("−log10 p (Fisher)")
    ax.set_title("Is being missing linked to failure?", loc="left")
    ax.legend(frameon=False, fontsize=9)
    return viz.save(fig, path)


# ---------------------------------------------------------------- orchestration


def compute_audit(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run every audit computation and return a JSON-serialisable result."""
    a = cfg["audit"]
    split = load_config("model")["split"]
    sensors = sensor_columns(df)
    x, y = df[sensors], df[LABEL]
    miss = missing_fraction(x)
    const = constant_sensors(x)
    near = near_constant_sensors(x, a["near_constant_top_freq"], a["near_constant_unique_ratio"])
    usable = [c for c in sensors if c not in const and miss[c] <= a["high_missing_threshold"]]
    info = informative_missingness(x, y)
    weekly = time_summary(df)
    outl = outlier_counts(x[usable], a["robust_z_threshold"])
    return {
        "shape": {"rows": len(df), "sensors": len(sensors)},
        "label": {
            "failures": int(y.sum()),
            "passes": int((y == 0).sum()),
            "base_rate": float(y.mean()),
        },
        "time": {
            "start": df[TIMESTAMP].min().isoformat(),
            "end": df[TIMESTAMP].max().isoformat(),
            "days": float((df[TIMESTAMP].max() - df[TIMESTAMP].min()).total_seconds() / 86400),
            "weeks": int(len(weekly)),
            "runs_per_week_min": int(weekly["runs"].min()),
            "runs_per_week_max": int(weekly["runs"].max()),
            "fail_rate_week_min": float(weekly["fail_rate"].min()),
            "fail_rate_week_max": float(weekly["fail_rate"].max()),
            "fail_rate_by_split_segment": fail_rate_by_segment(
                y, [split["train_frac"], split["val_frac"], split["test_frac"]]
            ),
            "weekly": [
                {"week": i.date().isoformat(), "runs": int(r.runs), "failures": int(r.failures)}
                for i, r in weekly.iterrows()
            ],
        },
        "missing": {
            "total_cell_frac": float(x.isna().to_numpy().mean()),
            "sensors_with_any": int((miss > 0).sum()),
            "sensors_complete": int((miss == 0).sum()),
            "threshold": a["high_missing_threshold"],
            "sensors_above_threshold": sorted(miss[miss > a["high_missing_threshold"]].index),
            "rows_with_any": int(x.isna().any(axis=1).sum()),
        },
        "constant_sensors": const,
        "near_constant_sensors": near,
        "duplicate_groups": duplicate_columns(x.drop(columns=const)),
        "informative_missingness": {
            "alpha": a["fdr_alpha"],
            "n_tested": int(len(info)),
            "significant": info[info["q"] < a["fdr_alpha"]].to_dict(orient="records"),
            "top": info.head(10).to_dict(orient="records"),
        },
        "correlation": correlation_summary(
            x[usable],
            a["high_corr_threshold"],
            load_config("rootcause")["clusters"]["min_abs_spearman"],
        ),
        "outliers": {
            "z": a["robust_z_threshold"],
            "n_sensors_checked": int(len(outl)),
            "sensors_with_any": int((outl > 0).sum()),
            "total_flags": int(outl.sum()),
            "top": outl.sort_values(ascending=False).head(10).to_dict(),
        },
        "usable_sensors": len(usable),
        "_frames": {"miss": miss, "weekly": weekly, "info": info},
    }


def make_figures(df: pd.DataFrame, res: dict[str, Any], fig_dir: Path) -> dict[str, Path]:
    """Write all audit figures and return their paths by key."""
    f = res["_frames"]
    any_missing = [c for c in sensor_columns(df) if f["miss"][c] > 0]
    figs = {
        "missing_hist": _fig_missing_hist(
            f["miss"], res["missing"]["threshold"], fig_dir / "audit_missing_hist.png"
        ),
        "missing_heatmap": _fig_missing_heatmap(
            df, any_missing, fig_dir / "audit_missing_heatmap.png"
        ),
        "fail_rate": _fig_weekly(
            f["weekly"],
            "fail_rate",
            "failure rate (%)",
            "Weekly failure rate",
            fig_dir / "audit_weekly_fail_rate.png",
        ),
        "runs": _fig_weekly(
            f["weekly"],
            "runs",
            "runs",
            "Production runs per week",
            fig_dir / "audit_weekly_runs.png",
        ),
        "clusters": _fig_cluster_sizes(
            res["correlation"]["cluster_sizes"], fig_dir / "audit_cluster_sizes.png"
        ),
    }
    if not f["info"].empty:
        figs["missingness_signal"] = _fig_missingness_signal(
            f["info"],
            res["informative_missingness"]["alpha"],
            fig_dir / "audit_missingness_signal.png",
        )
    return figs


def run_audit(cfg: dict[str, Any] | None = None, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load, audit, write figures, ``reports/metrics/audit.json`` and the markdown report."""
    from processlens.data.audit_report import render_markdown

    cfg = cfg or load_config("data")
    df = load_processed(cfg, root)
    res = compute_audit(df, cfg)
    report = root / cfg["audit"]["report"]
    figs = make_figures(df, res, root / cfg["audit"]["figures_dir"])
    public = {k: v for k, v in res.items() if not k.startswith("_")}
    metrics = root / "reports" / "metrics" / "audit.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps(public, indent=2, default=float), encoding="utf-8")
    rel = {k: Path(p).relative_to(report.parent).as_posix() for k, p in figs.items()}
    report.write_text(render_markdown(public, rel), encoding="utf-8")
    return {"report": report.relative_to(root).as_posix(), "metrics": metrics.as_posix()}
