"""Train, select, calibrate and evaluate defect models; log everything to MLflow.

Protocol:
1. time-ordered 60/20/20 split;
2. per model: rolling-origin CV PR-AUC inside train, plus metrics on validation (reported);
3. select the best non-dummy model by mean rolling-origin CV PR-AUC (validation holds
   too few failures to rank models reliably);
4. calibrate it on validation;
5. one final pass over test for the selected model, the reference model and the dummy.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from processlens.config import PROJECT_ROOT, load_config
from processlens.data.contract import load_processed, sensor_columns
from processlens.data.ingest import LABEL, TIMESTAMP, sha256_file
from processlens.data.split import TimeSplit, rolling_origin_folds, time_split
from processlens.features.pipeline import MODEL_NAMES, build_pipeline, feature_names
from processlens.models.calibrate import calibrate, reliability_table
from processlens.models.evaluate import evaluate_with_ci, metric_fns, paired_bootstrap_diff
from processlens.models.policy import policy_report

log = logging.getLogger(__name__)
ARTIFACT_DIR = "artifacts"


def git_sha(root: Path = PROJECT_ROOT) -> str:
    """Return the current git commit SHA (or ``unknown``)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cv_scores(name: str, x: pd.DataFrame, y: np.ndarray, cfg: dict[str, Any]) -> list[float]:
    """Rolling-origin PR-AUC of model ``name`` within the training window."""
    pr_auc = metric_fns(cfg)["pr_auc"]
    scores = []
    for fit_idx, ev_idx in rolling_origin_folds(len(y), cfg["split"]["cv_folds"]):
        if not 0 < y[ev_idx].sum() < len(ev_idx):
            continue
        pipe = build_pipeline(name, cfg, cfg["seed"]).fit(x.iloc[fit_idx], y[fit_idx])
        scores.append(pr_auc(y[ev_idx], pipe.predict_proba(x.iloc[ev_idx])[:, 1]))
    return scores


def fit_candidates(
    x: pd.DataFrame, y: np.ndarray, sp: TimeSplit, cfg: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Fit every model on train; record CV scores and validation metrics."""
    out = {}
    xtr, ytr, xva, yva = x.iloc[sp.train], y[sp.train], x.iloc[sp.val], y[sp.val]
    for name in MODEL_NAMES:
        t0 = time.perf_counter()
        cv = cv_scores(name, xtr, ytr, cfg)
        pipe = build_pipeline(name, cfg, cfg["seed"]).fit(xtr, ytr)
        p_val = pipe.predict_proba(xva)[:, 1]
        out[name] = {
            "pipeline": pipe,
            "cv_pr_auc": cv,
            "cv_pr_auc_mean": float(np.mean(cv)) if cv else float("nan"),
            "val": evaluate_with_ci(yva, p_val, cfg, cfg["seed"]),
            "n_features": len(feature_names(pipe)),
            "fit_seconds": time.perf_counter() - t0,
        }
        log.info(
            "%s: val PR-AUC %.3f, CV %.3f",
            name,
            out[name]["val"]["pr_auc"]["value"],
            out[name]["cv_pr_auc_mean"],
        )
    return out


def select_model(cands: dict[str, dict[str, Any]]) -> str:
    """Return the non-dummy model with the highest mean rolling-origin CV PR-AUC."""
    scored = {k: v["cv_pr_auc_mean"] for k, v in cands.items() if k != "dummy"}
    return max(scored, key=lambda k: scored[k])


def final_test_pass(
    x: pd.DataFrame,
    y: np.ndarray,
    sp: TimeSplit,
    cands: dict[str, dict[str, Any]],
    best: str,
    calibrated: Any,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run the single evaluation on the test window."""
    xte, yte = x.iloc[sp.test], y[sp.test]
    ref = cfg["selection"]["reference_model"]
    probs = {
        f"{best}_calibrated": calibrated.predict_proba(xte)[:, 1],
        f"{best}_uncalibrated": cands[best]["pipeline"].predict_proba(xte)[:, 1],
        ref: cands[ref]["pipeline"].predict_proba(xte)[:, 1],
        "dummy": cands["dummy"]["pipeline"].predict_proba(xte)[:, 1],
    }
    ev = cfg["evaluation"]
    return {
        "probs": probs,
        "metrics": {k: evaluate_with_ci(yte, p, cfg, cfg["seed"]) for k, p in probs.items()},
        "paired_vs_reference": {
            "model": best,
            "reference": ref,
            "metric": "pr_auc",
            **paired_bootstrap_diff(
                yte,
                probs[f"{best}_uncalibrated"],
                probs[ref],
                metric_fns(cfg)["pr_auc"],
                ev["bootstrap_iters"],
                ev["ci_level"],
                cfg["seed"],
            ),
        },
    }


def log_to_mlflow(
    cands: dict[str, dict[str, Any]],
    best: str,
    calibrated: Any,
    test: dict[str, Any],
    figures: dict[str, Path],
    tags: dict[str, str],
    cfg: dict[str, Any],
    root: Path,
) -> str | None:
    """One MLflow run per candidate plus a final run with the registered calibrated model."""
    import mlflow
    import mlflow.sklearn

    m = cfg["mlflow"]
    mlflow.set_tracking_uri(m["tracking_uri"])
    if mlflow.get_experiment_by_name(m["experiment"]) is None:
        mlflow.create_experiment(m["experiment"], (root / m["artifact_root"]).as_uri())
    mlflow.set_experiment(m["experiment"])
    for name, c in cands.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({**tags, "stage": "candidate"})
            mlflow.log_params(
                {
                    "model": name,
                    **cfg["models"].get(name, {}),
                    **{f"pre_{k}": v for k, v in cfg["preprocess"].items()},
                }
            )
            mlflow.log_metrics(
                {
                    "cv_pr_auc_mean": c["cv_pr_auc_mean"],
                    "n_features": c["n_features"],
                    **{f"val_{k}": v["value"] for k, v in c["val"].items()},
                }
            )
    with mlflow.start_run(run_name=f"final_{best}") as run:
        mlflow.set_tags({**tags, "stage": "final", "selected_model": best})
        mlflow.log_params({"selected_model": best, "calibration": cfg["calibration"]["method"]})
        for k, v in test["metrics"][f"{best}_calibrated"].items():
            mlflow.log_metric(f"test_{k}", v["value"])
        for p in figures.values():
            mlflow.log_artifact(str(p), artifact_path="figures")
        # cloudpickle: self-produced artifact; skops cannot serialise the LightGBM booster
        info = mlflow.sklearn.log_model(
            calibrated,
            name="model",
            registered_model_name=m["registered_model"],
            serialization_format="cloudpickle",
        )
        log.info("Registered %s from run %s", m["registered_model"], run.info.run_id)
        return str(info.model_uri)


def _public(cands: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {k: {kk: vv for kk, vv in v.items() if kk != "pipeline"} for k, v in cands.items()}


def run_train(
    cfg: dict[str, Any] | None = None, root: Path = PROJECT_ROOT, use_mlflow: bool = True
) -> dict[str, Any]:
    """Full Phase 2 run; writes ``reports/metrics/model.json`` and figures."""
    from processlens.models.figures import make_model_figures

    cfg = cfg or load_config("model")
    data_cfg = load_config("data")
    costs = load_config("costs")
    df = load_processed(data_cfg, root)
    x, y = df[sensor_columns(df)], df[LABEL].to_numpy()
    sp = time_split(df, cfg["split"]["train_frac"], cfg["split"]["val_frac"])

    cands = fit_candidates(x, y, sp, cfg)
    best = select_model(cands)
    calibrated = calibrate(
        cands[best]["pipeline"], x.iloc[sp.val], y[sp.val], cfg["calibration"]["method"]
    )
    test = final_test_pass(x, y, sp, cands, best, calibrated, cfg)
    yte, p_best = y[sp.test], test["probs"][f"{best}_calibrated"]
    policy = policy_report(yte, p_best, costs)
    n_bins = cfg["evaluation"]["ece_bins"]
    reliability = {
        "calibrated": reliability_table(yte, p_best, n_bins),
        "uncalibrated": reliability_table(yte, test["probs"][f"{best}_uncalibrated"], n_bins),
    }
    figures = make_model_figures(
        yte, test["probs"], best, reliability, policy, root / "reports" / "figures"
    )

    def window(idx: np.ndarray) -> dict[str, Any]:
        return {
            "rows": int(len(idx)),
            "failures": int(y[idx].sum()),
            "start": df[TIMESTAMP].iloc[idx[0]].isoformat(),
            "end": df[TIMESTAMP].iloc[idx[-1]].isoformat(),
        }

    tags = {
        "git_sha": git_sha(root),
        "data_sha256": sha256_file(root / data_cfg["paths"]["processed"]),
    }
    result = {
        "protocol": __doc__.split("Protocol:")[1].strip(),
        "tags": tags,
        "split": {"train": window(sp.train), "val": window(sp.val), "test": window(sp.test)},
        "candidates": _public(cands),
        "selected_model": best,
        "calibration": cfg["calibration"]["method"],
        "test": {k: v for k, v in test.items() if k != "probs"},
        "reliability_test": reliability,
        "policy_test": policy,
        "dropped_on_train": cands[best]["pipeline"].named_steps["drop"].dropped_,
    }
    out = root / "reports" / "metrics" / "model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    from processlens.models.report import render_model_report

    (root / "reports" / "model_report.md").write_text(
        render_model_report(json.loads(out.read_text(encoding="utf-8"))), encoding="utf-8"
    )
    art = root / ARTIFACT_DIR
    art.mkdir(exist_ok=True)
    joblib.dump(
        {"model": calibrated, "name": best, "features": list(x.columns)}, art / "model.joblib"
    )
    if use_mlflow:
        result["mlflow_model_uri"] = log_to_mlflow(
            cands, best, calibrated, test, figures, tags, cfg, root
        )
    return result
