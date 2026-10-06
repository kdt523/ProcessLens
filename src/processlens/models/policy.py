"""Cost-based inspection policy built on calibrated failure probabilities.

Assumptions (stated in the report):
- each inspection costs ``cost_inspection`` and always finds a defect if present;
- an uninspected failure costs ``cost_missed_defect``; a passed run costs nothing;
- costs are relative units per run, constant over time.
Under these assumptions, with calibrated probabilities, inspecting a run is worth
it exactly when ``p ≥ cost_inspection / cost_missed_defect`` (Bayes decision rule),
so no threshold is tuned on the test data.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def gains_curve(y: np.ndarray, p: np.ndarray, n_points: int = 101) -> dict[str, list[float]]:
    """Share of failures caught when inspecting the top ``x`` share of runs by risk."""
    order = np.argsort(-p, kind="stable")
    caught = np.concatenate([[0], np.cumsum(y[order])]) / max(y.sum(), 1)
    fracs = np.linspace(0, 1, n_points)
    k = np.round(fracs * len(y)).astype(int)
    return {"inspected_frac": fracs.tolist(), "caught_frac": caught[k].tolist()}


def caught_at_budget(y: np.ndarray, p: np.ndarray, budget: float) -> float:
    """Share of failures caught by inspecting the riskiest ``budget`` share of runs."""
    k = int(round(budget * len(y)))
    order = np.argsort(-p, kind="stable")
    return float(y[order[:k]].sum() / max(y.sum(), 1))


def evaluate_policy(
    y: np.ndarray, p: np.ndarray, cost_inspection: float, cost_missed: float
) -> dict[str, float]:
    """Return expected cost per run of the Bayes-threshold policy vs naive baselines."""
    threshold = cost_inspection / cost_missed
    inspect = p >= threshold
    rate = float(y.mean())
    budget = float(inspect.mean())
    missed = float((y.astype(bool) & ~inspect).sum()) / len(y)
    model_cost = cost_inspection * budget + cost_missed * missed
    costs = {
        "inspect_all": cost_inspection,
        "inspect_none": cost_missed * rate,
        "random_same_budget": cost_inspection * budget + cost_missed * rate * (1 - budget),
    }
    best_naive = min(costs["inspect_all"], costs["inspect_none"])
    return {
        "ratio": cost_missed / cost_inspection,
        "threshold": threshold,
        "inspected_frac": budget,
        "caught_frac": float(y[inspect].sum() / max(y.sum(), 1)),
        "cost_model": model_cost,
        **{f"cost_{k}": v for k, v in costs.items()},
        "savings_vs_best_naive": best_naive - model_cost,
        "savings_vs_random": costs["random_same_budget"] - model_cost,
    }


def policy_report(y: np.ndarray, p: np.ndarray, costs_cfg: dict[str, Any]) -> dict[str, Any]:
    """Gains curve, budget table, base policy and cost-ratio sensitivity."""
    ci = costs_cfg["cost_inspection"]
    return {
        "assumptions": __doc__.split("Assumptions (stated in the report):")[1].strip(),
        "gains": gains_curve(y, p),
        "budgets": [
            {
                "budget": b,
                "inspected": int(round(b * len(y))),
                "caught_frac": caught_at_budget(y, p, b),
                "caught": int(round(caught_at_budget(y, p, b) * y.sum())),
                "failures": int(y.sum()),
                "random_caught_frac": b,
            }
            for b in costs_cfg["inspection_budgets"]
        ],
        "base": evaluate_policy(y, p, ci, costs_cfg["cost_missed_defect"]),
        "sensitivity": [evaluate_policy(y, p, ci, ci * r) for r in costs_cfg["sensitivity_ratios"]],
    }
