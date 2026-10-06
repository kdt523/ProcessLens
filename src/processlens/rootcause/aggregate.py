"""Combine method rankings into a consensus, roll up to clusters, grade the evidence."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

METHOD_RANKS = {
    "univariate": ("p", True),  # column, ascending
    "stability": ("stability", False),
    "shap": ("shap", False),
}


def method_ranks(table: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Rank sensors per method (1 = most suspicious; ties share the average rank; NaN last)."""
    out = pd.DataFrame(index=table.index)
    for m in methods:
        col, asc = METHOD_RANKS[m]
        out[f"rank_{m}"] = table[col].rank(ascending=asc, method="average", na_option="bottom")
    return out


def consensus_score(ranks: pd.DataFrame, method: str) -> pd.Series:
    """Mean reciprocal rank (``mrr``) or normalised Borda count (``borda``); higher = stronger."""
    if method == "mrr":
        return (1.0 / ranks).mean(axis=1)
    if method == "borda":
        return (1.0 - (ranks - 1) / len(ranks)).mean(axis=1)
    raise ValueError(f"Unknown consensus method {method!r}")


def evidence_strength(q: float, effect: float, stability: float, rules: dict[str, Any]) -> str:
    """Grade evidence ``strong`` / ``moderate`` / ``weak`` using the METHODS.md rules."""
    for level in ("strong", "moderate"):
        r = rules[level]
        if (
            not np.isnan(q)
            and q <= r["max_q"]
            and abs(effect) >= r["min_abs_effect"]
            and stability >= r["min_stability"]
        ):
            return level
    return "weak"


def cluster_table(
    sensors: pd.DataFrame, clusters: pd.Series, rules: dict[str, Any]
) -> pd.DataFrame:
    """Roll sensor scores up to clusters; the best-scoring member is the representative.

    Cluster evidence uses the strongest value among members for each statistic.
    """
    df = sensors.join(clusters.rename("cluster"), how="left")
    rows = []
    for cid, g in df.groupby("cluster", sort=False):
        g = g.sort_values("score", ascending=False)
        rep = g.index[0]
        best_q = float(g["q"].min()) if g["q"].notna().any() else float("nan")
        eff = g["effect"].dropna()
        effect = float(eff.loc[eff.abs().idxmax()]) if len(eff) else float("nan")
        stab = float(g["stability"].max()) if "stability" in g else 0.0
        rows.append(
            {
                "cluster": int(cid),
                "representative": rep,
                "members": list(g.index),
                "n_members": len(g),
                "score": float(g["score"].iloc[0]),
                "q_value": best_q,
                "effect_size": effect,
                "stability_freq": stab,
                "shap_share": float(g["shap_share"].sum()) if "shap_share" in g else 0.0,
                "evidence": evidence_strength(best_q, effect, stab, rules),
            }
        )
    out = pd.DataFrame(rows).sort_values("score", ascending=False, kind="stable")
    out.insert(0, "consensus_rank", np.arange(1, len(out) + 1))
    return out.reset_index(drop=True)
