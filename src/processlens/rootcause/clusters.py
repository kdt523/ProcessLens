"""Hierarchical clustering of sensors on 1 − |Spearman ρ|."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from processlens.stats import spearman_matrix


def sensor_clusters(x: pd.DataFrame, min_abs_rho: float) -> pd.Series:
    """Return a cluster id per sensor; sensors in one cluster have average |ρ| ≥ ``min_abs_rho``.

    Average linkage on 1 − |ρ|, cut at distance 1 − ``min_abs_rho``. Pairs with no
    overlapping observations count as uncorrelated. Ids are ordered by first sensor.
    """
    if x.shape[1] == 1:
        return pd.Series([0], index=x.columns, name="cluster")
    corr = spearman_matrix(x).abs().to_numpy()
    dist = 1.0 - np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2, 0.0, 1.0)
    raw = fcluster(
        linkage(squareform(dist, checks=False), method="average"),
        t=1.0 - min_abs_rho,
        criterion="distance",
    )
    _, ids = np.unique(raw, return_index=False, return_inverse=True)
    first: dict[int, int] = {}
    renum = [first.setdefault(i, len(first)) for i in ids]
    return pd.Series(renum, index=x.columns, name="cluster")
