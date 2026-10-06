"""Time-ordered train / validation / test split and rolling-origin CV folds."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from processlens.data.ingest import TIMESTAMP


@dataclass(frozen=True)
class TimeSplit:
    """Row-index arrays for each time-ordered segment."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def _boundary(ts: pd.Series, target: int) -> int:
    """Move ``target`` forward so it never splits rows sharing a timestamp."""
    while 0 < target < len(ts) and ts.iloc[target] == ts.iloc[target - 1]:
        target += 1
    return target


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> TimeSplit:
    """Split a time-sorted frame into consecutive train / val / test blocks.

    Every validation timestamp is strictly after every training timestamp, and
    every test timestamp strictly after every validation timestamp.
    """
    ts = df[TIMESTAMP].reset_index(drop=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("Frame must be sorted by timestamp")
    n = len(ts)
    a = _boundary(ts, int(round(train_frac * n)))
    b = _boundary(ts, int(round((train_frac + val_frac) * n)))
    if not 0 < a < b < n:
        raise ValueError(f"Degenerate split boundaries: {a}, {b}, n={n}")
    idx = np.arange(n)
    return TimeSplit(train=idx[:a], val=idx[a:b], test=idx[b:])


def rolling_origin_folds(
    n: int, n_folds: int, min_train_frac: float = 0.4
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield expanding-window (fit, eval) index pairs over ``n`` time-ordered rows.

    The first fit window covers ``min_train_frac`` of the rows; the remainder is
    cut into ``n_folds`` consecutive evaluation blocks. Each fold fits on all rows
    before its evaluation block.
    """
    start = int(round(min_train_frac * n))
    edges = np.linspace(start, n, n_folds + 1).round().astype(int)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        yield np.arange(lo), np.arange(lo, hi)
