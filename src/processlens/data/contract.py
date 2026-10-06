"""Pandera data contract for the processed SECOM table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa

from processlens.config import PROJECT_ROOT, load_config
from processlens.data.ingest import LABEL, RUN_ID, SENSOR_PREFIX, TIMESTAMP


def build_schema(expected_rows: int | None, label_values: list[int]) -> pa.DataFrameSchema:
    """Return the contract: numeric sensors, binary label, monotonic non-null timestamp."""
    checks = [
        pa.Check(lambda df: df[TIMESTAMP].is_monotonic_increasing, error="timestamp not sorted"),
        pa.Check(
            lambda df: sum(c.startswith(SENSOR_PREFIX) for c in df.columns) > 0,
            error="no sensor columns",
        ),
    ]
    if expected_rows is not None:
        checks.append(pa.Check(lambda df: len(df) == expected_rows, error="unexpected row count"))
    return pa.DataFrameSchema(
        columns={
            RUN_ID: pa.Column("int64", unique=True, nullable=False),
            LABEL: pa.Column("int64", pa.Check.isin(label_values), nullable=False),
            TIMESTAMP: pa.Column("datetime64[ns]", nullable=False),
            rf"^{SENSOR_PREFIX}\d{{3}}$": pa.Column("float64", nullable=True, regex=True),
        },
        checks=checks,
        strict=True,
        coerce=False,
    )


def validate(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Validate ``df`` against the contract and return it (raises on violation)."""
    contract = (cfg or load_config("data"))["contract"]
    schema = build_schema(contract["expected_rows"], contract["label_values"])
    return schema.validate(df, lazy=True)


def load_processed(cfg: dict[str, Any] | None = None, root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Load the processed parquet and validate it against the contract."""
    cfg = cfg or load_config("data")
    df = pd.read_parquet(root / cfg["paths"]["processed"])
    return validate(df, cfg)


def sensor_columns(df: pd.DataFrame) -> list[str]:
    """Return the sensor column names in order."""
    return [c for c in df.columns if c.startswith(SENSOR_PREFIX)]
