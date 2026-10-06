import numpy as np
import pandas as pd
import pytest

from processlens.data.ingest import canonicalize, sensor_name


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Small unsorted raw-style frame: labels in {-1, 1}, 5 sensors."""
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({sensor_name(i + 1): rng.normal(size=n) for i in range(5)})
    df[sensor_name(5)] = 3.0  # constant sensor
    df.loc[:9, sensor_name(4)] = np.nan
    meta = pd.DataFrame(
        {
            "label": np.where(np.arange(n) % 5 == 0, 1, -1),
            "timestamp": pd.Timestamp("2008-07-19") + pd.to_timedelta(rng.permutation(n), "h"),
        }
    )
    return pd.concat([meta, df], axis=1)


@pytest.fixture
def frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    return canonicalize(raw_frame)


@pytest.fixture
def data_cfg(frame: pd.DataFrame) -> dict:
    return {"contract": {"expected_rows": len(frame), "label_values": [0, 1]}}
