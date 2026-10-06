import numpy as np
import pandas as pd
import pandera.errors as pe
import pytest

from processlens.data.contract import sensor_columns, validate


def test_valid_frame_passes(frame: pd.DataFrame, data_cfg: dict) -> None:
    out = validate(frame, data_cfg)
    assert len(out) == len(frame)
    assert len(sensor_columns(out)) == 5


def test_bad_label_fails(frame: pd.DataFrame, data_cfg: dict) -> None:
    frame.loc[0, "label"] = 2
    with pytest.raises(pe.SchemaErrors):
        validate(frame, data_cfg)


def test_unsorted_timestamp_fails(frame: pd.DataFrame, data_cfg: dict) -> None:
    bad = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(pe.SchemaErrors):
        validate(bad, data_cfg)


def test_null_timestamp_fails(frame: pd.DataFrame, data_cfg: dict) -> None:
    frame.loc[len(frame) - 1, "timestamp"] = pd.NaT
    with pytest.raises(pe.SchemaErrors):
        validate(frame, data_cfg)


def test_wrong_row_count_fails(frame: pd.DataFrame, data_cfg: dict) -> None:
    with pytest.raises(pe.SchemaErrors):
        validate(frame.iloc[:-1], data_cfg)


def test_non_numeric_sensor_fails(frame: pd.DataFrame, data_cfg: dict) -> None:
    frame["sensor_001"] = frame["sensor_001"].astype(str)
    with pytest.raises(pe.SchemaErrors):
        validate(frame, data_cfg)


def test_unknown_column_fails(frame: pd.DataFrame, data_cfg: dict) -> None:
    frame["extra"] = np.zeros(len(frame))
    with pytest.raises(pe.SchemaErrors):
        validate(frame, data_cfg)


def test_sensor_nans_allowed(frame: pd.DataFrame, data_cfg: dict) -> None:
    frame.loc[:5, "sensor_002"] = np.nan
    validate(frame, data_cfg)
