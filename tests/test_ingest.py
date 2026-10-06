from pathlib import Path

import pandas as pd
import pytest

from processlens.data.ingest import (
    canonicalize,
    parse_ucimlrepo_csv,
    parse_zip_files,
    sensor_name,
    sha256_file,
)


def test_sensor_name() -> None:
    assert sensor_name(1) == "sensor_001"
    assert sensor_name(590) == "sensor_590"


def test_canonicalize_maps_labels_and_sorts(raw_frame: pd.DataFrame) -> None:
    out = canonicalize(raw_frame)
    assert set(out["label"].unique()) == {0, 1}
    assert out["label"].sum() == (raw_frame["label"] == 1).sum()
    assert out["timestamp"].is_monotonic_increasing
    # run_id is the original row number, so each row keeps its own label and sensors
    back = out.set_index("run_id").sort_index()
    assert (back["label"].to_numpy() == (raw_frame["label"] == 1).to_numpy()).all()
    assert (back["sensor_001"].to_numpy() == raw_frame["sensor_001"].to_numpy()).all()


def test_canonicalize_rejects_unknown_labels(raw_frame: pd.DataFrame) -> None:
    raw_frame.loc[0, "label"] = 0
    with pytest.raises(ValueError):
        canonicalize(raw_frame)


def test_parse_zip_files(tmp_path: Path) -> None:
    (tmp_path / "secom.data").write_text("1.0 NaN 3\n4 5 6\n")
    (tmp_path / "secom_labels.data").write_text(
        '-1 "19/07/2008 11:55:00"\n1 "02/08/2008 13:17:00"\n'
    )
    df = parse_zip_files(tmp_path / "secom.data", tmp_path / "secom_labels.data", dayfirst=True)
    assert list(df.columns) == ["label", "timestamp", "sensor_001", "sensor_002", "sensor_003"]
    assert pd.isna(df.loc[0, "sensor_002"])
    assert df.loc[1, "timestamp"] == pd.Timestamp("2008-08-02 13:17:00")  # day first


def test_parse_ucimlrepo_csv(tmp_path: Path) -> None:
    path = tmp_path / "u.csv"
    path.write_text(
        "class,timestamp,Attribute 1,Attribute 2\n"
        "-1,19/07/2008 11:55:00,1.0,\n1,02/08/2008 13:17:00,2.0,3.0\n"
    )
    df = parse_ucimlrepo_csv(path, dayfirst=True)
    assert list(df.columns) == ["label", "timestamp", "sensor_001", "sensor_002"]
    assert df.loc[1, "timestamp"].month == 8


def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_bytes(b"abc")
    assert sha256_file(p).startswith("ba7816bf")
