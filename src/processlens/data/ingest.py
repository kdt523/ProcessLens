"""Download UCI SECOM, record raw checksums and build the processed parquet."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from processlens.config import PROJECT_ROOT, load_config

log = logging.getLogger(__name__)

UCIML_RAW = "secom_ucimlrepo.csv"
ZIP_FEATURES = "secom.data"
ZIP_LABELS = "secom_labels.data"
SENSOR_PREFIX = "sensor_"
LABEL = "label"
TIMESTAMP = "timestamp"
RUN_ID = "run_id"


def sensor_name(i: int) -> str:
    """Return the canonical column name for UCI attribute number ``i`` (1-based)."""
    return f"{SENSOR_PREFIX}{i:03d}"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_via_ucimlrepo(raw_dir: Path, uci_id: int) -> list[Path]:
    """Fetch SECOM with ``ucimlrepo`` and store the original table as CSV."""
    from ucimlrepo import fetch_ucirepo

    original = fetch_ucirepo(id=uci_id).data.original
    path = raw_dir / UCIML_RAW
    original.to_csv(path, index=False)
    return [path]


def fetch_via_zip(raw_dir: Path, url: str) -> list[Path]:
    """Download the SECOM zip and extract its files into ``raw_dir``."""
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed https URL)
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(raw_dir)
        return [raw_dir / name for name in zf.namelist()]


def fetch_raw(raw_dir: Path, uci_id: int, zip_url: str) -> list[Path]:
    """Fetch raw files (ucimlrepo first, zip fallback) unless already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for existing in ([raw_dir / UCIML_RAW], [raw_dir / ZIP_FEATURES, raw_dir / ZIP_LABELS]):
        if all(p.exists() for p in existing):
            log.info("Using cached raw files in %s", raw_dir)
            return existing
    try:
        return fetch_via_ucimlrepo(raw_dir, uci_id)
    except Exception as exc:  # network / API failures: fall back to the static zip
        log.warning("ucimlrepo failed (%s); falling back to zip", exc)
        return fetch_via_zip(raw_dir, zip_url)


def _parse_timestamps(values: pd.Series, dayfirst: bool) -> pd.Series:
    return pd.to_datetime(values.astype(str).str.strip('"'), dayfirst=dayfirst, format="mixed")


def parse_ucimlrepo_csv(path: Path, dayfirst: bool) -> pd.DataFrame:
    """Parse the ucimlrepo CSV into the canonical frame (unsorted)."""
    raw = pd.read_csv(path)
    sensors = raw.drop(columns=["class", "timestamp"])
    sensors.columns = [sensor_name(int(c.split()[-1])) for c in sensors.columns]
    meta = pd.DataFrame(
        {LABEL: raw["class"], TIMESTAMP: _parse_timestamps(raw["timestamp"], dayfirst)}
    )
    return pd.concat([meta, sensors], axis=1)


def parse_zip_files(features: Path, labels: Path, dayfirst: bool) -> pd.DataFrame:
    """Parse ``secom.data`` + ``secom_labels.data`` into the canonical frame (unsorted)."""
    x = pd.read_csv(features, sep=r"\s+", header=None, na_values=["NaN"])
    x.columns = [sensor_name(i + 1) for i in range(x.shape[1])]
    y = pd.read_csv(labels, sep=" ", header=None, names=["class", "ts"], quotechar='"')
    if len(x) != len(y):
        raise ValueError(f"Feature rows ({len(x)}) != label rows ({len(y)})")
    meta = pd.DataFrame({LABEL: y["class"], TIMESTAMP: _parse_timestamps(y["ts"], dayfirst)})
    return pd.concat([meta, x], axis=1)


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map labels to 0/1, keep the original row order as ``run_id`` and sort by time."""
    if not set(df[LABEL].unique()) <= {-1, 1}:
        raise ValueError(f"Unexpected raw label values: {sorted(df[LABEL].unique())}")
    out = pd.concat([pd.Series(range(len(df)), name=RUN_ID, dtype="int64"), df], axis=1)
    out[LABEL] = (out[LABEL] == 1).astype("int64")
    out = out.sort_values([TIMESTAMP, RUN_ID], kind="stable").reset_index(drop=True)
    sensor_cols = [c for c in out.columns if c.startswith(SENSOR_PREFIX)]
    out[sensor_cols] = out[sensor_cols].astype("float64")
    out[TIMESTAMP] = out[TIMESTAMP].astype("datetime64[ns]")
    if not out[TIMESTAMP].is_monotonic_increasing:
        raise AssertionError("Timestamps are not monotonic after sorting")
    return out


def build_frame(raw_files: list[Path], dayfirst: bool) -> pd.DataFrame:
    """Parse whichever raw format was fetched and return the canonical frame."""
    names = {p.name: p for p in raw_files}
    if UCIML_RAW in names:
        df = parse_ucimlrepo_csv(names[UCIML_RAW], dayfirst)
    else:
        df = parse_zip_files(names[ZIP_FEATURES], names[ZIP_LABELS], dayfirst)
    return canonicalize(df)


def run_ingest(cfg: dict[str, Any] | None = None, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Fetch, checksum, parse, validate and write SECOM; return a summary."""
    from processlens.data.contract import validate

    cfg = cfg or load_config("data")
    paths = cfg["paths"]
    raw_files = fetch_raw(
        root / paths["raw_dir"], cfg["source"]["uci_id"], cfg["source"]["zip_url"]
    )
    checksums = {p.name: sha256_file(p) for p in raw_files}
    (root / paths["checksums"]).write_text(json.dumps(checksums, indent=2), encoding="utf-8")

    df = validate(build_frame(raw_files, cfg["contract"]["timestamp_dayfirst"]), cfg)
    out = root / paths["processed"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    n_sensors = sum(c.startswith(SENSOR_PREFIX) for c in df.columns)
    summary = {
        "rows": len(df),
        "sensors": n_sensors,
        "failures": int(df[LABEL].sum()),
        "raw_files": checksums,
        "processed": out.relative_to(root).as_posix(),
        "processed_sha256": sha256_file(out),
    }
    log.info("Ingested %s", summary)
    return summary
