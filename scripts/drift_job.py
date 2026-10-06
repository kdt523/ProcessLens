"""Scheduled drift job: compare the training window with the last N days and log alerts.

Schedule with cron / Windows Task Scheduler, e.g. daily:
    uv run python scripts/drift_job.py --days 7
Exits with code 2 when any sensor is at alert level, so a scheduler can notify.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from processlens.config import load_config
from processlens.data.contract import load_processed
from processlens.data.ingest import TIMESTAMP
from processlens.monitoring.drift import run_drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, help="size of the recent window")
    args = parser.parse_args()
    end = load_processed(load_config("data"))[TIMESTAMP].max()
    start = end - pd.Timedelta(days=args.days)
    res = run_drift(str(start), str(end))
    print(json.dumps({"current": res["current"], "counts": res["counts"]}, indent=2))
    return 2 if res["counts"]["alert"] else 0


if __name__ == "__main__":
    sys.exit(main())
