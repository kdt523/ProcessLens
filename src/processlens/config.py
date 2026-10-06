"""Load YAML configs from ``configs/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Return the parsed contents of ``configs/<name>.yaml``."""
    path = config_dir / f"{name}.yaml"
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data
