"""Shared loaders and chart styling for the dashboard. Reads only reports/ (and the API)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
MUTED = "#898781"


@st.cache_data(ttl=300, max_entries=16)
def load_json(name: str) -> dict[str, Any] | None:
    """Load reports/metrics/<name>.json, or None if that step has not been run yet."""
    path = METRICS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(ttl=300, max_entries=4)
def load_text(rel: str) -> str | None:
    """Load a text file relative to the project root."""
    path = ROOT / rel
    return path.read_text(encoding="utf-8") if path.exists() else None


def require(name: str, command: str) -> dict[str, Any]:
    """Return a metrics file or stop the page with instructions."""
    data = load_json(name)
    if data is None:
        st.info(
            f"`reports/metrics/{name}.json` not found. Run `{command}` first.",
            icon=":material/info:",
        )
        st.stop()
    assert data is not None
    return data


def style(fig: go.Figure, height: int = 360, **layout: Any) -> go.Figure:
    """Recessive axes, light grid, consistent margins."""
    defaults: dict[str, Any] = {
        "height": height,
        "margin": {"l": 10, "r": 10, "t": 30, "b": 10},
        "legend": {"orientation": "h", "y": -0.2},
        "hovermode": "x unified",
    }
    fig.update_layout(**{**defaults, **layout})
    fig.update_xaxes(showgrid=False, linecolor=MUTED)
    fig.update_yaxes(gridcolor="rgba(137,135,129,0.2)", zeroline=False)
    return fig


def pct(v: float | None, digits: int = 1) -> str:
    """Format a share as a percentage."""
    return "n/a" if v is None else f"{v * 100:.{digits}f}%"
