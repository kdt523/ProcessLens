"""Smoke-test every dashboard page headlessly on the cached artifacts in reports/."""

import sys
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
METRICS = APP.parent / "reports" / "metrics"
PAGES = ["overview", "model", "rootcause", "benchmark", "copilot"]
NEEDS = {
    "overview": "audit",
    "model": "model",
    "rootcause": "rootcause",
    "benchmark": "benchmark",
    "copilot": None,
}


@pytest.fixture(autouse=True)
def _app_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    yield
    sys.modules.pop("common", None)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_error(page: str) -> None:
    need = NEEDS[page]
    if need and not (METRICS / f"{need}.json").exists():
        pytest.skip(f"{need}.json not generated")
    t0 = time.perf_counter()
    at = AppTest.from_file(str(APP / "app_pages" / f"{page}.py"), default_timeout=60).run()
    elapsed = time.perf_counter() - t0
    assert not at.exception, at.exception
    assert elapsed < 30  # generous for CI; local cold load is measured separately


def test_cost_slider_changes_threshold() -> None:
    if not (METRICS / "model.json").exists():
        pytest.skip("model.json not generated")
    at = AppTest.from_file(str(APP / "app_pages" / "model.py"), default_timeout=60).run()
    before = [m.value for m in at.metric]
    at.slider[0].set_value(40).run()
    after = [m.value for m in at.metric]
    assert not at.exception and before != after
