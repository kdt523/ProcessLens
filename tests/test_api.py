"""API tests through FastAPI's TestClient, with a tiny in-memory model bundle."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from processlens.api import main
from processlens.config import load_config
from processlens.features.pipeline import build_pipeline
from processlens.models.calibrate import calibrate

FEATURES = [f"sensor_{i:03d}" for i in range(1, 6)]


@pytest.fixture(scope="module")
def bundle() -> dict:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(300, 5)), columns=FEATURES)
    y = (x["sensor_002"] + rng.normal(size=300) > 1.5).astype(int).to_numpy()
    cfg = load_config("model")
    cfg["models"]["random_forest"]["n_estimators"] = 30
    pipe = build_pipeline("random_forest", cfg, 0).fit(x.iloc[:200], y[:200])
    cal = calibrate(pipe, x.iloc[200:], y[200:], "sigmoid")
    return {"model": cal, "pipeline": pipe, "name": "random_forest", "features": FEATURES}


@pytest.fixture
def client(monkeypatch, bundle) -> TestClient:
    monkeypatch.setattr(main, "load_model", lambda path=None: bundle)
    return TestClient(main.app)


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] and r.json()["model_name"] == "random_forest"


def test_score_returns_risk_and_contributors(client) -> None:
    runs = [{"sensor_001": 0.1, "sensor_002": 3.0, "sensor_003": None}, {"sensor_002": -2.0}]
    r = client.post("/score", json={"runs": runs, "top_n_contributors": 3})
    assert r.status_code == 200, r.text
    scores = r.json()["scores"]
    assert len(scores) == 2
    assert 0 <= scores[0]["risk"] <= 1
    assert scores[0]["risk"] > scores[1]["risk"]
    assert len(scores[0]["top_contributors"]) == 3
    assert scores[0]["top_contributors"][0]["sensor"] == "sensor_002"


def test_score_rejects_unknown_sensor(client) -> None:
    r = client.post("/score", json={"runs": [{"sensor_999": 1.0}]})
    assert r.status_code == 422
    assert "contract" in r.text


def test_score_rejects_non_numeric(client) -> None:
    r = client.post("/score", json={"runs": [{"sensor_001": "abc"}]})
    assert r.status_code == 422


def test_score_rejects_empty(client) -> None:
    assert client.post("/score", json={"runs": []}).status_code == 422


def test_rootcause_endpoint(client, monkeypatch) -> None:
    fake = {
        "note": "associations, not causes",
        "window": {"runs": 10},
        "evidence_counts": {"weak": 3},
        "suspects": [{"consensus_rank": 1}],
    }
    monkeypatch.setattr("processlens.rootcause.engine.run_rootcause", lambda *a, **k: fake)
    r = client.post("/rootcause", json={"top_k": 3})
    assert r.status_code == 200 and r.json()["suspects"][0]["consensus_rank"] == 1


def test_rootcause_bad_window_is_422(client, monkeypatch) -> None:
    def boom(*a, **k):
        raise ValueError("Window must contain both passing and failing runs")

    monkeypatch.setattr("processlens.rootcause.engine.run_rootcause", boom)
    assert client.post("/rootcause", json={}).status_code == 422


def test_drift_endpoint(client, monkeypatch) -> None:
    monkeypatch.setattr(
        "processlens.monitoring.drift.run_drift",
        lambda *a, **k: {"counts": {"ok": 1, "warn": 0, "alert": 0}},
    )
    r = client.get("/drift")
    assert r.status_code == 200 and r.json()["counts"]["ok"] == 1


def test_score_without_model_is_503(monkeypatch) -> None:
    def missing(path=None):
        raise FileNotFoundError("run make train")

    monkeypatch.setattr(main, "load_model", missing)
    r = TestClient(main.app).post("/score", json={"runs": [{"sensor_001": 1.0}]})
    assert r.status_code == 503
