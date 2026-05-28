from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

# Import the app but mock the model loading so tests don't need model.pkl at
# import time. The lifespan is bypassed by TestClient when used with
# lifespan=False or by patching app.state before the request.
import sys
import os

# Ensure project root is on path so schemas / routers are importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import app  # noqa: E402

VALID_SCENARIO = {
    "down": 3,
    "distance": 7,
    "yardline_100": 65,
    "quarter": 2,
    "seconds_remaining_quarter": 300,
    "score_differential": 0,
    "is_home_possession": True,
    "era_season": 2024,
    "era_week": 8,
}


def _make_mock_model(proba=0.65):
    """Return a mock scikit-learn-style classifier."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[1 - proba, proba]])
    mock_model.n_features_in_ = 11
    return mock_model


def _client_with_model(model=None):
    """Create a TestClient with a pre-injected mock model on app.state."""
    if model is None:
        model = _make_mock_model()
    app.state.model = model
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# 1. Health endpoint
def test_health_ok():
    client = _client_with_model()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# 2. Valid request returns 200
def test_predict_valid_request_returns_200():
    client = _client_with_model()
    resp = client.post("/predict/scenario", json=VALID_SCENARIO)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 3. win_probability is in [0.0, 1.0]
def test_predict_win_probability_in_range():
    client = _client_with_model(_make_mock_model(proba=0.72))
    resp = client.post("/predict/scenario", json=VALID_SCENARIO)
    assert resp.status_code == 200
    wp = resp.json()["win_probability"]
    assert 0.0 <= wp <= 1.0, f"win_probability out of range: {wp}"


# ---------------------------------------------------------------------------
# 4. Response contains ot_era field
def test_predict_response_has_ot_era():
    client = _client_with_model()
    resp = client.post("/predict/scenario", json=VALID_SCENARIO)
    assert resp.status_code == 200
    assert "ot_era" in resp.json(), "Response must include 'ot_era' field"


# ---------------------------------------------------------------------------
# 5. down=0 is rejected with 422
def test_predict_down_zero_rejected():
    client = _client_with_model()
    bad = {**VALID_SCENARIO, "down": 0}
    resp = client.post("/predict/scenario", json=bad)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 6. Missing required field is rejected with 422
def test_predict_missing_field_rejected():
    client = _client_with_model()
    bad = {k: v for k, v in VALID_SCENARIO.items() if k != "score_differential"}
    resp = client.post("/predict/scenario", json=bad)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 7. Scenario is echoed in response
def test_predict_scenario_echoed_in_response():
    client = _client_with_model()
    resp = client.post("/predict/scenario", json=VALID_SCENARIO)
    assert resp.status_code == 200
    echoed = resp.json()["scenario"]
    assert echoed["down"] == VALID_SCENARIO["down"]
    assert echoed["era_season"] == VALID_SCENARIO["era_season"]
    assert echoed["is_home_possession"] == VALID_SCENARIO["is_home_possession"]
