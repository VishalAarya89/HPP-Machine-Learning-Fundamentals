"""Integration tests for the Flask web app (routes, API, error handling)."""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent / "app"))
sys.path.append(str(Path(__file__).parent.parent / "src"))

MODEL_PATH = Path(__file__).parent.parent / "models" / "house_price_model.pkl"


@pytest.fixture()
def client():
    if not MODEL_PATH.exists():
        pytest.skip("Trained model not found; run src/model_training.py first")
    from web_app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Property Valuation Desk" in resp.data


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "healthy"


def test_api_predict_valid_payload(client):
    payload = {
        "area": 1200, "bedrooms": 3, "bathrooms": 2,
        "age": 5, "location": "City Center", "property_type": "Apartment",
    }
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["data"]["prediction"] > 0


def test_api_predict_missing_field(client):
    payload = {"area": 1200, "bedrooms": 3}
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 400
    assert resp.get_json()["success"] is False


def test_api_predict_invalid_json(client):
    resp = client.post("/api/predict", data="not json", content_type="application/json")
    assert resp.status_code == 400


def test_form_predict_renders_result(client):
    payload = {
        "area": "1200", "bedrooms": "3", "bathrooms": "2",
        "age": "5", "location": "City Center", "property_type": "Apartment",
    }
    resp = client.post("/predict", data=payload)
    assert resp.status_code == 200
    assert b"Estimate" in resp.data


def test_form_predict_shows_error_on_bad_input(client):
    payload = {
        "area": "-5", "bedrooms": "3", "bathrooms": "2",
        "age": "5", "location": "City Center", "property_type": "Apartment",
    }
    resp = client.post("/predict", data=payload)
    assert resp.status_code == 200
    assert b"Area must be" in resp.data
