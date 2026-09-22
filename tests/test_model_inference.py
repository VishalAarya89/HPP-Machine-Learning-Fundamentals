"""Unit and integration tests for model_inference.py

These tests require a trained model at models/house_price_model.pkl.
Run `python src/model_training.py` first if it doesn't exist.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent / "src"))
from model_inference import HousePriceModel, InputValidationError  # noqa: E402

MODEL_PATH = Path(__file__).parent.parent / "models" / "house_price_model.pkl"

VALID_PAYLOAD = {
    "area": 1200,
    "bedrooms": 3,
    "bathrooms": 2,
    "age": 5,
    "location": "City Center",
    "property_type": "Apartment",
}


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("Trained model not found; run src/model_training.py first")
    return HousePriceModel()


def test_predict_returns_positive_price(model):
    result = model.predict(VALID_PAYLOAD)
    assert result["prediction"] > 0


def test_predict_returns_confidence_interval(model):
    result = model.predict(VALID_PAYLOAD)
    assert result["confidence_interval_low"] <= result["prediction"] <= result["confidence_interval_high"]


def test_validate_input_rejects_missing_field():
    payload = dict(VALID_PAYLOAD)
    del payload["area"]
    with pytest.raises(InputValidationError):
        HousePriceModel.validate_input(payload)


def test_validate_input_rejects_negative_area():
    payload = dict(VALID_PAYLOAD, area=-10)
    with pytest.raises(InputValidationError):
        HousePriceModel.validate_input(payload)


def test_validate_input_rejects_bad_location():
    payload = dict(VALID_PAYLOAD, location="Atlantis")
    with pytest.raises(InputValidationError):
        HousePriceModel.validate_input(payload)


def test_validate_input_rejects_bad_property_type():
    payload = dict(VALID_PAYLOAD, property_type="Castle")
    with pytest.raises(InputValidationError):
        HousePriceModel.validate_input(payload)


def test_validate_input_rejects_non_numeric_area():
    payload = dict(VALID_PAYLOAD, area="not-a-number")
    with pytest.raises(InputValidationError):
        HousePriceModel.validate_input(payload)


def test_validate_input_accepts_valid_payload():
    cleaned = HousePriceModel.validate_input(VALID_PAYLOAD)
    assert cleaned["Area"] == 1200
    assert cleaned["Location"] == "City Center"


def test_predict_larger_area_gives_higher_price(model):
    small = model.predict(dict(VALID_PAYLOAD, area=800))
    large = model.predict(dict(VALID_PAYLOAD, area=4000))
    assert large["prediction"] > small["prediction"]
