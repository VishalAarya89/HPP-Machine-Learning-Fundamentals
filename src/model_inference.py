"""
model_inference.py
--------------------
Loads the persisted model and exposes a clean predict() function with
input validation, used by both the web app and the test suite.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from data_preprocessing import (
    VALID_LOCATIONS,
    VALID_PROPERTY_TYPES,
    engineer_features,
)

MODEL_PATH = Path(__file__).parent.parent / "models" / "house_price_model.pkl"


class InputValidationError(ValueError):
    """Raised when user-supplied prediction input is invalid."""


class HousePriceModel:
    """Thin wrapper around the persisted sklearn pipeline for safe inference."""

    def __init__(self, model_path: str | Path = MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run `python src/model_training.py` first."
            )
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.pipeline = bundle["pipeline"]
        self.model_name = bundle["model_name"]
        self.numeric_features = bundle["numeric_features"]
        self.categorical_features = bundle["categorical_features"]
        self.metrics = bundle["metrics"]
        self.version = bundle.get("version", "unknown")

    @staticmethod
    def validate_input(payload: dict) -> dict:
        """Validate and coerce a raw input dict. Raises InputValidationError on failure."""
        required = ["area", "bedrooms", "bathrooms", "age", "location", "property_type"]
        missing = [k for k in required if k not in payload or payload[k] in (None, "")]
        if missing:
            raise InputValidationError(f"Missing required field(s): {', '.join(missing)}")

        try:
            area = float(payload["area"])
            bedrooms = int(payload["bedrooms"])
            bathrooms = int(payload["bathrooms"])
            age = int(payload["age"])
        except (TypeError, ValueError) as e:
            raise InputValidationError(f"Numeric fields must be valid numbers: {e}")

        if area <= 0 or area > 100000:
            raise InputValidationError("Area must be a positive number (up to 100,000 sqft).")
        if bedrooms < 0 or bedrooms > 20:
            raise InputValidationError("Bedrooms must be between 0 and 20.")
        if bathrooms < 0 or bathrooms > 20:
            raise InputValidationError("Bathrooms must be between 0 and 20.")
        if age < 0 or age > 150:
            raise InputValidationError("Age must be between 0 and 150 years.")

        location = str(payload["location"]).strip()
        property_type = str(payload["property_type"]).strip()
        if location not in VALID_LOCATIONS:
            raise InputValidationError(f"Location must be one of: {sorted(VALID_LOCATIONS)}")
        if property_type not in VALID_PROPERTY_TYPES:
            raise InputValidationError(f"Property type must be one of: {sorted(VALID_PROPERTY_TYPES)}")

        return {
            "Area": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Age": age,
            "Location": location,
            "Property_Type": property_type,
        }

    def predict(self, payload: dict) -> dict:
        """
        Predict price for a single property.
        payload: dict with keys area, bedrooms, bathrooms, age, location, property_type
        Returns dict with prediction and a rough confidence interval based on model MAE.
        """
        clean = self.validate_input(payload)
        row = pd.DataFrame([clean])
        row = engineer_features(row)

        X = row[self.numeric_features + self.categorical_features]
        prediction = float(self.pipeline.predict(X)[0])

        mae = self.metrics.get("mae", 0)
        return {
            "prediction": round(prediction, 2),
            "confidence_interval_low": round(max(prediction - mae, 0), 2),
            "confidence_interval_high": round(prediction + mae, 2),
            "model_used": self.model_name,
            "model_version": self.version,
        }


if __name__ == "__main__":
    model = HousePriceModel()
    sample = {
        "area": 1200,
        "bedrooms": 3,
        "bathrooms": 2,
        "age": 5,
        "location": "City Center",
        "property_type": "Apartment",
    }
    result = model.predict(sample)
    print(result)
