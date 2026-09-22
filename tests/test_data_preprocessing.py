"""Unit tests for data_preprocessing.py"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent / "src"))
from data_preprocessing import (  # noqa: E402
    DataValidationError,
    clean_data,
    engineer_features,
    validate_schema,
)

VALID_ROW = {
    "Property_ID": "PROP0001",
    "Area": 1200,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Age": 5,
    "Location": "City Center",
    "Property_Type": "Apartment",
    "Price": 15000000,
}


def make_df(rows):
    return pd.DataFrame(rows)


def test_validate_schema_passes_for_valid_columns():
    df = make_df([VALID_ROW])
    validate_schema(df)  # should not raise


def test_validate_schema_raises_on_missing_column():
    df = make_df([VALID_ROW]).drop(columns=["Price"])
    with pytest.raises(DataValidationError):
        validate_schema(df)


def test_clean_data_drops_duplicates():
    df = make_df([VALID_ROW, VALID_ROW])
    cleaned = clean_data(df)
    assert len(cleaned) == 1


def test_clean_data_drops_negative_price():
    bad_row = dict(VALID_ROW, Property_ID="PROP0002", Price=-100)
    df = make_df([VALID_ROW, bad_row])
    cleaned = clean_data(df)
    assert len(cleaned) == 1
    assert (cleaned["Price"] > 0).all()


def test_clean_data_drops_zero_area():
    bad_row = dict(VALID_ROW, Property_ID="PROP0003", Area=0)
    df = make_df([VALID_ROW, bad_row])
    cleaned = clean_data(df)
    assert len(cleaned) == 1


def test_clean_data_drops_unrecognised_location():
    bad_row = dict(VALID_ROW, Property_ID="PROP0004", Location="Moon Base")
    df = make_df([VALID_ROW, bad_row])
    cleaned = clean_data(df)
    assert len(cleaned) == 1


def test_clean_data_imputes_missing_numeric_with_median():
    row2 = dict(VALID_ROW, Property_ID="PROP0005", Area=None)
    df = make_df([VALID_ROW, row2])
    cleaned = clean_data(df)
    assert len(cleaned) == 2
    assert cleaned["Area"].isnull().sum() == 0


def test_engineer_features_adds_expected_columns():
    df = make_df([VALID_ROW])
    engineered = engineer_features(df)
    for col in ["total_rooms", "bath_bed_ratio", "is_new", "area_per_room", "age_bucket"]:
        assert col in engineered.columns


def test_engineer_features_total_rooms_correct():
    df = make_df([VALID_ROW])
    engineered = engineer_features(df)
    assert engineered.loc[0, "total_rooms"] == VALID_ROW["Bedrooms"] + VALID_ROW["Bathrooms"]


def test_engineer_features_handles_zero_bedrooms_without_error():
    row = dict(VALID_ROW, Bedrooms=0, Bathrooms=0)
    df = make_df([row])
    engineered = engineer_features(df)
    assert engineered.loc[0, "bath_bed_ratio"] == 0
    assert engineered.loc[0, "area_per_room"] == row["Area"]
