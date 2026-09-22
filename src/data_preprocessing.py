"""
data_preprocessing.py
----------------------
Data loading, cleaning, and feature engineering for the House Price
Prediction project.

This module is deliberately kept independent of any specific model so it
can be reused by training, evaluation, and inference code alike.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Columns expected in the raw dataset
RAW_NUMERIC_COLS = ["Area", "Bedrooms", "Bathrooms", "Age"]
RAW_CATEGORICAL_COLS = ["Location", "Property_Type"]
TARGET_COL = "Price"
ID_COL = "Property_ID"

VALID_LOCATIONS = {"City Center", "Suburb", "Rural"}
VALID_PROPERTY_TYPES = {"House", "Apartment", "Villa"}


class DataValidationError(ValueError):
    """Raised when incoming data fails a validation/sanity check."""


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info("Loaded raw data: %s rows, %s columns", len(df), df.shape[1])
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Ensure the dataframe has the columns we expect, before anything else runs."""
    required = set(RAW_NUMERIC_COLS + RAW_CATEGORICAL_COLS + [TARGET_COL])
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
      - drop exact duplicate rows
      - coerce numeric columns to numeric dtype (invalid -> NaN -> dropped)
      - drop rows with impossible/nonsensical values
      - impute any remaining missing numeric values with the column median
      - standardise text categories (strip whitespace, title case)
    """
    df = df.copy()
    validate_schema(df)

    before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %s duplicate rows", before - len(df))

    # Coerce numerics
    for col in RAW_NUMERIC_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardise categorical text
    for col in RAW_CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip()

    # Impute missing numeric values with the column median BEFORE filtering,
    # otherwise a NaN (e.g. from a blank cell) would be dropped by the
    # impossible-value filter below rather than imputed.
    for col in RAW_NUMERIC_COLS + [TARGET_COL]:
        if df[col].isnull().any():
            median = df[col].median()
            n_missing = df[col].isnull().sum()
            df[col] = df[col].fillna(median)
            logger.info("Imputed %s missing values in '%s' with median=%.2f", n_missing, col, median)

    # Remove physically impossible rows (negative/zero area or price, absurd age)
    before = len(df)
    df = df[
        (df["Area"] > 0)
        & (df["Price"] > 0)
        & (df["Bedrooms"] >= 0)
        & (df["Bathrooms"] >= 0)
        & (df["Age"] >= 0)
        & (df["Age"] <= 150)
    ]
    logger.info("Dropped %s rows with impossible values", before - len(df))

    # Drop rows with unrecognised categories rather than silently guessing
    before = len(df)
    df = df[df["Location"].isin(VALID_LOCATIONS) & df["Property_Type"].isin(VALID_PROPERTY_TYPES)]
    logger.info("Dropped %s rows with unrecognised category values", before - len(df))

    df = df.reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features on top of the cleaned raw columns.

    New features:
      - price_per_sqft is NOT created here (it would leak the target); it's
        only ever computed on training data for reporting purposes elsewhere.
      - total_rooms = Bedrooms + Bathrooms
      - bath_bed_ratio = Bathrooms / Bedrooms (guarded against divide-by-zero)
      - is_new = Age <= 5 (flag for a "new build" premium)
      - area_per_room = Area / total_rooms
      - age_bucket = categorical bucket of Age
    """
    df = df.copy()
    df["total_rooms"] = df["Bedrooms"] + df["Bathrooms"]
    df["bath_bed_ratio"] = df["Bathrooms"] / df["Bedrooms"].replace(0, np.nan)
    df["bath_bed_ratio"] = df["bath_bed_ratio"].fillna(0)
    df["is_new"] = (df["Age"] <= 5).astype(int)
    df["area_per_room"] = df["Area"] / df["total_rooms"].replace(0, np.nan)
    df["area_per_room"] = df["area_per_room"].fillna(df["Area"])
    df["age_bucket"] = pd.cut(
        df["Age"],
        bins=[-1, 5, 15, 30, 1000],
        labels=["0-5", "6-15", "16-30", "30+"],
    ).astype(str)
    return df


def get_feature_lists() -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) used for modeling."""
    numeric_features = [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Age",
        "total_rooms",
        "bath_bed_ratio",
        "is_new",
        "area_per_room",
    ]
    categorical_features = ["Location", "Property_Type", "age_bucket"]
    return numeric_features, categorical_features


def prepare_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Full pipeline: load -> clean -> engineer features. Returns model-ready df."""
    df = load_raw_data(csv_path)
    df = clean_data(df)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    data = prepare_dataset("data/house_prices.csv")
    print(data.head())
    print(f"\nFinal dataset shape: {data.shape}")
