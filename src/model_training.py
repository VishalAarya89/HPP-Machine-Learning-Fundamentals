"""
model_training.py
-------------------
Trains, tunes, and compares multiple regression algorithms for house
price prediction, then persists the best pipeline to disk.

Algorithms compared:
  1. Linear Regression   (simple, interpretable baseline)
  2. Random Forest       (bagged trees, captures non-linearities)
  3. Gradient Boosting   (boosted trees, usually strongest tabular performer)

Run directly with:  python src/model_training.py
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).parent))
from data_preprocessing import get_feature_lists, prepare_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
MODELS_DIR = Path(__file__).parent.parent / "models"
DOCS_DIR = Path(__file__).parent.parent / "docs"


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def get_candidate_models() -> dict:
    """Return a dict of {name: (estimator, param_grid)} for GridSearchCV."""
    return {
        "linear_regression": (
            LinearRegression(),
            {},  # no hyperparameters to tune
        ),
        "random_forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "regressor__n_estimators": [100, 200],
                "regressor__max_depth": [None, 10, 20],
                "regressor__min_samples_leaf": [1, 2, 4],
            },
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "regressor__n_estimators": [100, 200],
                "regressor__learning_rate": [0.05, 0.1],
                "regressor__max_depth": [2, 3, 4],
            },
        ),
    }


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)) * 100,
    }


def train_and_compare(csv_path: str = "data/house_prices.csv") -> dict:
    """
    Train all three candidate algorithms with cross-validated hyperparameter
    search, evaluate each on a held-out test set, and persist the best model.

    Returns a results dict (also written to docs/model_performance.json).
    """
    df = prepare_dataset(csv_path)
    numeric_features, categorical_features = get_feature_lists()

    X = df[numeric_features + categorical_features]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    logger.info("Train size: %s, Test size: %s", len(X_train), len(X_test))

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    fitted_pipelines = {}

    for name, (estimator, param_grid) in get_candidate_models().items():
        logger.info("Training %s ...", name)
        t0 = time.time()
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", estimator)])

        if param_grid:
            search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            best_params = search.best_params_
        else:
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline
            best_params = {}

        # Cross-validated R^2 on training data (for stability estimate)
        cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring="r2")

        y_pred = best_pipeline.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)
        metrics["cv_r2_mean"] = float(cv_scores.mean())
        metrics["cv_r2_std"] = float(cv_scores.std())
        metrics["best_params"] = best_params
        metrics["train_time_seconds"] = round(time.time() - t0, 2)

        results[name] = metrics
        fitted_pipelines[name] = best_pipeline
        logger.info(
            "%s -> MAE=%.0f  R2=%.3f  CV_R2=%.3f+-%.3f",
            name, metrics["mae"], metrics["r2"], metrics["cv_r2_mean"], metrics["cv_r2_std"]
        )

    # Pick best model by test R^2
    best_name = max(results, key=lambda n: results[n]["r2"])
    best_pipeline = fitted_pipelines[best_name]
    logger.info("Best model: %s (R2=%.3f)", best_name, results[best_name]["r2"])

    # Persist
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "house_price_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "pipeline": best_pipeline,
                "model_name": best_name,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "metrics": results[best_name],
                "version": "1.0.0",
                "trained_at": pd.Timestamp.now().isoformat(),
            },
            f,
        )
    logger.info("Saved best model to %s", model_path)

    # Feature importance (tree models only; linear uses coefficients)
    feature_importance = extract_feature_importance(best_pipeline, numeric_features, categorical_features, best_name)

    DOCS_DIR.mkdir(exist_ok=True)
    with open(DOCS_DIR / "model_performance.json", "w") as f:
        json.dump(
            {
                "results_by_model": results,
                "best_model": best_name,
                "feature_importance": feature_importance,
                "n_train": len(X_train),
                "n_test": len(X_test),
            },
            f,
            indent=2,
        )

    return {
        "results": results,
        "best_model": best_name,
        "feature_importance": feature_importance,
        "model_path": str(model_path),
    }


def extract_feature_importance(pipeline, numeric_features, categorical_features, model_name) -> list[dict]:
    """Extract and normalise feature importances (or |coefficients|) as percentages."""
    preprocessor = pipeline.named_steps["preprocessor"]
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_feature_names

    regressor = pipeline.named_steps["regressor"]
    if hasattr(regressor, "feature_importances_"):
        raw = regressor.feature_importances_
    elif hasattr(regressor, "coef_"):
        raw = np.abs(regressor.coef_)
    else:
        return []

    raw = np.array(raw, dtype=float)
    pct = 100 * raw / raw.sum() if raw.sum() > 0 else raw

    importance = sorted(
        [{"feature": f, "importance_pct": round(float(p), 2)} for f, p in zip(all_feature_names, pct)],
        key=lambda d: -d["importance_pct"],
    )
    return importance


if __name__ == "__main__":
    output = train_and_compare()
    print(json.dumps(output["results"], indent=2))
    print(f"\nBest model: {output['best_model']}")
    print("\nTop features:")
    for f in output["feature_importance"][:5]:
        print(f"  {f['feature']}: {f['importance_pct']}%")
