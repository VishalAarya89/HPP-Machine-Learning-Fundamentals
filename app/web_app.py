"""
web_app.py
-----------
Flask web application + JSON API for the House Price Prediction model.

Routes:
  GET  /              -> HTML form for interactive predictions
  POST /predict        -> form submission handler (renders result in HTML)
  POST /api/predict     -> JSON API endpoint
  GET  /api/health       -> health check / model metadata
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.append(str(Path(__file__).parent.parent / "src"))
from model_inference import HousePriceModel, InputValidationError  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    model = HousePriceModel()
    logger.info("Model loaded successfully: %s", model.model_name)
except FileNotFoundError as e:
    logger.error(str(e))
    model = None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, error=None, form_data={})


@app.route("/predict", methods=["POST"])
def predict_form():
    form_data = request.form.to_dict()
    if model is None:
        return render_template("index.html", result=None, error="Model not loaded on server.", form_data=form_data)
    try:
        result = model.predict(form_data)
        return render_template("index.html", result=result, error=None, form_data=form_data)
    except InputValidationError as e:
        return render_template("index.html", result=None, error=str(e), form_data=form_data)
    except Exception:
        logger.exception("Unexpected error during prediction")
        return render_template(
            "index.html", result=None, error="An unexpected error occurred. Please check your inputs.", form_data=form_data
        )


@app.route("/api/predict", methods=["POST"])
def predict_api():
    if model is None:
        return jsonify({"error": "Model not loaded on server."}), 503
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400
    try:
        result = model.predict(payload)
        return jsonify({"success": True, "data": result}), 200
    except InputValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("Unexpected error during API prediction")
        return jsonify({"success": False, "error": "Internal server error."}), 500


@app.route("/api/health", methods=["GET"])
def health():
    if model is None:
        return jsonify({"status": "unhealthy", "reason": "model not loaded"}), 503
    return jsonify(
        {
            "status": "healthy",
            "model_name": model.model_name,
            "model_version": model.version,
            "metrics": {k: v for k, v in model.metrics.items() if k != "best_params"},
        }
    ), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
