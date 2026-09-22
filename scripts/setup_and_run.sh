#!/usr/bin/env bash
# Setup and run script for the House Price Prediction project.
#
# Usage:
#   ./scripts/setup_and_run.sh          # install deps, train model, run app
#   ./scripts/setup_and_run.sh --skip-train   # skip retraining if a model already exists

set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Installing dependencies"
pip install -r requirements.txt

SKIP_TRAIN=false
if [[ "${1:-}" == "--skip-train" ]]; then
  SKIP_TRAIN=true
fi

if [[ "$SKIP_TRAIN" == false || ! -f "models/house_price_model.pkl" ]]; then
  echo "==> Training model"
  python3 src/model_training.py
else
  echo "==> Skipping training, using existing model"
fi

echo "==> Running test suite"
python3 -m pytest tests/ -v

echo "==> Starting web application on http://localhost:5000"
python3 app/web_app.py
