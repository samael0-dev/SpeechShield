#!/usr/bin/env bash
# Render Build Script for SpeechShield
set -o errexit

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing CPU-only PyTorch to conserve memory (<512MiB)..."
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "==> Pre-building starter datasets and baseline ML weights..."
mkdir -p data/raw data/processed data/weights
python scripts/scrape_data.py --source sample
python scripts/train_model.py --mode baseline --baseline-type logreg

echo "==> Build complete! SpeechShield is ready for deployment."
