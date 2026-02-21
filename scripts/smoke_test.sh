#!/bin/bash

set -e   # fail immediately on error

echo "Running smoke tests..."

# 1. Health check
echo "Checking /health endpoint..."
curl -f http://localhost:8000/health

echo "Health check passed"

# 2. Prediction check
echo "Checking /predict endpoint..."
curl -f -X POST http://localhost:8000/predict \
  -F "file=@scripts/50.jpg"

echo "Prediction test passed"

echo "All smoke tests passed!"