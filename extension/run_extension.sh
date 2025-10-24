#!/bin/bash

# Part 2 extension pipeline runner.

# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

# Configuration
PYTHON_CMD="./venv/bin/python"
if [ ! -x "$PYTHON_CMD" ]; then
  echo "Embedded virtual environment not found. Falling back to system python."
  PYTHON_CMD="python3"
fi
SCRIPT_DIR="scripts"
LOG_DIR="logs"
RESULTS_DIR="results"
MODELS_DIR="models"

# Setup
echo "Running Part 2 extension pipeline ($(date))"

# Create directories for logs and results
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$MODELS_DIR"

echo "Cleaning previous logs"
rm -f "$LOG_DIR"/*.log
echo "Logs cleared"

# Step 1: Train Unsupervised Models
echo "Step 1: train Isolation Forest and Autoencoder"

$PYTHON_CMD "$SCRIPT_DIR/01_train_unsupervised.py" | tee "$LOG_DIR/01_unsupervised_training.log"

echo "Step 1 complete"

# Step 2: Resample and Evaluate All Models
echo "Step 2: resample and score models"

$PYTHON_CMD "$SCRIPT_DIR/02_resample_evaluate.py" | tee "$LOG_DIR/02_resampling.log"

echo "Step 2 complete"

# Step 3: Scott-Knott Statistical Ranking
echo "Step 3: run Scott-Knott ranking"

$PYTHON_CMD "$SCRIPT_DIR/03_scott_knott_ranking.py" | tee "$LOG_DIR/03_scott_knott.log"

echo "Step 3 complete"

# Step 3b: Generate ranking visualizations
echo "Step 3b: generate ranking visualizations"

$PYTHON_CMD "$SCRIPT_DIR/create_ranking_visualization.py" | tee "$LOG_DIR/03b_ranking_viz.log"

echo "Step 3b complete"

# Step 4: Model Explanation with SHAP
echo "Step 4: generate SHAP explanation"

$PYTHON_CMD "$SCRIPT_DIR/04_model_explanation.py" | tee "$LOG_DIR/04_explanation.log"

echo "Step 4 complete"

# Final Summary
echo "Extension pipeline finished ($(date)). Results stored in $RESULTS_DIR."
