#!/bin/bash

# ==============================================================================
# Part 2 Extension - Complete Pipeline Runner
#
# This script executes all steps for the Part 2 extension:
# 1. Train unsupervised models (Isolation Forest & Autoencoder).
# 2. Resample the dataset and evaluate all four models (RF, LSTM, IForest, AE).
# 3. Perform Scott-Knott statistical ranking.
# 4. Generate SHAP explanations for the best-performing classical model.
#
# Usage:
#   cd extension/
#   ./run_extension.sh
#
# Prerequisites:
#   - Python environment with all dependencies from requirements.txt installed.
#   - Part 1 pipeline must have been run successfully, as this script
#     depends on the data generated in ../replication/data/
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PYTHON_CMD="./venv/bin/python"
if [ ! -x "$PYTHON_CMD" ]; then
  echo "Embedded virtual environment not found. Falling back to system python."
  PYTHON_CMD="python3"
fi
SCRIPT_DIR="scripts"
LOG_DIR="logs"
RESULTS_DIR="results"
MODELS_DIR="models"

# --- Setup ---
echo "============================================================"
echo "          PART 2: EXTENSION PIPELINE - STARTING"
echo "============================================================"
echo "Timestamp: $(date)"
echo

# Create directories for logs and results
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$MODELS_DIR"

echo "Cleaning up previous run logs..."
rm -f "$LOG_DIR"/*.log
echo

# --- Step 1: Train Unsupervised Models ---
echo "------------------------------------------------------------"
echo "STEP 1: Training Unsupervised Models"
echo "------------------------------------------------------------"
echo "Running: 01_train_unsupervised.py"
echo "Description: Trains Isolation Forest and Autoencoder."
echo "Output Log: $LOG_DIR/01_unsupervised_training.log"
echo

$PYTHON_CMD "$SCRIPT_DIR/01_train_unsupervised.py" | tee "$LOG_DIR/01_unsupervised_training.log"

echo
echo "✅ STEP 1 COMPLETE"
echo

# --- Step 2: Resample and Evaluate All Models ---
echo "------------------------------------------------------------"
echo "STEP 2: Resampling and Evaluating All Models"
echo "------------------------------------------------------------"
echo "Running: 02_resample_evaluate.py"
echo "Description: Generates performance distributions over 10 runs."
echo "Output Log: $LOG_DIR/02_resampling.log"
echo

$PYTHON_CMD "$SCRIPT_DIR/02_resample_evaluate.py" | tee "$LOG_DIR/02_resampling.log"

echo
echo "✅ STEP 2 COMPLETE"
echo

# --- Step 3: Scott-Knott Statistical Ranking ---
echo "------------------------------------------------------------"
echo "STEP 3: Performing Scott-Knott Statistical Ranking"
echo "------------------------------------------------------------"
echo "Running: 03_scott_knott_ranking.py"
echo "Description: Ranks models based on statistical significance."
echo "Output Log: $LOG_DIR/03_scott_knott.log"
echo

$PYTHON_CMD "$SCRIPT_DIR/03_scott_knott_ranking.py" | tee "$LOG_DIR/03_scott_knott.log"

echo
echo "✅ STEP 3 COMPLETE"
echo

# --- Step 4: Model Explanation with SHAP ---
echo "------------------------------------------------------------"
echo "STEP 4: Generating Model Explanations with SHAP"
echo "------------------------------------------------------------"
echo "Running: 04_model_explanation.py"
echo "Description: Creates SHAP plots for the best model."
echo "Output Log: $LOG_DIR/04_explanation.log"
echo

$PYTHON_CMD "$SCRIPT_DIR/04_model_explanation.py" | tee "$LOG_DIR/04_explanation.log"

echo
echo "✅ STEP 4 COMPLETE"
echo

# --- Final Summary ---
echo "============================================================"
echo "          PART 2: EXTENSION PIPELINE - COMPLETE"
echo "============================================================"
echo
echo "All steps executed successfully."
echo "Check the '$RESULTS_DIR' directory for all generated reports and plots:"
echo "  - unsupervised_summary_HDFS_full.json (Unsupervised model performance)"
echo "  - resampling_results.json             (Raw data for statistical test)"
echo "  - scott_knott_ranking.csv             (Final model ranks)"
echo "  - *.png                               (Distribution and SHAP plots)"
echo "  - PART2_EXTENSION_REPORT.md           (Final summary report)"
echo
echo "Timestamp: $(date)"
echo "============================================================"
