#!/bin/bash
# RQ1 replication pipeline.
# This script runs the replication for both datasets (HDFS and BGL) end-to-end:
#   1. Downloads datasets (if needed)
#   2. Parses logs with Drain
#   3. Splits data (70/30 train/test)
#   4. Generates representations (MCV + Word2Vec)
#   5. Performs feature selection (correlation + VIF)
#   6. Trains classical models (RF with/without FS, LR, DT)
#   7. Trains LSTM models
#   8. Compares with/without feature selection
#   9. Generates comparison with original paper
#
# Usage:
#   ./run_complete_pipeline.sh              # Run everything for both datasets
#   ./run_complete_pipeline.sh --hdfs-only  # Run only HDFS
#   ./run_complete_pipeline.sh --bgl-only   # Run only BGL
#
# ============================================================================

set -euo pipefail  # Fail fast, even in pipelines
trap 'echo "Error on line $LINENO. Exiting."; exit 1' ERR

ARG="${1:-}"

# Default: run both datasets
RUN_HDFS=true
RUN_BGL=true

# Parse arguments
if [ "$ARG" = "--hdfs-only" ]; then
    RUN_BGL=false
elif [ "$ARG" = "--bgl-only" ]; then
    RUN_HDFS=false
fi

echo "Running RQ1 replication pipeline"
if [ "$RUN_HDFS" = true ] && [ "$RUN_BGL" = true ]; then
    echo "Datasets: HDFS and BGL"
elif [ "$RUN_HDFS" = true ]; then
    echo "Dataset: HDFS"
else
    echo "Dataset: BGL"
fi

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "Error: virtual environment not found."
    echo "Please create it first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

source venv/bin/activate
echo "Virtual environment activated (Python $(python --version))"

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p results

# Track start time
START_TIME=$(date +%s)

# Function to run pipeline for one dataset
run_dataset_pipeline() {
    local DATASET=$1
    local START=$(date +%s)

    echo ""
    echo "######################################################################"
    echo ""
    echo "Processing dataset: $DATASET"

    # Update config
    echo "Updating config for $DATASET"
    sed -i "s/^DATASET = .*/DATASET = \"$DATASET\"/" config.py
    sed -i 's/^DATA_MODE = .*/DATA_MODE = "full"/' config.py
    echo "Configuration updated"

    # Step 0: Download (only once, shared across datasets)
    if [ ! -f "data/raw/$DATASET/${DATASET}.log" ]; then
        echo "Step 0: download raw data"
        python scripts/00_download_datasets.py 2>&1 | tee logs/00_download_${DATASET}.log
        echo "Download complete"
    else
        echo "$DATASET raw data already present; skipping download"
    fi

    # Step 1: Parse logs
    echo "Step 1: parse logs"
    if [ "$DATASET" = "HDFS" ]; then
        python scripts/01_parse_logs.py HDFS data/raw/HDFS/HDFS.log 2>&1 | tee logs/01_parse_${DATASET}.log
    else
        python scripts/01_parse_logs.py BGL data/raw/BGL/BGL.log 2>&1 | tee logs/01_parse_${DATASET}.log
    fi
    echo "Parsing complete"

    # Step 2: Split data
    echo "Step 2: split data"
    python scripts/02_split_data.py 2>&1 | tee logs/02_split_${DATASET}.log
    echo "Split complete"

    # Step 3: Generate representations
    echo "Step 3: generate representations"
    python scripts/03_generate_representations.py 2>&1 | tee logs/03_representations_${DATASET}.log
    echo "Representations ready"

    # Step 4: Feature selection
    echo "Step 4: run feature selection"
    python scripts/04_feature_selection.py 2>&1 | tee logs/04_feature_selection_${DATASET}.log
    echo "Feature selection complete"

    # Step 5: Train classical models
    echo "Step 5: train classical models"
    python scripts/05_train_evaluate.py 2>&1 | tee logs/05_classical_${DATASET}.log
    echo "Classical models trained"

    # Step 6: Train LSTM
    echo "Step 6: train LSTM"
    python scripts/06_train_lstm.py 2>&1 | tee logs/06_lstm_${DATASET}.log
    echo "LSTM trained"

    # Step 7: Generate performance summary table
    echo "Step 7: generate performance metrics summary table"
    python scripts/create_summary_table.py 2>&1 | tee logs/07_summary_table_${DATASET}.log
    echo "Performance summary table generated"

    local END=$(date +%s)
    local DURATION=$((END - START))
    local MINUTES=$((DURATION / 60))
    local SECONDS=$((DURATION % 60))

    echo "$DATASET pipeline complete in ${MINUTES}m ${SECONDS}s"
    echo ""
}

# Run pipeline for selected datasets
if [ "$RUN_HDFS" = true ]; then
    run_dataset_pipeline "HDFS"
fi

if [ "$RUN_BGL" = true ]; then
    run_dataset_pipeline "BGL"
fi

# Step 8: Compare results
if [ "$RUN_HDFS" = true ] && [ "$RUN_BGL" = true ]; then
    echo "Step 8: compare results with paper"
    if [ -f "scripts/07_compare_with_paper.py" ]; then
        python scripts/07_compare_with_paper.py 2>&1 | tee logs/08_compare_results.log
        echo "Comparison complete"
    else
        echo "Comparison script not found; skipping."
    fi
fi

# Calculate total time
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

# Final summary
echo ""
echo "Pipeline complete in ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "Key artefacts stored in results/; logs stored in logs/."
