#!/bin/bash
# ============================================================================
# COMPLETE RQ1 REPLICATION PIPELINE - ONE COMMAND TO RULE THEM ALL
# ============================================================================
# This script runs the COMPLETE replication for BOTH datasets (HDFS and BGL)
# from start to finish.
#
# What it does:
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

set -e  # Exit on error
trap 'echo "❌ Error on line $LINENO. Exiting."; exit 1' ERR

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default: run both datasets
RUN_HDFS=true
RUN_BGL=true

# Parse arguments
if [ "$1" = "--hdfs-only" ]; then
    RUN_BGL=false
elif [ "$1" = "--bgl-only" ]; then
    RUN_HDFS=false
fi

echo ""
echo "========================================================================"
echo "  RQ1 REPLICATION - COMPLETE PIPELINE"
echo "========================================================================"
echo ""
echo "This will run the COMPLETE replication pipeline including:"
echo "  ✓ Dataset download"
echo "  ✓ Log parsing (Drain)"
echo "  ✓ Data splitting"
echo "  ✓ Representation generation (MCV + Word2Vec)"
echo "  ✓ Feature selection (Correlation + VIF)"
echo "  ✓ Model training (RF, LR, DT, LSTM)"
echo "  ✓ Comparison with/without feature selection"
echo ""

if [ "$RUN_HDFS" = true ] && [ "$RUN_BGL" = true ]; then
    echo -e "${BLUE}Datasets: HDFS + BGL${NC}"
    echo -e "${YELLOW}Estimated time: 40-50 minutes${NC}"
elif [ "$RUN_HDFS" = true ]; then
    echo -e "${BLUE}Dataset: HDFS only${NC}"
    echo -e "${YELLOW}Estimated time: 15-20 minutes${NC}"
else
    echo -e "${BLUE}Dataset: BGL only${NC}"
    echo -e "${YELLOW}Estimated time: 20-30 minutes${NC}"
fi

echo ""
echo "========================================================================"
echo ""

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Error: Virtual environment not found!${NC}"
    echo "Please create it first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo "  Python: $(python --version)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Track start time
START_TIME=$(date +%s)

# Function to run pipeline for one dataset
run_dataset_pipeline() {
    local DATASET=$1
    local START=$(date +%s)

    echo ""
    echo "######################################################################"
    echo "#"
    echo "#  PROCESSING: $DATASET"
    echo "#"
    echo "######################################################################"
    echo ""

    # Update config
    echo "Configuring for $DATASET..."
    sed -i "s/^DATASET = .*/DATASET = \"$DATASET\"/" config.py
    sed -i 's/^DATA_MODE = .*/DATA_MODE = "full"/' config.py
    echo -e "${GREEN}✓ Configuration updated${NC}"
    echo ""

    # Step 0: Download (only once, shared across datasets)
    if [ ! -f "data/raw/$DATASET/${DATASET}.log" ]; then
        echo "=========================================="
        echo "STEP 0: Download $DATASET Dataset"
        echo "=========================================="
        python scripts/00_download_datasets.py 2>&1 | tee logs/00_download_${DATASET}.log
        echo -e "${GREEN}✓ Download complete${NC}"
        echo ""
    else
        echo "✓ $DATASET dataset already downloaded, skipping..."
        echo ""
    fi

    # Step 1: Parse logs
    echo "=========================================="
    echo "STEP 1: Parse Logs - $DATASET"
    echo "=========================================="
    if [ "$DATASET" = "HDFS" ]; then
        python scripts/01_parse_logs.py HDFS data/raw/HDFS/HDFS.log 2>&1 | tee logs/01_parse_${DATASET}.log
    else
        python scripts/01_parse_logs.py BGL data/raw/BGL/BGL.log 2>&1 | tee logs/01_parse_${DATASET}.log
    fi
    echo -e "${GREEN}✓ Parsing complete${NC}"
    echo ""

    # Step 2: Split data
    echo "=========================================="
    echo "STEP 2: Split Data - $DATASET"
    echo "=========================================="
    python scripts/02_split_data.py 2>&1 | tee logs/02_split_${DATASET}.log
    echo -e "${GREEN}✓ Splitting complete${NC}"
    echo ""

    # Step 3: Generate representations
    echo "=========================================="
    echo "STEP 3: Generate Representations - $DATASET"
    echo "=========================================="
    python scripts/03_generate_representations.py 2>&1 | tee logs/03_representations_${DATASET}.log
    echo -e "${GREEN}✓ Representations complete${NC}"
    echo ""

    # Step 4: Feature selection
    echo "=========================================="
    echo "STEP 4: Feature Selection - $DATASET"
    echo "=========================================="
    python scripts/04_feature_selection.py 2>&1 | tee logs/04_feature_selection_${DATASET}.log
    echo -e "${GREEN}✓ Feature selection complete${NC}"
    echo ""

    # Step 5: Train classical models
    echo "=========================================="
    echo "STEP 5: Train Classical Models - $DATASET"
    echo "=========================================="
    python scripts/05_train_evaluate.py 2>&1 | tee logs/05_classical_${DATASET}.log
    echo -e "${GREEN}✓ Classical models complete${NC}"
    echo ""

    # Step 6: Train LSTM
    echo "=========================================="
    echo "STEP 6: Train LSTM Model - $DATASET"
    echo "=========================================="
    python scripts/06_train_lstm.py 2>&1 | tee logs/06_lstm_${DATASET}.log
    echo -e "${GREEN}✓ LSTM complete${NC}"
    echo ""

    local END=$(date +%s)
    local DURATION=$((END - START))
    local MINUTES=$((DURATION / 60))
    local SECONDS=$((DURATION % 60))

    echo ""
    echo -e "${GREEN}✓✓✓ $DATASET PIPELINE COMPLETE ✓✓✓${NC}"
    echo "Time taken: ${MINUTES}m ${SECONDS}s"
    echo ""
}

# Run pipeline for selected datasets
if [ "$RUN_HDFS" = true ]; then
    run_dataset_pipeline "HDFS"
fi

if [ "$RUN_BGL" = true ]; then
    run_dataset_pipeline "BGL"
fi

# Step 7: Compare results
if [ "$RUN_HDFS" = true ] && [ "$RUN_BGL" = true ]; then
    echo "=========================================="
    echo "STEP 7: Compare Results with Paper"
    echo "=========================================="
    if [ -f "scripts/07_compare_with_paper.py" ]; then
        python scripts/07_compare_with_paper.py 2>&1 | tee logs/07_compare_results.log
        echo -e "${GREEN}✓ Comparison complete${NC}"
    else
        echo "Note: Comparison script not found, skipping..."
    fi
    echo ""
fi

# Calculate total time
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

# Final summary
echo ""
echo "######################################################################"
echo "#"
echo "#  🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY! 🎉"
echo "#"
echo "######################################################################"
echo ""
echo -e "${GREEN}Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s${NC}"
echo ""
echo "Results are available in:"
echo "  📊 results/*_classical_results.json"
echo "  📊 results/*_comparison_table.csv"
echo "  📊 results/lstm_*_results.json"
echo "  📈 results/correlation_matrix_*.png"
echo "  📈 results/roc_*_*.png"
echo ""
echo "Logs are available in:"
echo "  📝 logs/*.log"
echo ""
echo "========================================================================"
echo "What was accomplished:"
echo ""
if [ "$RUN_HDFS" = true ]; then
    echo "  ✅ HDFS dataset: parsed, split, represented, feature-selected, trained"
fi
if [ "$RUN_BGL" = true ]; then
    echo "  ✅ BGL dataset: parsed, split, represented, feature-selected, trained"
fi
echo "  ✅ Classical models: RF (with/without FS), LR, DT"
echo "  ✅ Deep learning: LSTM"
echo "  ✅ Feature selection: Correlation + VIF analysis"
echo "  ✅ Comparison: With vs. without feature selection"
echo ""
echo "========================================================================"
echo "Next steps:"
echo "  1. Review results in results/ directory"
echo "  2. Compare metrics with original paper (Wu et al. 2023)"
echo "  3. Document any differences and analyze why"
echo "  4. Proceed to extension tasks (unsupervised models)"
echo "========================================================================"
echo ""
