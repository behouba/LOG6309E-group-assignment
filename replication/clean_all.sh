#!/bin/bash
# Clean up all generated data and results for fresh pipeline run
# This script removes all intermediate files but keeps raw data

echo "=========================================="
echo "Cleaning Replication Pipeline"
echo "=========================================="

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "Removing generated data..."
rm -rf data/parsed/*
rm -rf data/split/*
rm -rf data/representations/*
echo "  Cleaned parsed, split, and representations"

echo ""
echo "Removing results..."
rm -rf results/*
echo "  Cleaned results directory"

echo ""
echo "Removing logs (keeping directory)..."
rm -f logs/*.log
echo "  Cleaned log files"

echo ""
echo "Removing model checkpoints..."
rm -rf results/models/*
echo "  Cleaned model checkpoints"

echo ""
echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo ""
echo "What was kept:"
echo "  Raw data in data/raw/"
echo "  Scripts in scripts/"
echo "  Virtual environment (venv/)"
echo ""
echo "You can now run the full pipeline from scratch:"
echo "  ./run_complete_pipeline.sh"
echo ""
