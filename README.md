# Log-Based Anomaly Detection Project

This repository contains two deliverables for replicating and extending the experiments from Wu et al. (2023) on log-based anomaly detection.

## Structure

- `replication/` – Part 1. Implements the full pipeline for Research Question 1 (log parsing, feature engineering, supervised models, feature selection, comparison with the paper).
- `extension/` – Part 2. Adds unsupervised models, resampling-based evaluation, Scott-Knott ranking, and SHAP explanations on top of the replication outputs.

## Setup

1. Python 3.10+ with venv support.
2. From the repository root:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r replication/requirements.txt
   ```

The requirements include PyTorch, SHAP, and the Drain parser (via git). CUDA-enabled PyTorch wheels are selected automatically by pip if a compatible GPU is present.

## Running the Pipelines

### Part 1 – Replication

```bash
cd replication
./run_complete_pipeline.sh            # HDFS + BGL
./run_complete_pipeline.sh --hdfs-only
./run_complete_pipeline.sh --bgl-only
```

Results (metrics, plots, comparisons) are stored under `replication/results/`, with logs in `replication/logs/`.

### Part 2 – Extension

Requires the processed HDFS data produced by Part 1.

```bash
cd extension
./run_extension.sh
```

Outputs (model metrics, resampling summaries, ranking CSV, SHAP plots) are saved in `extension/results/`.

