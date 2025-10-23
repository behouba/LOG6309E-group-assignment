# Replication of RQ1

This directory contains our reproduction of Research Question 1 from Wu et&nbsp;al. (2023), *On the Effectiveness of Log Representation for Log-based Anomaly Detection*.

---

## Contents

- `data/`: raw, parsed, and processed datasets (HDFS and BGL)
- `scripts/`: pipeline steps (`01_parse_logs.py` … `08_verify_completeness.py`)
- `results/`: metrics, plots, and comparison tables
- `run_complete_pipeline.sh`: end-to-end execution helper
- `README.md`: current file

---

## How to Reproduce

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./run_complete_pipeline.sh
```

The script executes the following steps in order:

1. optional preprocessing following Qin et&nbsp;al.
2. log parsing with Drain (dataset specific regex rules in `scripts/01_parse_logs.py`)
3. train/test split (70/30 for HDFS, 80/20 for BGL)
4. feature generation
   - Message Count Vector (for classical model)
   - Word2Vec embeddings (for LSTM; falls back to in-domain training if no pretrained model is available)
5. feature filtering with correlation clustering and variance inflation factor (VIF)
6. model training and evaluation
   - Random Forest, Logistic Regression, Decision Tree (classical models)
   - LSTM sequence model
7. performance metrics summary table generation
8. comparison against the reference results

Each script writes logs to `logs/` and intermediate artefacts to `data/processed/`.

---

## Datasets

- **HDFS**: parsed with block identifiers; produces one session per block (575,061 sessions).
- **BGL**: parsed with 6-hour windows as in the paper; yields 824 sessions.

Both datasets come from LogHub. Download helpers are included in `scripts/00_download_data.py`.

---

## Results

### Performance Summary

A comprehensive performance metrics table is automatically generated in `results/performance_metrics_table.png` showing Precision, Recall, F1-Score, and AUC for all models across both datasets.

**Key findings:**

| Dataset | Model | Precision | Recall | F1 | AUC | Notes |
|---------|-------|----------:|-------:|---:|----:|-------|
| HDFS | Random Forest | 99.9% | 99.9% | 99.91% | 1.000 | Feature selection has no effect (48 → 16 features) |
| HDFS | LSTM | 99.6% | 99.4% | 99.49% | 0.998 | Word2Vec embeddings on event templates |
| BGL | Random Forest | 94.9% | 90.4% | 92.59% | 0.976 | Feature selection improves precision by ~1pp (371 → 165 features) |
| BGL | LSTM | 61.3% | 88.0% | 72.28% | 0.798 | Fewer training sessions (824) lead to overfitting |

Comparison with paper (difference reported as ours − paper):

- HDFS Random Forest: +0.01% F1
- HDFS LSTM: +3.69% F1
- BGL Random Forest: −1.31% F1
- BGL LSTM: −14.22% F1

The gap on BGL LSTM matches the discussion in Section&nbsp;6 of Wu et&nbsp;al.; with only 659 training sessions after the split, the sequence model overfits quickly, whereas classical models remain stable.

---

## Feature Reduction

Correlation analysis clusters features with Spearman |ρ| ≥ 0.95 and keeps one representative per cluster. VIF (threshold 5.0) removes remaining redundancy. Resulting feature sets and before/after metrics are saved under `results/feature_selection/`.

### Visualizations

The pipeline generates the following key visualizations in `results/`:

- **`performance_metrics_table.png`**: Comprehensive summary of all models' performance metrics
- **`comparison_HDFS.png`**: HDFS model comparison with/without feature selection
- **`comparison_BGL.png`**: BGL model comparison with/without feature selection
- **`roc_HDFS_full_rf_with_fs.png`**: ROC curve for HDFS Random Forest
- **`roc_BGL_full_rf_with_fs.png`**: ROC curve for BGL Random Forest
- **`correlation_matrix_HDFS.png`**: Correlation analysis visualization

---

## Verification

Use `python scripts/08_verify_completeness.py` to confirm that every intermediate file and metric has been produced. The script checks both datasets and flags missing artefacts.

---

## References

- Wu, X., Li, H., & Khomh, F. (2023). On the effectiveness of log representation for log-based anomaly detection. *Empirical Software Engineering, 28*, 137. https://doi.org/10.1007/s10664-023-10364-1
- He, P., Zhu, J., Zheng, Z., & Lyu, M. R. (2017). Drain: An online log parsing approach with fixed depth tree.
- Qin, X. et al. Log preprocessing techniques referenced by Wu et&nbsp;al.
