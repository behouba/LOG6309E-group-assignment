# Extension - Unsupervised Models and Analysis

This folder extends the replication by evaluating two unsupervised approaches on HDFS and by ranking all four models (two supervised from Part 1, two unsupervised added here).

---

## Directory Layout

- `scripts/01_train_unsupervised.py`: trains the Isolation Forest and Autoencoder using the Message Count Vector features.
- `scripts/02_resample_evaluate.py`: repeated train/test splits (10×) for all four models.
- `scripts/03_scott_knott_ranking.py`: Scott-Knott ESD on the resampled scores.
- `scripts/create_ranking_visualization.py`: generates ranking visualizations (bar charts and summary table).
- `scripts/04_model_explanation.py`: SHAP analysis for the top-ranked model.
- `config.py`: configuration values used across scripts.
- `run_extension.sh`: runs the whole sequence.
- `results/`: JSON metrics, ranking CSV, SHAP plots, and ranking visualizations.
- `logs/`: execution traces.

---

## Running the Pipeline

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./run_extension.sh
```

Prerequisite: Part 1 must already have produced the processed HDFS data (`../replication/data/processed/`).

The script executes:
1. Train Isolation Forest and Autoencoder.
2. Resample train/test splits ten times and score Random Forest, LSTM, Isolation Forest, Autoencoder.
3. Apply Scott-Knott ESD to the F1 and AUC distributions.
4. Generate ranking visualizations (bar charts and summary table).
5. Run SHAP on the highest ranked model (Random Forest).

Intermediate artefacts are written to `results/` and `logs/`.

---

## Key Outcomes

| Model | Type | Mean F1 | Mean AUC | Notes |
|-------|------|--------:|---------:|-------|
| Random Forest | Supervised | 0.998 | 1.000 | Remains the strongest model across resamples. |
| LSTM | Supervised | 0.993 | 0.999 | Slightly below Random Forest; sensitive to session count. |
| Isolation Forest | Unsupervised | 0.661 | 0.962 | Better F1 than the Autoencoder with modest tuning effort. |
| Autoencoder | Unsupervised | 0.619 | 0.986 | High AUC but lower F1 due to false positives. |

Scott-Knott groups the supervised models in the top tier, followed by Isolation Forest and then the Autoencoder.

SHAP analysis of the Random Forest shows that a small number of log templates (e.g., Template_10, Template_6) dominate the predictions, which supports using template frequency monitoring for this dataset.

### Visualizations

The pipeline generates the following visualizations in `results/`:

- **`scott_knott_ranking_table.png`**: Summary table showing statistical ranking for each model by F1-Score and AUC
- **`scott_knott_ranking_visualization.png`**: Bar charts visualizing model rankings for both metrics
- **`f1_score_distribution.png`**: Distribution of F1-scores across 10 resampling iterations
- **`auc_distribution.png`**: Distribution of AUC values across 10 resampling iterations
- **`shap_summary.png`**: SHAP feature importance summary for Random Forest
- **`shap_waterfall_single.png`**: Detailed SHAP explanation for a single prediction

---

## Next Steps

- Explore different contamination settings for Isolation Forest when applying the model to other datasets.
- If more labelled data becomes available, revisit sequence models with larger architectures or alternative embeddings.
