#!/usr/bin/env python3
"""
Create comprehensive performance metrics table for replication results
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10

results_dir = Path(__file__).parent.parent / "results"

EXPECTED_FILES = {
    "HDFS": {
        "classical": "HDFS_full_classical_results.json",
        "lstm": "lstm_HDFS_full_results.json",
    },
    "BGL": {
        "classical": "BGL_full_classical_results.json",
        "lstm": "lstm_BGL_full_results.json",
    },
}


def load_json(path: Path):
    try:
        with path.open() as handle:
            return json.load(handle)
    except FileNotFoundError:
        print(f"[WARN] Missing results file: {path.name}; skipping dataset.")
        return None


def collect_dataset_rows(dataset: str):
    file_map = EXPECTED_FILES[dataset]
    classical_path = results_dir / file_map["classical"]
    lstm_path = results_dir / file_map["lstm"]

    classical_results = load_json(classical_path)
    lstm_results = load_json(lstm_path)

    if classical_results is None or lstm_results is None:
        return []

    rows = []
    for model_name, metrics in classical_results.items():
        rows.append(
            {
                "Dataset": dataset,
                "Model": model_name,
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1-Score": f"{metrics['f1_score']:.4f}",
                "AUC": f"{metrics['auc']:.4f}",
            }
        )

    rows.append(
        {
            "Dataset": dataset,
            "Model": "LSTM",
            "Precision": f"{lstm_results['precision']:.4f}",
            "Recall": f"{lstm_results['recall']:.4f}",
            "F1-Score": f"{lstm_results['f1_score']:.4f}",
            "AUC": f"{lstm_results['auc']:.4f}",
        }
    )
    return rows


all_rows = []
for dataset in EXPECTED_FILES:
    dataset_rows = collect_dataset_rows(dataset)
    if dataset_rows:
        all_rows.extend(dataset_rows)

if not all_rows:
    print("[WARN] No completed results found. Skipping performance summary generation.")
    sys.exit(0)

df = pd.DataFrame(all_rows)

results_dir.mkdir(parents=True, exist_ok=True)

# Save as CSV
csv_file = results_dir / "performance_metrics_summary.csv"
df.to_csv(csv_file, index=False)
print(f"✓ Created: {csv_file}")

# Create visual table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("tight")
ax.axis("off")

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc="center",
    loc="center",
    colWidths=[0.12, 0.18, 0.15, 0.15, 0.15, 0.15],
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for col_idx in range(len(df.columns)):
    table[(0, col_idx)].set_facecolor("#4472C4")
    table[(0, col_idx)].set_text_props(weight="bold", color="white")

for row_idx in range(1, len(df) + 1):
    if row_idx % 2 == 0:
        for col_idx in range(len(df.columns)):
            table[(row_idx, col_idx)].set_facecolor("#E7E6E6")

    dataset_color = "#C6E0B4" if df.iloc[row_idx - 1]["Dataset"] == "HDFS" else "#FFE699"
    table[(row_idx, 0)].set_facecolor(dataset_color)

plt.title("Performance Metrics Summary - Replication (RQ1)", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()

output_file = results_dir / "performance_metrics_table.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"✓ Created: {output_file}")

print("\nPerformance Summary:")
print(df.to_string(index=False))
