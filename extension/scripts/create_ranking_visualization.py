#!/usr/bin/env python3
"""
Create Scott-Knott ranking visualization for extension results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

results_dir = Path(__file__).parent.parent / "results"

# Load Scott-Knott ranking
ranking_df = pd.read_csv(results_dir / "scott_knott_ranking.csv")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Colors for ranks
colors = ['#2E7D32', '#558B2F', '#9E9D24', '#F57C00']

# Plot F1-Score ranking
models = ranking_df['Model'].values
f1_ranks = ranking_df['F1_Rank'].values

bars1 = ax1.barh(models, f1_ranks, color=[colors[int(r)-1] for r in f1_ranks])
ax1.set_xlabel('Rank (Lower is Better)', fontsize=11, fontweight='bold')
ax1.set_title('Scott-Knott Ranking by F1-Score', fontsize=12, fontweight='bold')
ax1.invert_xaxis()
ax1.set_xlim(5, 0)
ax1.set_xticks([1, 2, 3, 4])
ax1.grid(axis='x', alpha=0.3)

# Add rank labels
for i, (model, rank) in enumerate(zip(models, f1_ranks)):
    ax1.text(rank + 0.15, i, f'Rank {int(rank)}',
            va='center', fontsize=10, fontweight='bold')

# Plot AUC ranking
auc_ranks = ranking_df['AUC_Rank'].values

bars2 = ax2.barh(models, auc_ranks, color=[colors[int(r)-1] for r in auc_ranks])
ax2.set_xlabel('Rank (Lower is Better)', fontsize=11, fontweight='bold')
ax2.set_title('Scott-Knott Ranking by AUC', fontsize=12, fontweight='bold')
ax2.invert_xaxis()
ax2.set_xlim(5, 0)
ax2.set_xticks([1, 2, 3, 4])
ax2.grid(axis='x', alpha=0.3)

# Add rank labels
for i, (model, rank) in enumerate(zip(models, auc_ranks)):
    ax2.text(rank + 0.15, i, f'Rank {int(rank)}',
            va='center', fontsize=10, fontweight='bold')

plt.suptitle('Statistical Ranking of Models (Extension - RQ2)',
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

output_file = results_dir / "scott_knott_ranking_visualization.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Created: {output_file}")

# Create a summary table visualization
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')

# Prepare data
table_data = []
for _, row in ranking_df.iterrows():
    f1_rank = int(row['F1_Rank'])
    auc_rank = int(row['AUC_Rank'])
    table_data.append([
        row['Model'],
        f"Rank {f1_rank}",
        f"Rank {auc_rank}",
        f"{(f1_rank + auc_rank) / 2:.1f}"
    ])

table = ax.table(cellText=table_data,
                colLabels=['Model', 'F1-Score Rank', 'AUC Rank', 'Average Rank'],
                cellLoc='center', loc='center',
                colWidths=[0.3, 0.2, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(table_data) + 1):
    # Highlight best model (Random Forest)
    if i == 1:
        for j in range(4):
            table[(i, j)].set_facecolor('#C6E0B4')
            table[(i, j)].set_text_props(weight='bold')
    elif i % 2 == 0:
        for j in range(4):
            table[(i, j)].set_facecolor('#E7E6E6')

plt.title('Scott-Knott Test Results Summary',
          fontsize=13, fontweight='bold', pad=15)

output_file2 = results_dir / "scott_knott_ranking_table.png"
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"✓ Created: {output_file2}")

print("\nRanking Summary:")
print(ranking_df.to_string(index=False))
