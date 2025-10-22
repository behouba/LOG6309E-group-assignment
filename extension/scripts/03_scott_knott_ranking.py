"""
Statistical Ranking using Scott-Knott Test

This script takes the resampling results and applies the Scott-Knott Effect
Size Difference (SK-ESD) test to group the models into statistically
distinct ranks based on their F1-score and AUC.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR

ALPHA = 0.05
CLIFFS_DELTA_THRESHOLD = 0.147


def cliffs_delta(a, b):
    """Compute Cliff's delta effect size."""
    a = np.asarray(a)
    b = np.asarray(b)
    m = len(a)
    n = len(b)
    if m == 0 or n == 0:
        return 0.0
    greater = 0
    lesser = 0
    for value in a:
        greater += np.sum(value > b)
        lesser += np.sum(value < b)
    return (greater - lesser) / (m * n)


def scott_knott(groups):
    """Custom Scott-Knott Effect Size Difference implementation."""

    def split(model_list):
        if len(model_list) <= 1:
            return [model_list]

        samples = [groups[model] for model in model_list]
        if any(len(sample) == 0 for sample in samples):
            return [model_list]

        f_stat, p_value = stats.f_oneway(*samples)
        if np.isnan(f_stat) or p_value >= ALPHA:
            return [model_list]

        overall = np.concatenate(samples)
        overall_mean = np.mean(overall)

        best_score = -np.inf
        best_split = None

        for idx in range(1, len(model_list)):
            left_models = model_list[:idx]
            right_models = model_list[idx:]

            left_values = np.concatenate([groups[m] for m in left_models])
            right_values = np.concatenate([groups[m] for m in right_models])

            delta = cliffs_delta(left_values, right_values)
            if abs(delta) < CLIFFS_DELTA_THRESHOLD:
                continue

            n_left = len(left_values)
            n_right = len(right_values)
            score = (
                n_left * (np.mean(left_values) - overall_mean) ** 2
                + n_right * (np.mean(right_values) - overall_mean) ** 2
            )
            if score > best_score:
                best_score = score
                best_split = (left_models, right_models)

        if best_split is None:
            return [model_list]

        left_group, right_group = best_split
        return split(left_group) + split(right_group)

    ordered_models = sorted(
        groups.keys(), key=lambda m: np.mean(groups[m]), reverse=True
    )
    raw_groups = split(ordered_models)

    # Sort groups by mean for rank assignment
    ranked_groups = sorted(
        raw_groups,
        key=lambda models: np.mean(np.concatenate([groups[m] for m in models])),
        reverse=True,
    )

    ranks = {}
    for rank, model_names in enumerate(ranked_groups, start=1):
        for model in model_names:
            ranks[model] = rank
    return ranks


def perform_scott_knott(data, metric):
    """Wrapper to compute Scott-Knott ranks for a given metric."""
    print(f"\n--- Performing Scott-Knott Ranking for {metric.upper()} ---")
    metric_groups = {
        model: group[metric].values
        for model, group in data.groupby("model")
    }

    ranks = scott_knott(metric_groups)
    for model, rank in sorted(ranks.items(), key=lambda x: x[1]):
        mean_score = np.mean(metric_groups[model])
        print(f"  Rank {rank}: {model} (mean={mean_score:.4f})")
    return ranks

def plot_distributions(data, metric, output_file):
    """Plot the performance distribution of models"""
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=data, x='model', y=metric)
    sns.stripplot(data=data, x='model', y=metric, color=".25", jitter=True, dodge=True)
    
    plt.title(f"Performance Distribution of Models ({metric.upper()})", fontsize=16, fontweight='bold')
    plt.ylabel(metric.upper(), fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300)
    print(f"\n✓ Performance distribution plot saved to: {output_file}")
    plt.close()

def main():
    print("="*70)
    print("PART 2: STATISTICAL RANKING WITH SCOTT-KNOTT")
    print("="*70)
    
    # --- 1. Load Data ---
    results_dir = Path(RESULTS_DIR)
    input_path = results_dir / "resampling_results.json"
    
    if not input_path.exists():
        print(f"Error: Resampling results not found at {input_path}")
        print("Please run scripts/02_resample_evaluate.py first.")
        return
        
    with open(input_path, 'r') as f:
        resampling_data = json.load(f)
        
    # --- 2. Prepare DataFrame ---
    records = []
    for run_data in resampling_data:
        run_id = run_data['run']
        for model_name, metrics in run_data.items():
            if model_name == 'run': continue
            records.append({
                "run": run_id,
                "model": model_name,
                "f1_score": metrics['f1_score'],
                "auc": metrics['auc']
            })
            
    df = pd.DataFrame(records)
    print("Loaded performance data for all models over 10 runs.")
    
    # --- 3. Perform Scott-Knott for F1-Score and AUC ---
    f1_ranks = perform_scott_knott(df, 'f1_score')
    auc_ranks = perform_scott_knott(df, 'auc')
    
    # --- 4. Save Ranks ---
    ranking_summary = (
        pd.DataFrame(
            {
                "Model": list(f1_ranks.keys()),
                "F1_Rank": list(f1_ranks.values()),
                "AUC_Rank": [auc_ranks[model] for model in f1_ranks.keys()],
            }
        )
        .sort_values(by=["F1_Rank", "AUC_Rank", "Model"])
        .reset_index(drop=True)
    )
    
    output_csv = results_dir / "scott_knott_ranking.csv"
    ranking_summary.to_csv(output_csv, index=False)
    
    print(f"\n--- Final Ranking Summary ---")
    print(ranking_summary.to_string(index=False))
    print(f"\n✓ Ranking summary saved to: {output_csv}")
    
    # --- 5. Create Visualizations ---
    plot_distributions(df, 'f1_score', results_dir / "f1_score_distribution.png")
    plot_distributions(df, 'auc', results_dir / "auc_distribution.png")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
