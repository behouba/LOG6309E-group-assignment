"""
Automated Comparison with Original Paper Results

Compares replication results with Wu et al. (2023) paper Tables 2 & 3.
Generates side-by-side comparison tables and visualizations.

Reference:
Wu, X., Li, H., & Khomh, F. (2023). On the effectiveness of log representation
for log-based anomaly detection. Empirical Software Engineering, 28, 137.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Original paper results from Tables 2 and 3
PAPER_RESULTS = {
    "HDFS": {
        # Table 2 - HDFS results (Random Forest with MCV representation)
        "Random_Forest_MCV": {
            "precision": 1.000,
            "recall": 1.000,
            "f1_score": 0.999,
            "auc": None  # Not always reported in paper
        },
        # LSTM with Word2Vec (from Table 2)
        "LSTM_Word2Vec": {
            "precision": 0.997,
            "recall": 0.921,
            "f1_score": 0.958,
            "auc": None
        }
    },
    "BGL": {
        # Table 3 - BGL results (Random Forest with MCV representation)
        "Random_Forest_MCV": {
            "precision": 0.959,
            "recall": 0.921,
            "f1_score": 0.939,
            "auc": None
        },
        # LSTM with Word2Vec (from Table 3)
        "LSTM_Word2Vec": {
            "precision": 0.871,
            "recall": 0.914,
            "f1_score": 0.865,
            "auc": None
        }
    }
}


def load_replication_results(results_dir: Path) -> dict:
    """
    Load replication results from JSON files

    Args:
        results_dir: Path to results directory

    Returns:
        Dictionary with replication results
    """
    results = {}

    for dataset in PAPER_RESULTS.keys():
        dataset_results = {}

        # Attempt to load per-dataset classical results
        exp_names = [
            f"{dataset}_full",
            f"{dataset}_sample",
            dataset.lower()
        ]

        classical_data = None
        selected_exp = None

        for exp_name in exp_names:
            candidate = results_dir / f"{exp_name}_classical_results.json"
            if candidate.exists():
                with open(candidate, 'r') as f:
                    classical_data = json.load(f)
                selected_exp = exp_name
                break

        if classical_data:
            # Prefer Random Forest without feature selection to match paper setup
            rf_metrics = classical_data.get("RF_NoFS") or classical_data.get("Random_Forest_MCV")
            if rf_metrics:
                dataset_results["Random_Forest_MCV"] = {
                    "precision": rf_metrics.get("precision"),
                    "recall": rf_metrics.get("recall"),
                    "f1_score": rf_metrics.get("f1_score"),
                    "auc": rf_metrics.get("auc")
                }
            # Optionally capture FS variant
            rf_fs_metrics = classical_data.get("RF_WithFS")
            if rf_fs_metrics:
                dataset_results["Random_Forest_MCV_WithFS"] = {
                    "precision": rf_fs_metrics.get("precision"),
                    "recall": rf_fs_metrics.get("recall"),
                    "f1_score": rf_fs_metrics.get("f1_score"),
                    "auc": rf_fs_metrics.get("auc")
                }

        # Load LSTM results
        lstm_file = None
        if selected_exp:
            lstm_candidate = results_dir / f"lstm_{selected_exp}_results.json"
            if lstm_candidate.exists():
                lstm_file = lstm_candidate

        if not lstm_file:
            fallback = results_dir / f"lstm_{dataset}_full_results.json"
            if fallback.exists():
                lstm_file = fallback

        if lstm_file and lstm_file.exists():
            with open(lstm_file, 'r') as f:
                lstm_metrics = json.load(f)
            dataset_results["LSTM_Word2Vec"] = {
                "precision": lstm_metrics.get("precision"),
                "recall": lstm_metrics.get("recall"),
                "f1_score": lstm_metrics.get("f1_score"),
                "auc": lstm_metrics.get("auc")
            }

        if dataset_results:
            results[dataset] = dataset_results

    return results


def calculate_differences(paper_val, replication_val):
    """
    Calculate absolute and percentage difference

    Args:
        paper_val: Value from paper
        replication_val: Value from replication

    Returns:
        Tuple of (absolute_diff, percentage_diff)
    """
    if paper_val is None or replication_val is None:
        return None, None

    abs_diff = replication_val - paper_val
    pct_diff = (abs_diff / paper_val * 100) if paper_val != 0 else 0

    return abs_diff, pct_diff


def create_comparison_table(paper_results: dict, replication_results: dict,
                           dataset: str) -> pd.DataFrame:
    """
    Create comparison table for a dataset

    Args:
        paper_results: Results from paper
        replication_results: Results from replication
        dataset: Dataset name

    Returns:
        Comparison DataFrame
    """
    rows = []

    paper_data = paper_results.get(dataset, {})
    replication_data = replication_results.get(dataset, {})

    for model_name in paper_data.keys():
        paper_metrics = paper_data[model_name]

        # Find matching replication model (with or without FS suffix)
        replication_metrics = None
        for rep_model in replication_data.keys():
            if model_name in rep_model or rep_model.startswith(model_name):
                replication_metrics = replication_data[rep_model]
                break

        if replication_metrics is None:
            replication_metrics = replication_data.get(model_name, {
                "precision": None, "recall": None, "f1_score": None, "auc": None
            })

        # Create row for each metric
        for metric in ["precision", "recall", "f1_score", "auc"]:
            paper_val = paper_metrics.get(metric)
            rep_val = replication_metrics.get(metric)

            if paper_val is not None or rep_val is not None:
                abs_diff, pct_diff = calculate_differences(paper_val, rep_val)

                row = {
                    "Model": model_name,
                    "Metric": metric.replace("_", " ").title(),
                    "Paper": f"{paper_val:.4f}" if paper_val is not None else "N/A",
                    "Replication": f"{rep_val:.4f}" if rep_val is not None else "N/A",
                    "Abs Diff": f"{abs_diff:+.4f}" if abs_diff is not None else "N/A",
                    "% Diff": f"{pct_diff:+.2f}%" if pct_diff is not None else "N/A"
                }
                rows.append(row)

    return pd.DataFrame(rows)


def plot_comparison(paper_results: dict, replication_results: dict,
                   dataset: str, output_file: Path):
    """
    Create visual comparison plot

    Args:
        paper_results: Results from paper
        replication_results: Results from replication
        dataset: Dataset name
        output_file: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{dataset} Dataset: Replication vs. Paper Results', fontsize=16, fontweight='bold')

    metrics = ['precision', 'recall', 'f1_score']
    metric_names = ['Precision', 'Recall', 'F1-Score']

    paper_data = paper_results.get(dataset, {})
    replication_data = replication_results.get(dataset, {})

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        # Collect data
        models = []
        paper_values = []
        replication_values = []

        for model_name in paper_data.keys():
            paper_val = paper_data[model_name].get(metric)

            # Find matching replication model
            rep_val = None
            for rep_model in replication_data.keys():
                if model_name in rep_model or rep_model.startswith(model_name):
                    rep_val = replication_data[rep_model].get(metric)
                    break

            if paper_val is not None and rep_val is not None:
                models.append(model_name.replace("_", "\n"))
                paper_values.append(paper_val)
                replication_values.append(rep_val)

        if models:
            x = np.arange(len(models))
            width = 0.35

            ax.bar(x - width/2, paper_values, width, label='Paper', color='#3498db', alpha=0.8)
            ax.bar(x + width/2, replication_values, width, label='Replication', color='#e74c3c', alpha=0.8)

            ax.set_xlabel('Model', fontweight='bold')
            ax.set_ylabel(metric_name, fontweight='bold')
            ax.set_title(metric_name, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=0, ha='center', fontsize=9)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.1)

            # Add value labels on bars
            for i, (p_val, r_val) in enumerate(zip(paper_values, replication_values)):
                ax.text(i - width/2, p_val + 0.02, f'{p_val:.3f}', ha='center', va='bottom', fontsize=8)
                ax.text(i + width/2, r_val + 0.02, f'{r_val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved comparison plot: {output_file}")


def create_summary_report(paper_results: dict, replication_results: dict,
                         output_file: Path):
    """
    Create summary report comparing all results

    Args:
        paper_results: Results from paper
        replication_results: Results from replication
        output_file: Path to save report
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("REPLICATION VALIDATION REPORT\n")
        f.write("Comparison with Wu et al. (2023) Original Paper Results\n")
        f.write("="*80 + "\n\n")

        for dataset in sorted(paper_results.keys()):
            f.write(f"\n{'='*80}\n")
            f.write(f"{dataset} DATASET\n")
            f.write(f"{'='*80}\n\n")

            paper_data = paper_results[dataset]
            replication_data = replication_results.get(dataset, {})

            for model_name, paper_metrics in paper_data.items():
                f.write(f"\n{model_name}\n")
                f.write("-" * 80 + "\n")

                # Find matching replication model
                rep_metrics = None
                for rep_model in replication_data.keys():
                    if model_name in rep_model or rep_model.startswith(model_name):
                        rep_metrics = replication_data[rep_model]
                        break

                if rep_metrics is None:
                    f.write("  WARNING: No matching replication results found!\n")
                    continue

                # Compare each metric
                f.write(f"{'Metric':<15} {'Paper':<12} {'Replication':<12} {'Abs Diff':<12} {'% Diff':<12}\n")
                f.write("-" * 80 + "\n")

                all_close = True
                for metric in ['precision', 'recall', 'f1_score', 'auc']:
                    paper_val = paper_metrics.get(metric)
                    rep_val = rep_metrics.get(metric)

                    if paper_val is not None and rep_val is not None:
                        abs_diff, pct_diff = calculate_differences(paper_val, rep_val)

                        metric_display = metric.replace("_", " ").title()
                        f.write(f"{metric_display:<15} {paper_val:<12.4f} {rep_val:<12.4f} ")
                        f.write(f"{abs_diff:+12.4f} {pct_diff:+11.2f}%\n")

                        # Check if difference is significant (> 5%)
                        if abs(pct_diff) > 5:
                            all_close = False
                            f.write(f"  ⚠️  WARNING: Large difference detected (>{abs(pct_diff):.1f}%)\n")

                if all_close:
                    f.write("\n  ✓ Results closely match paper values\n")

        f.write(f"\n{'='*80}\n")
        f.write("SUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write("\nReplication Quality Assessment:\n")
        f.write("  - Differences < 5%: Excellent replication\n")
        f.write("  - Differences 5-10%: Good replication (acceptable variance)\n")
        f.write("  - Differences > 10%: Review needed\n\n")

        f.write("Note: Some differences are expected due to:\n")
        f.write("  - Random initialization\n")
        f.write("  - Data sampling/splitting\n")
        f.write("  - Implementation variations\n")
        f.write("  - Hardware/software environment differences\n")

    print(f"  Saved summary report: {output_file}")


def main():
    """Main function"""
    print("\n" + "="*80)
    print("Automated Comparison with Paper Results")
    print("Wu et al. (2023) - Tables 2 & 3")
    print("="*80)

    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / "results"

    # Load replication results
    print("\nLoading replication results...")
    replication_results = load_replication_results(results_dir)

    if not replication_results:
        print("❌ Error: No replication results found!")
        print(f"   Please run experiments first to generate results in: {results_dir}")
        return 1

    print(f"  Found results for datasets: {', '.join(replication_results.keys())}")

    # Create comparison tables
    print("\nGenerating comparison tables...")
    for dataset in PAPER_RESULTS.keys():
        if dataset in replication_results:
            print(f"\n  Processing {dataset}...")

            # Create comparison table
            df = create_comparison_table(PAPER_RESULTS, replication_results, dataset)

            if not df.empty:
                # Save table
                output_csv = results_dir / f"comparison_{dataset}.csv"
                df.to_csv(output_csv, index=False)
                print(f"    Saved comparison table: {output_csv}")

                # Print to console
                print(f"\n    {dataset} Comparison:")
                print("    " + "-"*76)
                print(df.to_string(index=False).replace('\n', '\n    '))
                print()

                # Create visualization
                output_plot = results_dir / f"comparison_{dataset}.png"
                plot_comparison(PAPER_RESULTS, replication_results, dataset, output_plot)
        else:
            print(f"  ⚠️  No replication results found for {dataset}")

    # Create summary report
    print("\nGenerating summary report...")
    summary_file = results_dir / "replication_validation_report.txt"
    create_summary_report(PAPER_RESULTS, replication_results, summary_file)

    # Create combined visualization
    print("\nGenerating combined comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Replication Validation: All Datasets and Metrics',
                 fontsize=16, fontweight='bold')

    dataset_axes = {
        "HDFS": axes[0, :],
        "BGL": axes[1, :]
    }

    for dataset_idx, (dataset, (ax_left, ax_right)) in enumerate(dataset_axes.items()):
        if dataset not in replication_results:
            continue

        paper_data = PAPER_RESULTS.get(dataset, {})
        replication_data = replication_results.get(dataset, {})

        # Plot F1-scores
        models = []
        paper_f1 = []
        rep_f1 = []

        for model_name in paper_data.keys():
            paper_val = paper_data[model_name].get('f1_score')

            rep_val = None
            for rep_model in replication_data.keys():
                if model_name in rep_model or rep_model.startswith(model_name):
                    rep_val = replication_data[rep_model].get('f1_score')
                    break

            if paper_val is not None and rep_val is not None:
                models.append(model_name.replace("_", "\n"))
                paper_f1.append(paper_val)
                rep_f1.append(rep_val)

        if models:
            x = np.arange(len(models))
            width = 0.35

            ax_left.bar(x - width/2, paper_f1, width, label='Paper', color='#3498db', alpha=0.8)
            ax_left.bar(x + width/2, rep_f1, width, label='Replication', color='#e74c3c', alpha=0.8)
            ax_left.set_title(f'{dataset} - F1-Score Comparison', fontweight='bold')
            ax_left.set_ylabel('F1-Score', fontweight='bold')
            ax_left.set_xticks(x)
            ax_left.set_xticklabels(models, rotation=0, fontsize=8)
            ax_left.legend()
            ax_left.grid(axis='y', alpha=0.3)
            ax_left.set_ylim(0, 1.1)

            # Percentage differences
            pct_diffs = [((r - p) / p * 100) if p != 0 else 0
                         for p, r in zip(paper_f1, rep_f1)]

            colors = ['green' if abs(d) < 5 else 'orange' if abs(d) < 10 else 'red'
                     for d in pct_diffs]

            ax_right.bar(x, pct_diffs, color=colors, alpha=0.7)
            ax_right.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax_right.axhline(y=5, color='orange', linestyle='--', linewidth=0.5, alpha=0.5)
            ax_right.axhline(y=-5, color='orange', linestyle='--', linewidth=0.5, alpha=0.5)
            ax_right.set_title(f'{dataset} - Percentage Difference', fontweight='bold')
            ax_right.set_ylabel('% Difference', fontweight='bold')
            ax_right.set_xticks(x)
            ax_right.set_xticklabels(models, rotation=0, fontsize=8)
            ax_right.grid(axis='y', alpha=0.3)

            for i, d in enumerate(pct_diffs):
                ax_right.text(i, d + (1 if d > 0 else -1), f'{d:+.1f}%',
                            ha='center', va='bottom' if d > 0 else 'top', fontsize=8)

    plt.tight_layout()
    output_combined = results_dir / "comparison_all_datasets.png"
    plt.savefig(output_combined, dpi=300, bbox_inches='tight')
    print(f"  Saved combined plot: {output_combined}")

    print("\n" + "="*80)
    print("✓ Comparison completed successfully!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - Comparison tables: {results_dir}/comparison_*.csv")
    print(f"  - Comparison plots: {results_dir}/comparison_*.png")
    print(f"  - Summary report: {summary_file}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
