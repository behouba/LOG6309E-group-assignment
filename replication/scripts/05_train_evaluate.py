import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_experiment_name
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                            roc_curve, confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    # Precision, Recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    # AUC
    auc = None
    if y_pred_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except:
            pass

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'accuracy': (tp + tn) / (tp + tn + fp + fn)
    }

    return metrics


def print_metrics(metrics, model_name="Model"):
    print(f"\n{model_name} Performance:")
    print("-" * 50)
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1_score']:.4f}")
    if metrics['auc'] is not None:
        print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={metrics['true_negative']}, FP={metrics['false_positive']}")
    print(f"    FN={metrics['false_negative']}, TP={metrics['true_positive']}")


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42, **kwargs):
    print(f"\nTraining Random Forest...")
    print(f"  n_estimators: {n_estimators}")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  Class distribution: {np.bincount(y_train)}")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,  # Use all cores
        **kwargs
    )

    model.fit(X_train, y_train)
    print("  Training completed!")

    return model


def train_logistic_regression(X_train, y_train, C=100, max_iter=1000, random_state=42):
    print(f"\nTraining Logistic Regression...")
    print(f"  C: {C}")
    print(f"  X_train shape: {X_train.shape}")

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("  Training completed!")

    return model


def train_svm(X_train, y_train, C=1.0, kernel='rbf', random_state=42):
    print(f"\nTraining SVM...")
    print(f"  C: {C}")
    print(f"  Kernel: {kernel}")
    print(f"  X_train shape: {X_train.shape}")

    model = SVC(
        C=C,
        kernel=kernel,
        random_state=random_state,
        probability=True  # Enable probability estimates for AUC
    )

    model.fit(X_train, y_train)
    print("  Training completed!")

    return model


def train_decision_tree(X_train, y_train, max_depth=None, random_state=42):
    print(f"\nTraining Decision Tree...")
    print(f"  max_depth: {max_depth}")
    print(f"  X_train shape: {X_train.shape}")

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )

    model.fit(X_train, y_train)
    print("  Training completed!")

    return model


def evaluate_classical_model(model, X_test, y_test, model_name="Model"):
    print(f"\nEvaluating {model_name}...")

    # Predictions
    y_pred = model.predict(X_test)

    # Probability predictions (if available)
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        # Check if we have both classes (binary classification)
        if proba.shape[1] > 1:
            y_pred_proba = proba[:, 1]
        else:
            # Only one class - use those probabilities
            y_pred_proba = proba[:, 0]
            print(f"  Warning: Model only learned one class. AUC may not be meaningful.")

    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)

    print_metrics(metrics, model_name)

    return metrics


def plot_roc_curve(y_true, y_pred_proba, model_name, output_file):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"  ROC curve saved to: {output_file}")


def save_results(results, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {output_file}")


def compare_results(results_dict):
    print("\n" + "="*60)
    print("Performance Comparison")
    print("="*60)

    # Create comparison table
    comparison_data = []

    for config_name, metrics in results_dict.items():
        comparison_data.append({
            'Configuration': config_name,
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'AUC': f"{metrics['auc']:.4f}" if metrics['auc'] else 'N/A',
            'Accuracy': f"{metrics['accuracy']:.4f}"
        })

    df = pd.DataFrame(comparison_data)
    print("\n" + df.to_string(index=False))

    return df


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Import full config to expose dataset name for logging
    import config

    print("="*60)
    print("Model Training and Evaluation Pipeline")
    print("="*60)
    print(f"Dataset: {config.DATASET}")
    print(f"Mode: {config.DATA_MODE}")

    # Load data
    repr_dir = project_root / "data" / "representations"
    experiment_name = get_experiment_name()

    # 1. Load MCV data (without feature selection)
    mcv_file = repr_dir / f"{experiment_name}_MCV.npz"
    mcv_fs_file = repr_dir / f"{experiment_name}_MCV_feature_selected.npz"

    results = {}

    # ========================================
    # Classical Model: Random Forest
    # ========================================
    print("\n" + "="*60)
    print("CLASSICAL MODEL: Random Forest")
    print("="*60)

    if mcv_file.exists():
        print(f"\n1. Training WITHOUT feature selection")
        print("-" * 60)

        data = np.load(mcv_file, allow_pickle=True)
        X_train = data['x_train']
        y_train = data['y_train']
        X_test = data['x_test']
        y_test = data['y_test']

        # Train model
        rf_model = train_random_forest(X_train, y_train, n_estimators=100)

        # Evaluate
        metrics = evaluate_classical_model(rf_model, X_test, y_test, "Random Forest (No FS)")
        results['RF_NoFS'] = metrics

        # Plot ROC curve
        if metrics['auc']:
            y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
            plot_file = project_root / "results" / f"roc_{experiment_name}_rf_no_fs.png"
            plot_roc_curve(y_test, y_pred_proba, "Random Forest (No FS)", plot_file)

    if mcv_fs_file.exists():
        print(f"\n2. Training WITH feature selection")
        print("-" * 60)

        data = np.load(mcv_fs_file, allow_pickle=True)
        X_train = data['x_train']
        y_train = data['y_train']
        X_test = data['x_test']
        y_test = data['y_test']

        # Train model
        rf_model_fs = train_random_forest(X_train, y_train, n_estimators=100)

        # Evaluate
        metrics = evaluate_classical_model(rf_model_fs, X_test, y_test, "Random Forest (With FS)")
        results['RF_WithFS'] = metrics

        # Plot ROC curve
        if metrics['auc']:
            y_pred_proba = rf_model_fs.predict_proba(X_test)[:, 1]
            plot_file = project_root / "results" / f"roc_{experiment_name}_rf_with_fs.png"
            plot_roc_curve(y_test, y_pred_proba, "Random Forest (With FS)", plot_file)

    # ========================================
    # Additional Classical Models (Optional)
    # ========================================
    print("\n" + "="*60)
    print("ADDITIONAL CLASSICAL MODELS (Optional)")
    print("="*60)

    if mcv_file.exists():
        print("\nTraining Logistic Regression...")
        data = np.load(mcv_file, allow_pickle=True)
        X_train = data['x_train']
        y_train = data['y_train']
        X_test = data['x_test']
        y_test = data['y_test']

        lr_model = train_logistic_regression(X_train, y_train)
        metrics = evaluate_classical_model(lr_model, X_test, y_test, "Logistic Regression")
        results['LR'] = metrics

        print("\nTraining Decision Tree...")
        dt_model = train_decision_tree(X_train, y_train, max_depth=10)
        metrics = evaluate_classical_model(dt_model, X_test, y_test, "Decision Tree")
        results['DT'] = metrics

    # ========================================
    # Deep Learning Model: LSTM
    # ========================================
    print("\n" + "="*60)
    print("DEEP LEARNING MODEL: LSTM")
    print("="*60)
    print("\nNote: LSTM implementation requires PyTorch and event-level embeddings.")
    print("LSTM training is implemented in a separate script due to complexity.")
    print("See material/models/LSTM.py for reference implementation.")

    # ========================================
    # Results Summary
    # ========================================
    if results:
        comparison_df = compare_results(results)

        # Save results
        results_file = project_root / "results" / f"{experiment_name}_classical_results.json"
        save_results(results, results_file)

        # Save comparison table
        table_file = project_root / "results" / f"{experiment_name}_comparison_table.csv"
        comparison_df.to_csv(table_file, index=False)
        print(f"Comparison table saved to: {table_file}")

    print("\n" + "="*60)
    print("Training and evaluation completed!")
    print("="*60)


if __name__ == "__main__":
    main()
