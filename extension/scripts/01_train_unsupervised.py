"""
Train Unsupervised Anomaly Detection Models

Trains two unsupervised models on HDFS dataset:
1. Isolation Forest (Classical)
2. Autoencoder (Deep Learning)

Uses MCV representation from Part 1.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATASET,
    DATA_MODE,
    MCV_PATH,
    RESULTS_DIR,
    ISOLATION_FOREST_CONFIG,
    AUTOENCODER_CONFIG
)
from models.isolation_forest import IsolationForestDetector
from models.autoencoder import AutoencoderDetector


def load_data():
    """Load MCV representation from Part 1"""
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    print(f"Dataset: {DATASET}")
    print(f"Mode: {DATA_MODE}")
    print(f"Representation: MCV (Message Count Vector)")
    print(f"Loading from: {MCV_PATH}")

    data = np.load(MCV_PATH, allow_pickle=True)

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    print(f"\nData shapes:")
    print(f"  x_train: {x_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  x_test: {x_test.shape}")
    print(f"  y_test: {y_test.shape}")

    print(f"\nClass distribution:")
    print(f"  Train - Normal: {np.sum(y_train == 0)}, Anomaly: {np.sum(y_train == 1)}")
    print(f"  Test  - Normal: {np.sum(y_test == 0)}, Anomaly: {np.sum(y_test == 1)}")

    anomaly_rate = np.sum(y_train == 1) / len(y_train)
    print(f"  Anomaly rate: {anomaly_rate:.4f} ({anomaly_rate*100:.2f}%)")

    return x_train, y_train, x_test, y_test


def train_isolation_forest(x_train, y_train, x_test, y_test):
    """Train and evaluate Isolation Forest"""
    print("\n" + "="*70)
    print("TRAINING ISOLATION FOREST")
    print("="*70)

    # Calculate contamination from training data
    contamination = np.sum(y_train == 1) / len(y_train)
    print(f"Contamination rate: {contamination:.4f}")

    # Override config contamination with actual rate
    config = ISOLATION_FOREST_CONFIG.copy()
    config['contamination'] = contamination

    # Initialize and train model
    model = IsolationForestDetector(**config)
    model.fit(x_train)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = model.print_evaluation(x_test, y_test)

    # Save results
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"isolation_forest_{DATASET}_{DATA_MODE}_results.json"
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\n✓ Results saved to: {output_path}")

    return model, metrics


def train_autoencoder(x_train, y_train, x_test, y_test):
    """Train and evaluate Autoencoder"""
    print("\n" + "="*70)
    print("TRAINING AUTOENCODER")
    print("="*70)

    # Initialize and train model
    model = AutoencoderDetector(**AUTOENCODER_CONFIG)
    model.fit(x_train)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = model.print_evaluation(x_test, y_test)

    # Save results
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"autoencoder_{DATASET}_{DATA_MODE}_results.json"
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\n✓ Results saved to: {output_path}")

    return model, metrics


def save_summary(iforest_metrics, autoencoder_metrics):
    """Save combined summary of both models"""
    summary = {
        "dataset": DATASET,
        "data_mode": DATA_MODE,
        "representation": "MCV",
        "models": {
            "isolation_forest": {
                "category": "classical_unsupervised",
                "precision": iforest_metrics['precision'],
                "recall": iforest_metrics['recall'],
                "f1_score": iforest_metrics['f1_score'],
                "auc": iforest_metrics['auc'],
                "accuracy": iforest_metrics['accuracy']
            },
            "autoencoder": {
                "category": "deep_learning_unsupervised",
                "precision": autoencoder_metrics['precision'],
                "recall": autoencoder_metrics['recall'],
                "f1_score": autoencoder_metrics['f1_score'],
                "auc": autoencoder_metrics['auc'],
                "accuracy": autoencoder_metrics['accuracy']
            }
        }
    }

    results_dir = Path(RESULTS_DIR)
    output_path = results_dir / f"unsupervised_summary_{DATASET}_{DATA_MODE}.json"

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print("\n" + "="*70)
    print("SUMMARY - UNSUPERVISED MODELS")
    print("="*70)
    print(f"\nIsolation Forest:")
    print(f"  F1-Score: {iforest_metrics['f1_score']:.4f}")
    print(f"  AUC:      {iforest_metrics['auc']:.4f}")

    print(f"\nAutoencoder:")
    print(f"  F1-Score: {autoencoder_metrics['f1_score']:.4f}")
    print(f"  AUC:      {autoencoder_metrics['auc']:.4f}")

    print(f"\n✓ Summary saved to: {output_path}")
    print("="*70)


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("PART 2: UNSUPERVISED ANOMALY DETECTION")
    print("="*70)
    print(f"Training two unsupervised models:")
    print(f"  1. Isolation Forest (Classical)")
    print(f"  2. Autoencoder (Deep Learning)")
    print("="*70 + "\n")

    # Load data
    x_train, y_train, x_test, y_test = load_data()

    # Train Isolation Forest
    iforest_model, iforest_metrics = train_isolation_forest(
        x_train, y_train, x_test, y_test
    )

    # Train Autoencoder
    autoencoder_model, autoencoder_metrics = train_autoencoder(
        x_train, y_train, x_test, y_test
    )

    # Save summary
    save_summary(iforest_metrics, autoencoder_metrics)

    print("\n" + "="*70)
    print("✓ UNSUPERVISED MODEL TRAINING COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run resampling evaluation: python scripts/02_resample_evaluate.py")
    print("  2. Statistical ranking: python scripts/03_scott_knott_ranking.py")
    print("  3. Model explanation: python scripts/04_model_explanation.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
