import sys
import json
import numpy as np
from pathlib import Path

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
    print(f"Loading {DATASET} ({DATA_MODE}) MCV features from {MCV_PATH}")

    data = np.load(MCV_PATH, allow_pickle=True)

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    anomaly_rate = np.sum(y_train == 1) / max(len(y_train), 1)
    print(f"Train sessions: {x_train.shape[0]}, anomalies: {np.sum(y_train == 1)} ({anomaly_rate*100:.2f}%)")
    print(f"Test sessions:  {x_test.shape[0]}, anomalies: {np.sum(y_test == 1)}")

    return x_train, y_train, x_test, y_test


def train_isolation_forest(x_train, y_train, x_test, y_test):
    print("Fit Isolation Forest")

    # Calculate contamination from training data
    contamination = np.sum(y_train == 1) / len(y_train)
    print(f"Contamination rate set to {contamination:.4f}")

    config = ISOLATION_FOREST_CONFIG.copy()
    config['contamination'] = contamination
    model = IsolationForestDetector(**config)
    model.fit(x_train)

    print("Evaluate Isolation Forest")
    metrics = model.print_evaluation(x_test, y_test)
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"isolation_forest_{DATASET}_{DATA_MODE}_results.json"
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Isolation Forest metrics stored in {output_path}")

    return model, metrics


def train_autoencoder(x_train, y_train, x_test, y_test):
    print("Fit Autoencoder")

    model = AutoencoderDetector(**AUTOENCODER_CONFIG)
    model.fit(x_train)

    print("Evaluate Autoencoder")
    metrics = model.print_evaluation(x_test, y_test)
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"autoencoder_{DATASET}_{DATA_MODE}_results.json"
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Autoencoder metrics stored in {output_path}")

    return model, metrics


def save_summary(iforest_metrics, autoencoder_metrics):
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

    print("Summary (F1 / AUC)")
    print(f"  Isolation Forest: {iforest_metrics['f1_score']:.4f} / {iforest_metrics['auc']:.4f}")
    print(f"  Autoencoder:      {autoencoder_metrics['f1_score']:.4f} / {autoencoder_metrics['auc']:.4f}")
    print(f"Summary saved to {output_path}")


def main():
    print("Train Isolation Forest and Autoencoder on MCV features")

    x_train, y_train, x_test, y_test = load_data()

    iforest_model, iforest_metrics = train_isolation_forest(
        x_train, y_train, x_test, y_test
    )

    autoencoder_model, autoencoder_metrics = train_autoencoder(
        x_train, y_train, x_test, y_test
    )
    save_summary(iforest_metrics, autoencoder_metrics)

    print("Unsupervised models prepared; continue with resampling, ranking, and explanation steps as needed.")


if __name__ == "__main__":
    main()
