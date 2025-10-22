from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    AUTOENCODER_CONFIG,
    DATASET,
    DATA_MODE,
    ISOLATION_FOREST_CONFIG,
    LSTM_CONFIG,
    MCV_PATH,
    RANDOM_SEED,
    RESULTS_DIR,
    SCOTT_KNOTT_CONFIG,
    WORD2VEC_PATH,
)
from models.autoencoder import AutoencoderDetector
from models.isolation_forest import IsolationForestDetector
from models.lstm import LSTMSequenceClassifier, LSTMTrainingConfig


def build_metrics(y_true, y_pred, y_scores):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = float("nan")

    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc": float(auc),
        "accuracy": float(accuracy),
        "confusion_matrix": {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        },
    }


def evaluate_random_forest(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        y_scores = y_pred
    return build_metrics(y_test, y_pred, y_scores)


def main():
    print("Resampling evaluation for Random Forest, LSTM, Isolation Forest, and Autoencoder")
    print("Loading MCV and Word2Vec data")
    try:
        mcv_data = np.load(MCV_PATH, allow_pickle=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"MCV data not found at {MCV_PATH}") from exc

    try:
        w2v_data = np.load(WORD2VEC_PATH, allow_pickle=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"Word2Vec data not found at {WORD2VEC_PATH}") from exc

    X_mcv = mcv_data["x_train"]
    y = mcv_data["y_train"]
    X_w2v = w2v_data["x_train"]
    session_ids = w2v_data["train_session_ids"]

    if len(X_mcv) != len(X_w2v):
        raise SystemExit(
            "Mismatch between MCV and Word2Vec training samples. "
            "Re-run Part 1 preprocessing to regenerate aligned datasets."
        )

    print(f"Training sessions: {X_mcv.shape[0]} (dataset {DATASET}, mode {DATA_MODE})")

    n_resamples = SCOTT_KNOTT_CONFIG["n_resamples"]
    test_size = SCOTT_KNOTT_CONFIG["test_size"]
    base_seed = SCOTT_KNOTT_CONFIG["random_seed"]

    all_results = []
    lstm_cfg = LSTMTrainingConfig(**LSTM_CONFIG)

    print(f"Running {n_resamples} resamples (test size {test_size})")

    for i in tqdm(range(n_resamples), desc="Resampling Progress"):
        current_seed = base_seed + i
        indices = np.arange(len(X_mcv))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=current_seed,
            stratify=y,
        )

        X_train = X_mcv[train_idx]
        X_test = X_mcv[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        X_train_w2v = X_w2v[train_idx]
        X_test_w2v = X_w2v[test_idx]
        train_sessions = session_ids[train_idx]
        test_sessions = session_ids[test_idx]

        run_results = {"run": i + 1}
        rf = RandomForestClassifier(
            n_estimators=200, random_state=current_seed, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        run_results["Random Forest"] = evaluate_random_forest(rf, X_test, y_test)

        # LSTM
        lstm = LSTMSequenceClassifier(lstm_cfg)
        lstm.fit(X_train_w2v, y_train, session_ids=train_sessions)
        run_results["LSTM"] = lstm.evaluate(
            X_test_w2v, y_test, session_ids=test_sessions
        )
        contamination = np.sum(y_train == 1) / len(y_train)
        iforest_config = ISOLATION_FOREST_CONFIG.copy()
        iforest_config["contamination"] = contamination
        iforest_config["random_state"] = current_seed
        iforest = IsolationForestDetector(**iforest_config)
        iforest.fit(X_train)
        run_results["Isolation Forest"] = iforest.evaluate(X_test, y_test)
        ae_config = AUTOENCODER_CONFIG.copy()
        ae_config["random_state"] = current_seed
        autoencoder = AutoencoderDetector(**ae_config)
        autoencoder.fit(X_train)
        run_results["Autoencoder"] = autoencoder.evaluate(X_test, y_test)

        all_results.append(run_results)

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "resampling_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(f"Resampling evaluation complete. Results saved to {output_path}")

    df_records = []
    for run in all_results:
        run_id = run["run"]
        for model_name, metrics in run.items():
            if model_name == "run":
                continue
            df_records.append(
                {
                    "run": run_id,
                    "model": model_name,
                    "f1_score": metrics["f1_score"],
                    "auc": metrics["auc"],
                }
            )

    summary = pd.DataFrame(df_records)
    print("Average performance (F1 / AUC):")
    averages = (
        summary.groupby("model")[["f1_score", "auc"]]
        .mean()
        .sort_values(by="f1_score", ascending=False)
    )
    for model_name, row in averages.iterrows():
        print(f"  {model_name}: {row['f1_score']:.4f} / {row['auc']:.4f}")


if __name__ == "__main__":
    main()
