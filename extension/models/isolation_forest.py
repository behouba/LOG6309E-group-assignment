"""
Isolation Forest wrapper used by the extension scripts.

The class provides a thin abstraction around scikit-learn's IsolationForest
estimator so that downstream scripts can call `fit`, `evaluate`, and
`print_evaluation` in a consistent way with the other detectors.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _score_to_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray,
) -> Dict[str, Any]:
    """Compute the standard set of metrics used across the extension."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, anomaly_scores)
    except ValueError:
        auc = float("nan")

    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

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


class IsolationForestDetector:
    """
    Wrapper around scikit-learn's IsolationForest with evaluation helpers.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Any = "auto",
        contamination: float = 0.1,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.extra_params: Dict[str, Any] = dict(kwargs)
        params = {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "random_state": self.random_state,
            **self.extra_params,
        }
        self.model = IsolationForest(**params)
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """Fit the underlying Isolation Forest model."""
        self.model.fit(X)
        self._is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher means more anomalous)."""
        if not self._is_fitted:
            raise RuntimeError("IsolationForestDetector must be fitted before scoring.")

        # scikit-learn returns higher scores for inliers; invert to match anomaly logic
        raw_scores = self.model.decision_function(X)
        return -raw_scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (1 for anomaly, 0 for normal)."""
        if not self._is_fitted:
            raise RuntimeError("IsolationForestDetector must be fitted before predicting.")

        raw_predictions = self.model.predict(X)
        return np.where(raw_predictions == -1, 1, 0)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model and return a metrics dictionary."""
        scores = self.decision_function(X)
        predictions = self.predict(X)
        return _score_to_metrics(y_true, predictions, scores)

    def print_evaluation(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model, print a short summary, and return metrics."""
        metrics = self.evaluate(X, y_true)
        print(
            "Isolation Forest Metrics - "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1_score']:.4f}, "
            f"AUC: {metrics['auc']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.4f}"
        )
        return metrics
