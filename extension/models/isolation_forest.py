"""
Isolation Forest for Unsupervised Anomaly Detection

Classical unsupervised anomaly detection model based on random forest principles.
Isolates anomalies by randomly partitioning the feature space.

Reference:
Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest.
In ICDM (pp. 413-422). IEEE.
"""

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    accuracy_score
)


class IsolationForestDetector:
    """
    Isolation Forest wrapper for log anomaly detection

    Uses scikit-learn's IsolationForest with appropriate configuration
    for log data with known anomaly contamination rate.
    """

    def __init__(self, n_estimators=100, max_samples="auto",
                 contamination=0.029, random_state=42):
        """
        Initialize Isolation Forest detector

        Args:
            n_estimators: Number of isolation trees
            max_samples: Number of samples to draw for each tree
            contamination: Expected proportion of anomalies (HDFS: 2.9%)
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        self.model = SklearnIsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1  # Use all CPUs
        )

        self.is_fitted = False

    def fit(self, X, y=None):
        """
        Fit the Isolation Forest model

        Args:
            X: Training data (n_samples, n_features)
            y: Not used (unsupervised), kept for API consistency

        Returns:
            self
        """
        print(f"Training Isolation Forest...")
        print(f"  n_estimators: {self.n_estimators}")
        print(f"  max_samples: {self.max_samples}")
        print(f"  contamination: {self.contamination}")
        print(f"  Training samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")

        self.model.fit(X)
        self.is_fitted = True

        print(f"✓ Model fitted successfully")
        return self

    def predict(self, X):
        """
        Predict anomalies

        Args:
            X: Data to predict (n_samples, n_features)

        Returns:
            y_pred: Binary predictions (0=normal, 1=anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # IsolationForest returns -1 for anomalies, 1 for normal
        # Convert to 0=normal, 1=anomaly
        predictions = self.model.predict(X)
        y_pred = np.where(predictions == -1, 1, 0)

        return y_pred

    def decision_function(self, X):
        """
        Get anomaly scores (for AUC calculation)

        Args:
            X: Data to score (n_samples, n_features)

        Returns:
            scores: Anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        # Get anomaly scores (negative for anomalies)
        scores = self.model.decision_function(X)

        # Invert so higher score = more anomalous (for AUC consistency)
        return -scores

    def evaluate(self, X, y_true):
        """
        Evaluate model performance

        Args:
            X: Test data (n_samples, n_features)
            y_true: True labels (0=normal, 1=anomaly)

        Returns:
            metrics_dict: Dictionary with precision, recall, f1, auc, accuracy
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        # Get predictions
        y_pred = self.predict(X)

        # Get anomaly scores for AUC
        scores = self.decision_function(X)

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        try:
            auc = roc_auc_score(y_true, scores)
        except ValueError:
            auc = 0.0  # If only one class present

        accuracy = accuracy_score(y_true, y_pred)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics_dict = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(auc),
            "accuracy": float(accuracy),
            "confusion_matrix": {
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn)
            }
        }

        return metrics_dict

    def print_evaluation(self, X, y_true):
        """
        Print evaluation metrics in readable format

        Args:
            X: Test data
            y_true: True labels
        """
        metrics = self.evaluate(X, y_true)

        print("\n" + "="*60)
        print("ISOLATION FOREST - EVALUATION RESULTS")
        print("="*60)
        print(f"Precision:  {metrics['precision']:.4f}")
        print(f"Recall:     {metrics['recall']:.4f}")
        print(f"F1-Score:   {metrics['f1_score']:.4f}")
        print(f"AUC:        {metrics['auc']:.4f}")
        print(f"Accuracy:   {metrics['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"           Predicted")
        print(f"           Neg   Pos")
        print(f"Actual Neg {cm['TN']:5d} {cm['FP']:5d}")
        print(f"       Pos {cm['FN']:5d} {cm['TP']:5d}")
        print("="*60)

        return metrics


# Example usage
if __name__ == "__main__":
    # This is a standalone test - actual usage is in the training script
    print("Isolation Forest Model Implementation")
    print("Use scripts/01_train_unsupervised.py to train the model")
