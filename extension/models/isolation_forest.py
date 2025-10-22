import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    accuracy_score
)


class IsolationForestDetector:
    def __init__(self, n_estimators=100, max_samples="auto",
                 contamination=0.029, random_state=42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        self.model = SklearnIsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

        self.is_fitted = False

    def fit(self, X, y=None):
        print("Training Isolation Forest")
        print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}, Contamination: {self.contamination}")
        self.model.fit(X)
        self.is_fitted = True
        print("Model fitted successfully.")
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Convert IsolationForest output (-1=anomaly, 1=normal) to 0/1
        predictions = self.model.predict(X)
        y_pred = np.where(predictions == -1, 1, 0)

        return y_pred

    def decision_function(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        scores = self.model.decision_function(X)
        # Invert so higher score = more anomalous
        return -scores

    def evaluate(self, X, y_true):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        y_pred = self.predict(X)
        scores = self.decision_function(X)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        try:
            auc = roc_auc_score(y_true, scores)
        except ValueError:
            auc = 0.0

        accuracy = accuracy_score(y_true, y_pred)
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
        metrics = self.evaluate(X, y_true)

        print("Isolation Forest evaluation:")
        print(f"  Precision {metrics['precision']:.4f}, Recall {metrics['recall']:.4f}, F1 {metrics['f1_score']:.4f}, AUC {metrics['auc']:.4f}, Accuracy {metrics['accuracy']:.4f}")
        cm = metrics['confusion_matrix']
        print(f"  Confusion matrix: TN {cm['TN']}, FP {cm['FP']}, FN {cm['FN']}, TP {cm['TP']}")

        return metrics


if __name__ == "__main__":
    print("Isolation Forest Model Implementation")
    print("Use scripts/01_train_unsupervised.py to train the model")
