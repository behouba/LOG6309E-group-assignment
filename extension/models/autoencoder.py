"""
Feed-forward autoencoder used as an unsupervised anomaly detector.

This module supplies a small PyTorch-based autoencoder that mirrors the
implementation used during the replication effort, but wrapped in a class
that offers the same API as other detectors in this extension.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully at runtime
    TORCH_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _ensure_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for the autoencoder detector. "
            "Install it with `pip install torch` inside the virtual environment."
        )


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, scores)
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


class FeedForwardAutoencoder(nn.Module):
    """Simple symmetric autoencoder for tabular data."""

    def __init__(self, input_dim: int, hidden_dims: List[int], encoding_dim: int):
        super().__init__()

        encoder_layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        prev_dim = encoding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(inputs)
        return self.decoder(latent)


class AutoencoderDetector:
    """
    Autoencoder anomaly detector with percentile-based thresholding.
    """

    def __init__(
        self,
        encoding_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        epochs: int = 30,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        threshold_percentile: float = 95.0,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
        **_,
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [64, 32]
        self.encoding_dim = encoding_dim
        self.hidden_dims = list(hidden_dims)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self.device_str = device

        self.model: Optional[FeedForwardAutoencoder] = None
        self.threshold_: Optional[float] = None
        self.device: Optional[torch.device] = None
        self._fitted = False

    def _set_seed(self) -> None:
        if self.random_state is None:
            return
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _prepare(self, X: np.ndarray) -> torch.Tensor:
        _ensure_torch()
        if X.ndim != 2:
            raise ValueError(
                "AutoencoderDetector expects 2D tabular input of shape (n_samples, n_features)."
            )
        tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
        return tensor

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        _ensure_torch()
        features = self._prepare(X)
        input_dim = features.shape[1]

        self._set_seed()
        self.device = torch.device(
            self.device_str if self.device_str else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = FeedForwardAutoencoder(input_dim, self.hidden_dims, self.encoding_dim).to(
            self.device
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        dataset = TensorDataset(features)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                reconstruction = self.model(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(loader), 1)
            print(f"[Autoencoder] Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.6f}")

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(features.to(self.device)).cpu()
        reconstruction_errors = torch.mean((reconstructed - features) ** 2, dim=1).numpy()

        self.threshold_ = float(np.percentile(reconstruction_errors, self.threshold_percentile))
        self._fitted = True
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self.model is None or self.threshold_ is None:
            raise RuntimeError("AutoencoderDetector must be fitted before scoring.")

        features = self._prepare(X)
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(features.to(self.device)).cpu()
        errors = torch.mean((reconstructed - features) ** 2, dim=1).numpy()
        return errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._score(X)
        return (scores > self.threshold_).astype(int)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        scores = self._score(X)
        predictions = (scores > self.threshold_).astype(int)
        return _compute_metrics(y_true, predictions, scores)

    def print_evaluation(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        metrics = self.evaluate(X, y_true)
        print(
            "Autoencoder Metrics - "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1_score']:.4f}, "
            f"AUC: {metrics['auc']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.4f}"
        )
        return metrics
