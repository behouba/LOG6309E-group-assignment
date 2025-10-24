"""
LSTM sequence classifier used for session-level anomaly detection.

The implementation mirrors the behaviour of the LSTM pipeline from the
replication project while providing a reusable interface for scripts in this
extension (training, evaluation, printing metrics).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully at runtime
    TORCH_AVAILABLE = False


def _ensure_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for the LSTM sequence classifier. "
            "Install it with `pip install torch` inside the virtual environment."
        )


def _to_numpy_list(values: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Ensure each element is a float32 numpy array."""
    result: List[np.ndarray] = []
    for seq in values:
        arr = np.asarray(seq, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                "Expected each sequence to have shape (n_events, embedding_dim); "
                f"received array with shape {arr.shape}"
            )
        result.append(arr)
    return result


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
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


class SessionWindowDataset(Dataset):
    """Dataset that slices session sequences into fixed-size windows."""

    def __init__(
        self,
        sequences: Sequence[np.ndarray],
        labels: Sequence[int],
        window_size: int,
        stride: int,
        session_ids: Optional[Sequence[object]] = None,
    ) -> None:
        _ensure_torch()

        sequences = _to_numpy_list(sequences)
        labels = np.asarray(labels, dtype=np.int64)

        if session_ids is None:
            self.session_keys = np.arange(len(sequences))
        else:
            self.session_keys = np.array(session_ids, dtype=object)

        if len(sequences) != len(labels):
            raise ValueError("Sequences and labels must have the same length.")

        self.window_size = int(window_size)
        self.stride = int(stride)

        self.embedding_dim: Optional[int] = None
        for seq in sequences:
            if seq.shape[0] > 0:
                self.embedding_dim = seq.shape[1]
                break
        if self.embedding_dim is None:
            raise ValueError("Unable to determine embedding dimension from empty sequences.")

        self.session_label_map = {
            self.session_keys[idx]: int(labels[idx]) for idx in range(len(labels))
        }

        windows: List[np.ndarray] = []
        window_labels: List[int] = []
        window_session_keys: List[object] = []

        for idx, (seq, label) in enumerate(zip(sequences, labels)):
            session_key = self.session_keys[idx]
            if seq.shape[0] == 0:
                padded = np.zeros((self.window_size, self.embedding_dim), dtype=np.float32)
                windows.append(padded)
                window_labels.append(int(label))
                window_session_keys.append(session_key)
                continue

            if seq.shape[0] < self.window_size:
                padding_rows = self.window_size - seq.shape[0]
                padding = np.zeros((padding_rows, self.embedding_dim), dtype=np.float32)
                padded = np.vstack([seq, padding]).astype(np.float32)
                windows.append(padded)
                window_labels.append(int(label))
                window_session_keys.append(session_key)
                continue

            for start in range(0, seq.shape[0] - self.window_size + 1, self.stride):
                window = seq[start : start + self.window_size].astype(np.float32)
                windows.append(window)
                window_labels.append(int(label))
                window_session_keys.append(session_key)

        if not windows:
            raise ValueError("No windows were generated from the provided sequences.")

        self.windows = np.stack(windows, axis=0)
        self.labels = np.array(window_labels, dtype=np.int64)
        self.window_session_keys = np.array(window_session_keys, dtype=object)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        window = torch.from_numpy(self.windows[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        session_key = self.window_session_keys[idx]
        return window, label, session_key


class SessionLSTM(nn.Module):
    """Lightweight LSTM classifier operating on windowed sequences."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(inputs)
        last_hidden = hidden[-1]
        return self.fc(last_hidden)


@dataclass
class LSTMTrainingConfig:
    window_size: int = 32
    stride: int = 32
    hidden_dim: int = 32
    num_layers: int = 1
    num_epochs: int = 4
    batch_size: int = 256
    learning_rate: float = 1e-3
    val_fraction: float = 0.1
    patience: int = 2
    threshold: float = 0.5
    random_state: Optional[int] = None


class LSTMSequenceClassifier:
    """Trains an LSTM on session sequences and reports session-level metrics."""

    def __init__(self, config: LSTMTrainingConfig):
        self.config = config
        self.model: Optional[SessionLSTM] = None
        self.device: Optional[torch.device] = None
        self.criterion: Optional[nn.Module] = None
        self._fitted = False

    def _set_seed(self) -> None:
        if self.config.random_state is None:
            return
        seed = self.config.random_state
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _aggregate_sessions(
        self,
        dataset: SessionWindowDataset,
        session_keys: Iterable[object],
        window_scores: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        score_sum: Dict[object, float] = {}
        counts: Dict[object, int] = {}

        for key, score in zip(session_keys, window_scores):
            if isinstance(key, (np.ndarray, list, tuple)) and len(key) == 1:
                key = key[0]
            if hasattr(key, "item"):
                try:
                    key = key.item()
                except Exception:  # pylint: disable=broad-except
                    pass
            score_sum[key] = score_sum.get(key, 0.0) + float(score)
            counts[key] = counts.get(key, 0) + 1

        y_true: List[int] = []
        y_pred: List[int] = []
        y_score: List[float] = []

        for key in dataset.session_keys:
            total = counts.get(key, 0)
            avg_score = score_sum.get(key, 0.0) / total if total > 0 else 0.0
            y_score.append(avg_score)
            y_pred.append(1 if avg_score >= self.config.threshold else 0)
            y_true.append(dataset.session_label_map[key])

        return (
            np.array(y_true, dtype=np.int64),
            np.array(y_pred, dtype=np.int64),
            np.array(y_score, dtype=np.float32),
        )

    def _train_one_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float]:
        assert self.model is not None and self.criterion is not None
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for windows, labels, _ in loader:
            windows = windows.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(windows)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * windows.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += windows.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def _evaluate_dataset(
        self, dataset: SessionWindowDataset
    ) -> Tuple[float, float, Dict[str, float]]:
        assert self.model is not None and self.criterion is not None
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_scores: List[float] = []
        all_keys: List[object] = []
        with torch.no_grad():
            for windows, labels, keys in loader:
                windows = windows.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(windows)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * windows.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += windows.size(0)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_scores.extend(probs.tolist())
                if isinstance(keys, torch.Tensor):
                    keys = keys.cpu().numpy().tolist()
                else:
                    keys = list(keys)
                all_keys.extend(keys)

        avg_loss = total_loss / max(total, 1)
        window_accuracy = correct / max(total, 1)

        session_y_true, session_y_pred, session_scores = self._aggregate_sessions(
            dataset, all_keys, all_scores
        )
        metrics = _compute_metrics(session_y_true, session_y_pred, session_scores)
        return avg_loss, window_accuracy, metrics

    def fit(
        self,
        X_sequences: Sequence[np.ndarray],
        y_labels: Sequence[int],
        session_ids: Optional[Sequence[object]] = None,
    ) -> "LSTMSequenceClassifier":
        _ensure_torch()

        X_sequences = _to_numpy_list(X_sequences)
        y_labels = np.asarray(y_labels, dtype=np.int64)
        if session_ids is None:
            session_ids = np.arange(len(X_sequences))
        else:
            session_ids = np.array(session_ids, dtype=object)

        if len(X_sequences) != len(y_labels):
            raise ValueError("Sequences and labels must have the same length.")

        self._set_seed()
        indices = np.arange(len(X_sequences))
        if 0 < self.config.val_fraction < 1 and len(X_sequences) > 1:
            stratify = y_labels if len(np.unique(y_labels)) > 1 else None
            train_idx, val_idx = train_test_split(
                indices,
                test_size=self.config.val_fraction,
                random_state=self.config.random_state,
                stratify=stratify,
            )
        else:
            train_idx = indices
            val_idx = None

        train_dataset = SessionWindowDataset(
            [X_sequences[i] for i in train_idx],
            y_labels[train_idx],
            self.config.window_size,
            self.config.stride,
            session_ids=session_ids[train_idx],
        )
        val_dataset = (
            SessionWindowDataset(
                [X_sequences[i] for i in val_idx],
                y_labels[val_idx],
                self.config.window_size,
                self.config.stride,
                session_ids=session_ids[val_idx],
            )
            if val_idx is not None and len(val_idx) > 0
            else None
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SessionLSTM(
            input_dim=train_dataset.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        best_f1 = -1.0
        epochs_without_improvement = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None

        for epoch in range(self.config.num_epochs):
            train_loss, train_acc = self._train_one_epoch(train_loader, optimizer)
            if val_dataset is not None:
                val_loss, val_acc, val_metrics = self._evaluate_dataset(val_dataset)
                monitor_f1 = val_metrics["f1_score"]
                print(
                    f"[LSTM] Epoch {epoch + 1}/{self.config.num_epochs} "
                    f"- train_loss: {train_loss:.4f}, train_acc(win): {train_acc:.4f}, "
                    f"val_loss: {val_loss:.4f}, val_acc(win): {val_acc:.4f}, "
                    f"val_f1: {monitor_f1:.4f}"
                )
            else:
                val_loss, val_acc, val_metrics = self._evaluate_dataset(train_dataset)
                monitor_f1 = val_metrics["f1_score"]
                print(
                    f"[LSTM] Epoch {epoch + 1}/{self.config.num_epochs} "
                    f"- train_loss: {train_loss:.4f}, train_acc(win): {train_acc:.4f}, "
                    f"train_eval_f1: {monitor_f1:.4f}"
                )

            if monitor_f1 > best_f1:
                best_f1 = monitor_f1
                epochs_without_improvement = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    print("[LSTM] Early stopping triggered.")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._fitted = True
        return self

    def evaluate(
        self,
        X_sequences: Sequence[np.ndarray],
        y_labels: Sequence[int],
        session_ids: Optional[Sequence[object]] = None,
    ) -> Dict[str, float]:
        if not self._fitted or self.model is None:
            raise RuntimeError("LSTMSequenceClassifier must be fitted before evaluation.")

        if session_ids is None:
            session_ids = np.arange(len(X_sequences))
        else:
            session_ids = np.array(session_ids, dtype=object)

        dataset = SessionWindowDataset(
            X_sequences,
            y_labels,
            self.config.window_size,
            self.config.stride,
            session_ids=session_ids,
        )
        _, _, metrics = self._evaluate_dataset(dataset)
        return metrics

    def print_evaluation(
        self,
        X_sequences: Sequence[np.ndarray],
        y_labels: Sequence[int],
        session_ids: Optional[Sequence[object]] = None,
    ) -> Dict[str, float]:
        metrics = self.evaluate(X_sequences, y_labels, session_ids=session_ids)
        print(
            "LSTM Metrics - "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1_score']:.4f}, "
            f"AUC: {metrics['auc']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.4f}"
        )
        return metrics
