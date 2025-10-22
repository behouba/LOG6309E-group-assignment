import numpy as np
import sys
from pathlib import Path
import json
import random

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True

except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. LSTM training unavailable.")


class LogDataset(Dataset):
    def __init__(self, x_data, y_data, window_size=50, stride=50, session_ids=None):
        self.x_data = x_data
        self.y_data = y_data
        self.window_size = window_size
        self.stride = stride

        if session_ids is None:
            self.session_keys = np.arange(len(x_data))
        else:
            self.session_keys = np.array(session_ids, dtype=object)

        # Determine embedding dimension from first non-empty sequence
        self.embedding_dim = None
        for seq in x_data:
            if hasattr(seq, "shape") and len(seq) > 0:
                self.embedding_dim = seq.shape[1]
                break
        if self.embedding_dim is None:
            raise ValueError("No events found in the provided sequences for LSTM training.")

        # Mapping from session key to label for aggregation
        self.session_label_map = {
            self.session_keys[idx]: int(y_data[idx])
            for idx in range(len(self.session_keys))
        }

        self.windows = []
        self.labels = []
        self.window_session_keys = []

        for idx, (seq, label) in enumerate(zip(x_data, y_data)):
            session_key = self.session_keys[idx]

            if len(seq) < window_size:
                padding = np.zeros((window_size - len(seq), self.embedding_dim), dtype=np.float32)
                if len(seq) == 0:
                    seq_padded = np.zeros((window_size, self.embedding_dim), dtype=np.float32)
                else:
                    seq_padded = np.vstack([seq, padding])
                self.windows.append(seq_padded.astype(np.float32))
                self.labels.append(int(label))
                self.window_session_keys.append(session_key)
            else:
                for start_idx in range(0, len(seq) - window_size + 1, stride):
                    window = seq[start_idx:start_idx + window_size]
                    self.windows.append(window.astype(np.float32))
                    self.labels.append(int(label))
                    self.window_session_keys.append(session_key)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.window_session_keys = np.array(self.window_session_keys, dtype=object)

        print(f"Created {len(self.windows)} windows from {len(x_data)} sequences")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.windows[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.window_session_keys[idx]
        )


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=8, num_layers=1, num_classes=2):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # LSTM forward
        # out shape: (batch_size, seq_len, hidden_dim)
        out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        # h_n shape: (num_layers, batch_size, hidden_dim)
        last_hidden = h_n[-1]  # (batch_size, hidden_dim)

        # Fully connected layer
        out = self.fc(last_hidden)  # (batch_size, num_classes)

        return out


def train_lstm(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def evaluate_lstm(test_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []
    all_probs = []
    all_session_keys = []

    with torch.no_grad():
        for data, target, session_keys in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            probs = torch.softmax(output, dim=1)[:, 1]  # Probability of anomaly class

            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            if isinstance(session_keys, (list, tuple)):
                batch_keys = list(session_keys)
            elif isinstance(session_keys, torch.Tensor):
                batch_keys = session_keys.cpu().numpy().tolist()
            else:
                batch_keys = list(session_keys)
            all_session_keys.extend(batch_keys)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    return (
        avg_loss,
        accuracy,
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_session_keys, dtype=object)
    )


def compute_metrics(y_true, y_pred, y_proba):
    from sklearn.metrics import (precision_recall_fscore_support,
                                roc_auc_score, confusion_matrix)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, y_proba)
    except:
        auc = None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc) if auc else None,
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
        'confusion_matrix': {
            'TP': int(tp), 'TN': int(tn),
            'FP': int(fp), 'FN': int(fn)
        }
    }

    return metrics


def aggregate_session_predictions(session_keys, window_probs, dataset, threshold=0.5):
    session_prob_sum = {}
    session_counts = {}

    for key, prob in zip(session_keys, window_probs):
        if isinstance(key, (np.ndarray, list)):
            key = key[0]
        if hasattr(key, "item"):
            try:
                key = key.item()
            except Exception:  # pylint: disable=broad-except
                pass

        session_prob_sum[key] = session_prob_sum.get(key, 0.0) + float(prob)
        session_counts[key] = session_counts.get(key, 0) + 1

    session_true = []
    session_pred = []
    session_probs = []

    for key in dataset.session_keys:
        prob = 0.0
        if key in session_prob_sum and session_counts[key] > 0:
            prob = session_prob_sum[key] / session_counts[key]

        session_probs.append(prob)
        session_pred.append(1 if prob >= threshold else 0)
        session_true.append(dataset.session_label_map[key])

    return (
        np.array(session_true, dtype=np.int64),
        np.array(session_pred, dtype=np.int64),
        np.array(session_probs, dtype=np.float32)
    )


def main():
    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required for LSTM training.")
        print("Install with: pip install torch")
        return

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Import config
    sys.path.insert(0, str(project_root))
    import config

    print("="*60)
    print("LSTM Model Training")
    print("="*60)
    print(f"Dataset: {config.DATASET}")
    print(f"Mode: {config.DATA_MODE}")

    # Configuration
    lstm_cfg = config.LSTM
    window_size = lstm_cfg.get("window_size", 50)
    stride = lstm_cfg.get("stride", 50)
    hidden_dim = lstm_cfg.get("hidden_dim", 8)
    num_epochs = lstm_cfg.get("num_epochs", 10)
    learning_rate = lstm_cfg.get("learning_rate", 0.001)
    batch_size = lstm_cfg.get("batch_size", 32)

    seed = lstm_cfg.get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    repr_dir = project_root / "data" / "representations"
    exp_name = config.get_experiment_name()
    emb_file = repr_dir / f"{exp_name}_Word2Vec.npz"

    if not emb_file.exists():
        print(f"\nError: Embeddings not found at: {emb_file}")
        print("Run 03_generate_representations.py to generate Word2Vec embeddings first.")
        print("\nNote: Word2Vec generation requires gensim package.")
        return

    print(f"\nLoading embeddings from: {emb_file}")
    data = np.load(emb_file, allow_pickle=True)

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    train_session_ids = data['train_session_ids'] if 'train_session_ids' in data.files else np.arange(len(x_train))
    test_session_ids = data['test_session_ids'] if 'test_session_ids' in data.files else np.arange(len(x_test))

    train_session_ids = np.array(train_session_ids, dtype=object)
    test_session_ids = np.array(test_session_ids, dtype=object)

    print(f"Data loaded:")
    print(f"  x_train: {x_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  Embedding dim: {x_train[0].shape[1] if len(x_train[0].shape) > 1 else 'N/A'}")

    if len(x_train[0].shape) < 2:
        print("\nError: Expected event-level embeddings (n_events, emb_dim)")
        print("The loaded data appears to be sequence-level embeddings.")
        return

    embedding_dim = x_train[0].shape[1]

    print("\nCreating datasets...")
    train_dataset = LogDataset(x_train, y_train, window_size, stride, session_ids=train_session_ids)
    test_dataset = LogDataset(x_test, y_test, window_size, stride, session_ids=test_session_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = LSTMModel(
        input_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        num_classes=2
    ).to(device)

    print(f"\nModel architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nTraining for {num_epochs} epochs...")
    print("="*60)

    best_f1 = -1
    best_metrics = None

    for epoch in range(num_epochs):
        train_loss, train_acc = train_lstm(
            train_loader, model, criterion, optimizer, device
        )

        (
            test_loss,
            test_acc_window,
            y_true_windows,
            y_pred_windows,
            y_proba_windows,
            session_keys_windows
        ) = evaluate_lstm(test_loader, model, criterion, device)

        session_y_true, session_y_pred, session_y_proba = aggregate_session_predictions(
            session_keys_windows,
            y_proba_windows,
            test_dataset,
            threshold=0.5
        )

        metrics = compute_metrics(session_y_true, session_y_pred, session_y_proba)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc (window-level): {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc (window-level): {test_acc_window:.2f}%")
        print(f"  Sessions evaluated: {len(session_y_true)}")
        auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else 'N/A'
        print(f"  Session Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
              f"F1: {metrics['f1_score']:.4f}, AUC: {auc_str}")

        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_metrics = metrics.copy()
            best_metrics["threshold"] = 0.5
            best_metrics["num_sessions"] = int(len(session_y_true))
            best_metrics["num_windows"] = int(len(y_true_windows))

            # Save model
            model_dir = project_root / "results" / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / f"lstm_{exp_name}_best.pth")

    # Save results
    if best_metrics is None:
        print("Warning: No valid metrics were recorded during training.")
        return

    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)

    print("\n" + "="*60)
    print("Best Results:")
    print("="*60)
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    print(f"  F1-score:  {best_metrics['f1_score']:.4f}")
    if best_metrics['auc'] is not None:
        print(f"  AUC:       {best_metrics['auc']:.4f}")
    print(f"  Accuracy:  {best_metrics['accuracy']:.4f}")

    # Save metrics
    results_file = project_root / "results" / f"lstm_{exp_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(best_metrics, f, indent=4)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
