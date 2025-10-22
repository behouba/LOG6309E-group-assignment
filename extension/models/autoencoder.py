import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    accuracy_score
)


class AutoencoderModel(nn.Module):
    def __init__(self, input_dim, encoding_dim=32, hidden_dims=[64, 32]):
        super(AutoencoderModel, self).__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = encoding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class AutoencoderDetector:
    def __init__(self, encoding_dim=32, hidden_dims=[64, 32],
                 epochs=50, batch_size=32, learning_rate=0.001,
                 threshold_percentile=95, random_state=42):
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = None
        self.threshold = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_fitted = False

    def fit(self, X, y=None):
        print("Training Autoencoder")
        print(f"  Samples: {X.shape[0]}, Input dim: {X.shape[1]}, Device: {self.device}")
        self.model = AutoencoderModel(
            input_dim=X.shape[1],
            encoding_dim=self.encoding_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)

        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_target in dataloader:
                output = self.model(batch_X)
                loss = criterion(output, batch_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                avg_loss = epoch_loss / len(dataloader)
                print(f"  Epoch {epoch+1}/{self.epochs} loss: {avg_loss:.6f}")
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            reconstruction_errors = reconstruction_errors.cpu().numpy()
        self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
        print(f"  Threshold (p{self.threshold_percentile}): {self.threshold:.6f}")
        self.is_fitted = True
        print("Model fitted successfully.")

        return self

    def _get_reconstruction_error(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            errors = errors.cpu().numpy()

        return errors

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        errors = self._get_reconstruction_error(X)
        y_pred = (errors > self.threshold).astype(int)

        return y_pred

    def decision_function(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        return self._get_reconstruction_error(X)

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

        print("Autoencoder evaluation:")
        print(f"  Precision {metrics['precision']:.4f}, Recall {metrics['recall']:.4f}, F1 {metrics['f1_score']:.4f}, AUC {metrics['auc']:.4f}, Accuracy {metrics['accuracy']:.4f}")
        cm = metrics['confusion_matrix']
        print(f"  Confusion matrix: TN {cm['TN']}, FP {cm['FP']}, FN {cm['FN']}, TP {cm['TP']}")

        return metrics


if __name__ == "__main__":
    print("Autoencoder Model Implementation")
    print("Use scripts/01_train_unsupervised.py to train the model")
