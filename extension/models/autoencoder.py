"""
Autoencoder for Unsupervised Anomaly Detection

Deep learning unsupervised model that learns compressed representations
and detects anomalies based on reconstruction error.

Reference:
Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality
of data with neural networks. science, 313(5786), 504-507.
"""

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
    """
    Autoencoder neural network architecture
    """

    def __init__(self, input_dim, encoding_dim=32, hidden_dims=[64, 32]):
        """
        Initialize autoencoder architecture

        Args:
            input_dim: Number of input features
            encoding_dim: Dimension of encoded representation (bottleneck)
            hidden_dims: List of hidden layer dimensions
        """
        super(AutoencoderModel, self).__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        # Bottleneck
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers (mirror of encoder)
        decoder_layers = []
        prev_dim = encoding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Forward pass through autoencoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """Get encoded representation"""
        return self.encoder(x)


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector

    Trains on normal data, detects anomalies based on reconstruction error.
    High reconstruction error indicates anomaly.
    """

    def __init__(self, encoding_dim=32, hidden_dims=[64, 32],
                 epochs=50, batch_size=32, learning_rate=0.001,
                 threshold_percentile=95, random_state=42):
        """
        Initialize Autoencoder detector

        Args:
            encoding_dim: Bottleneck dimension
            hidden_dims: Hidden layer dimensions
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Adam optimizer learning rate
            threshold_percentile: Percentile for anomaly threshold
            random_state: Random seed
        """
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = None
        self.threshold = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_fitted = False

    def fit(self, X, y=None):
        """
        Train the autoencoder

        Args:
            X: Training data (n_samples, n_features)
            y: Not used (unsupervised), kept for API consistency

        Returns:
            self
        """
        print(f"Training Autoencoder...")
        print(f"  Device: {self.device}")
        print(f"  Input dimension: {X.shape[1]}")
        print(f"  Encoding dimension: {self.encoding_dim}")
        print(f"  Hidden dimensions: {self.hidden_dims}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Training samples: {X.shape[0]}")

        # Initialize model
        self.model = AutoencoderModel(
            input_dim=X.shape[1],
            encoding_dim=self.encoding_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)

        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)  # Input = target for autoencoder
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_target in dataloader:
                # Forward pass
                output = self.model(batch_X)
                loss = criterion(output, batch_target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")

        # Calculate reconstruction errors on training data
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            reconstruction_errors = reconstruction_errors.cpu().numpy()

        # Set threshold based on percentile of training errors
        self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
        print(f"  Anomaly threshold (percentile {self.threshold_percentile}): {self.threshold:.6f}")

        self.is_fitted = True
        print(f"✓ Model fitted successfully")

        return self

    def _get_reconstruction_error(self, X):
        """
        Calculate reconstruction error for samples

        Args:
            X: Data (n_samples, n_features)

        Returns:
            errors: Reconstruction error per sample
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            errors = errors.cpu().numpy()

        return errors

    def predict(self, X):
        """
        Predict anomalies based on reconstruction error

        Args:
            X: Data to predict (n_samples, n_features)

        Returns:
            y_pred: Binary predictions (0=normal, 1=anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        errors = self._get_reconstruction_error(X)
        y_pred = (errors > self.threshold).astype(int)

        return y_pred

    def decision_function(self, X):
        """
        Get anomaly scores (reconstruction error)

        Args:
            X: Data to score (n_samples, n_features)

        Returns:
            scores: Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        return self._get_reconstruction_error(X)

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
        print("AUTOENCODER - EVALUATION RESULTS")
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
    print("Autoencoder Model Implementation")
    print("Use scripts/01_train_unsupervised.py to train the model")
