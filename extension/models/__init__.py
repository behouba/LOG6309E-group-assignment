"""
Model package exposing detectors used across the extension scripts.

The modules are lightweight wrappers that provide a consistent API for
training and evaluating classical and deep learning detectors.
"""

from .isolation_forest import IsolationForestDetector  # noqa: F401
from .autoencoder import AutoencoderDetector  # noqa: F401
from .lstm import LSTMSequenceClassifier, LSTMTrainingConfig  # noqa: F401

__all__ = [
    "IsolationForestDetector",
    "AutoencoderDetector",
    "LSTMSequenceClassifier",
    "LSTMTrainingConfig",
]
