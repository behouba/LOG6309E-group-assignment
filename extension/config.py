"""
Configuration for Part 2 Extension

Part 2 Requirements:
1. Two unsupervised/semi-supervised models from different categories
2. Scott-Knott statistical ranking (4 models total: 2 from Part 1 + 2 new)
3. Model explanation (SHAP/LIME) on best model
4. Use ONE dataset (HDFS chosen for consistency)
"""

# Dataset selection (use one dataset from Part 1)
DATASET = "HDFS"  # Options: HDFS or BGL
DATA_MODE = "full"  # Options: sample or full

# Paths (reuse Part 1 data where possible)
REPLICATION_DIR = "../replication"
DATA_DIR = f"{REPLICATION_DIR}/data"
RESULTS_DIR = "./results"
LOGS_DIR = "./logs"
MODELS_DIR = "./models"

# Reuse parsed data from Part 1
PARSED_DATA_PATH = f"{DATA_DIR}/parsed/{DATASET}.log_structured.json"
SPLIT_DATA_PATH = f"{DATA_DIR}/split/{DATASET}_{DATA_MODE}_split.npz"

# Representation paths from Part 1
MCV_PATH = f"{DATA_DIR}/representations/{DATASET}_{DATA_MODE}_MCV.npz"
# Word2Vec exports in Part 1 use capitalized naming convention
WORD2VEC_PATH = f"{DATA_DIR}/representations/{DATASET}_{DATA_MODE}_Word2Vec.npz"

# Feature selection paths from Part 1
FEATURE_SELECTED_MCV_PATH = f"{DATA_DIR}/representations/{DATASET}_{DATA_MODE}_MCV_selected.npz"

# Part 2 Models Configuration

# Random seed for reproducibility
RANDOM_SEED = 42

# Model 1: Isolation Forest (Classical Unsupervised)
ISOLATION_FOREST_CONFIG = {
    "n_estimators": 100,
    "max_samples": "auto",
    "contamination": 0.029,  # Based on HDFS anomaly rate
    "random_state": 42
}

# Model 2: Autoencoder (Deep Learning Unsupervised)
AUTOENCODER_CONFIG = {
    "encoding_dim": 32,
    "hidden_dims": [64, 32],
    "epochs": 30,
    "batch_size": 128,
    "learning_rate": 0.001,
    "threshold_percentile": 95  # Anomaly threshold
}

# Supervised Deep Model (LSTM) configuration for resampling
LSTM_CONFIG = {
    "window_size": 32,
    "stride": 32,
    "hidden_dim": 32,
    "num_layers": 1,
    "num_epochs": 4,
    "batch_size": 256,
    "learning_rate": 0.001,
    "val_fraction": 0.1,
    "patience": 2,
    "threshold": 0.5,
    "random_state": RANDOM_SEED
}

# Scott-Knott Configuration
SCOTT_KNOTT_CONFIG = {
    "n_resamples": 10,  # Number of train/test resamplings
    "test_size": 0.3,
    "random_seed": 42,
    "alpha": 0.05  # Significance level
}

# SHAP/LIME Configuration
EXPLAINER_CONFIG = {
    "method": "shap",  # Options: shap, lime
    "n_samples": 100,  # Number of samples to explain
    "save_plots": True,
    # Prefer interpretable models when multiple have top rank
    "model_priority": ["Random Forest", "Isolation Forest", "Autoencoder"]
}
