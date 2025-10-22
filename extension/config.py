DATASET = "HDFS"
DATA_MODE = "full"
REPLICATION_DIR = "../replication"
DATA_DIR = f"{REPLICATION_DIR}/data"
RESULTS_DIR = "./results"
LOGS_DIR = "./logs"
MODELS_DIR = "./models"
PARSED_DATA_PATH = f"{DATA_DIR}/parsed/{DATASET}.log_structured.json"
SPLIT_DATA_PATH = f"{DATA_DIR}/split/{DATASET}_{DATA_MODE}_split.npz"

MCV_PATH = f"{DATA_DIR}/representations/{DATASET}_{DATA_MODE}_MCV.npz"
WORD2VEC_PATH = f"{DATA_DIR}/representations/{DATASET}_{DATA_MODE}_Word2Vec.npz"
FEATURE_SELECTED_MCV_PATH = f"{DATA_DIR}/representations/{DATASET}_{DATA_MODE}_MCV_selected.npz"

RANDOM_SEED = 42
ISOLATION_FOREST_CONFIG = {
    "n_estimators": 100,
    "max_samples": "auto",
    "contamination": 0.029,
    "random_state": 42
}

AUTOENCODER_CONFIG = {
    "encoding_dim": 32,
    "hidden_dims": [64, 32],
    "epochs": 30,
    "batch_size": 128,
    "learning_rate": 0.001,
    "threshold_percentile": 95
}
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

SCOTT_KNOTT_CONFIG = {
    "n_resamples": 10,
    "test_size": 0.3,
    "random_seed": 42,
    "alpha": 0.05
}

EXPLAINER_CONFIG = {
    "method": "shap",
    "n_samples": 100,
    "save_plots": True,
    "model_priority": ["Random Forest", "Isolation Forest", "Autoencoder"]
}
