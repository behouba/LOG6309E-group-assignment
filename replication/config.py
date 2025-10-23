# Dataset selection
# Options: "HDFS", "BGL", "Spirit", "Thunderbird"
DATASET = "BGL"

# Data mode
# Options: "sample" (100k lines for quick testing), "full" (complete dataset)
DATA_MODE = "full"

# Dataset-specific configurations
DATASET_FILES = {
    "HDFS": {
        "sample": "HDFS_100k.log_structured.json",
        "full": "HDFS.log_structured.json",
        "anomaly_labels": "anomaly_label.csv"  # For HDFS only
    },
    "BGL": {
        "sample": "BGL_100k.log_structured.json",  # Would need to create this
        "full": "BGL.log_structured.json",
        "anomaly_labels": None  # BGL labels are in the log file itself
    },
    "Spirit": {
        "sample": None,
        "full": "Spirit.log_structured.json",
        "anomaly_labels": None
    },
    "Thunderbird": {
        "sample": None,
        "full": "Thunderbird.log_structured.json",
        "anomaly_labels": None
    }
}

# Model hyperparameters
RANDOM_FOREST = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": 42
}

LSTM = {
    "hidden_dim": 8,
    "num_layers": 1,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "batch_size": 64, # Adjust based on GPU memory
    "window_size": 50,
    "stride": 50,
    "random_seed": 42
}

# Word2Vec configuration
WORD2VEC = {
    # Mode options: "pretrained" (load existing embeddings) or "train" (train on templates)
    "mode": "pretrained",
    "pretrained_model": "word2vec-google-news-300",
    # Optional local path to a pre-downloaded Word2Vec binary (e.g., GoogleNews-vectors-negative300.bin.gz)
    "local_path": "data/representations/GoogleNews-vectors-negative300.bin.gz",
    # Parameters used when mode == "train"
    "train_params": {
        "vector_size": 100,
        "window": 5,
        "min_count": 1,
        "workers": 4,
        "epochs": 10
    }
}

# Feature selection thresholds
FEATURE_SELECTION = {
    "correlation_threshold": 0.95,  # For hierarchical clustering
    "vif_threshold": 5.0  # For VIF analysis
}

# Data splitting
TRAIN_TEST_SPLIT = {
    "train_ratio": 0.7,
    "random_state": 42
}

# GPU Configuration
GPU_CONFIG = {
    "use_gpu": True,  
    "device": "cuda",
    "mixed_precision": True,
    "num_workers": 4,
    "pin_memory": True,
    "batch_size_multiplier": 2,
}

# Helper functions
def get_parsed_file():
    return DATASET_FILES[DATASET][DATA_MODE]

def get_anomaly_labels_file():
    return DATASET_FILES[DATASET]["anomaly_labels"]

def get_experiment_name():
    return f"{DATASET}_{DATA_MODE}"
