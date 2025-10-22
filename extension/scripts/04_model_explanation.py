import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATASET,
    DATA_MODE,
    EXPLAINER_CONFIG,
    ISOLATION_FOREST_CONFIG,
    MCV_PATH,
    MODELS_DIR,
    RESULTS_DIR,
)
from models.isolation_forest import IsolationForestDetector

try:
    import shap
except ImportError:
    print("Error: SHAP is not installed. Please run: pip install shap")
    sys.exit(1)


def get_best_model_name():
    ranking_file = Path(RESULTS_DIR) / "scott_knott_ranking.csv"
    if not ranking_file.exists():
        print("Warning: Scott-Knott ranking file not found; using preferred explanation model.")
        return EXPLAINER_CONFIG.get("model_priority", ["Random Forest"])[0]

    ranking_df = pd.read_csv(ranking_file)
    ranking_df = ranking_df.sort_values(by=["F1_Rank", "AUC_Rank"])

    priority = EXPLAINER_CONFIG.get("model_priority", [])
    for model in ranking_df["Model"]:
        if model in priority:
            return model

    return ranking_df.iloc[0]["Model"]


def main():
    print("SHAP explanation for top-ranked model")
    best_model_name = get_best_model_name()
    print(f"Best performing model from ranking: {best_model_name}")

    supported_models = {"Random Forest", "Isolation Forest"}
    if best_model_name not in supported_models:
        print(
            f"Currently supported explanation targets: {', '.join(sorted(supported_models))}."
        )
        print(
            f"'{best_model_name}' is not directly explainable with the selected technique."
        )
        return

    print("Loading MCV data for explanation")
    mcv_data = np.load(MCV_PATH, allow_pickle=True)
    x_train = mcv_data["x_train"]
    y_train = mcv_data["y_train"]
    x_test = mcv_data["x_test"]

    feature_names = [f"Template_{i}" for i in range(x_train.shape[1])]
    print(f"Retraining {best_model_name} on the full training data")
    if "Random Forest" in best_model_name:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(x_train, y_train)
        predict_fn = lambda data: model.predict_proba(data)[:, 1]
    else:
        contamination = np.mean(y_train)
        iforest_config = ISOLATION_FOREST_CONFIG.copy()
        iforest_config["contamination"] = contamination
        model = IsolationForestDetector(**iforest_config)
        model.fit(x_train)
        predict_fn = model.decision_function

    MODELS_DIR_PATH = Path(MODELS_DIR)
    MODELS_DIR_PATH.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR_PATH / f"{best_model_name.replace(' ', '_')}.joblib")
    print(f"Model saved to {MODELS_DIR_PATH}")
    print("Initializing SHAP explainer")
    n_background = min(100, x_train.shape[0])
    method = EXPLAINER_CONFIG.get("method", "shap")

    if "Random Forest" in best_model_name and method == "shap":
        explainer = shap.TreeExplainer(model)
        shap_explainer_type = "tree"
    else:
        background_data = shap.kmeans(x_train, n_background)
        explainer = shap.KernelExplainer(predict_fn, background_data)
        shap_explainer_type = "kernel"
    print(f"SHAP explainer type: {shap_explainer_type}")
    print("Computing SHAP values for a test subset")
    sample_size = min(EXPLAINER_CONFIG.get("n_samples", 100), x_test.shape[0])
    test_sample = x_test[:sample_size]
    if shap_explainer_type == "tree":
        raw_exp = explainer(test_sample)
        positive_exp = shap.Explanation(
            values=raw_exp.values[..., 1],
            base_values=raw_exp.base_values[..., 1],
            data=raw_exp.data,
            feature_names=feature_names,
        )
    else:
        shap_values = explainer.shap_values(test_sample)
        if isinstance(shap_values, list):
            shap_matrix = shap_values[0]
        else:
            shap_matrix = shap_values
        base_vals = np.broadcast_to(
            np.array(explainer.expected_value, dtype=np.float32),
            (shap_matrix.shape[0],),
        )
        positive_exp = shap.Explanation(
            values=shap_matrix,
            base_values=base_vals,
            data=test_sample,
            feature_names=feature_names,
        )
    shap_matrix = positive_exp.values
    expected_values = positive_exp.base_values
    results_dir = Path(RESULTS_DIR)
    
    print("Generating SHAP summary plot")
    plt.figure()
    shap.summary_plot(
        shap_matrix, test_sample, feature_names=feature_names, show=False
    )
    plt.title(f"SHAP Feature Importance for {best_model_name}", fontsize=14)
    plt.tight_layout()
    summary_plot_path = results_dir / "shap_summary.png"
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved to {summary_plot_path}")

    print("Generating SHAP waterfall plot for a single prediction")
    sample_idx = 0
    shap.plots.waterfall(positive_exp[sample_idx], max_display=15, show=False)
    waterfall_path = results_dir / "shap_waterfall_single.png"
    plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Waterfall plot saved to {waterfall_path}")

    print("Top SHAP contributions")
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False)

    print(importance_df.head(5).to_string(index=False))
    print("These templates drive the anomaly predictions.")

if __name__ == "__main__":
    main()
