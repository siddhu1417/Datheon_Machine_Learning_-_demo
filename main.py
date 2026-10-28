"""Run the complete Iris classification benchmark."""
import json
import joblib
import pandas as pd

from config import DATASET, CSV_PATH, OUTPUTS_DIR, PLOTS_DIR, RESULTS_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.dataset import load_dataset, save_dataset_summary
from src.eda import generate_eda
from src.preprocessing import split_data
from src.train import train_models
from src.evaluate import evaluate_models
from src.visualize import generate_model_plots


def main():
    for directory in (OUTPUTS_DIR, PLOTS_DIR, RESULTS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    X, y, classes = load_dataset(DATASET, CSV_PATH)
    summary = save_dataset_summary(X, y, RESULTS_DIR / "dataset_summary.json")
    print(f"Dataset: {DATASET} | shape: {X.shape} | features: {list(X.columns)} | classes: {classes}")
    print(f"Missing values: {sum(summary['missing_values'].values())} | duplicate feature rows: {summary['duplicate_feature_rows']}")
    print(pd.DataFrame({"dtype": X.dtypes.astype(str)}).to_string())
    print(X.head().assign(**{y.name: y.head()}).to_string(index=False))
    generate_eda(X, y, PLOTS_DIR)
    X_train, X_test, y_train, y_test = split_data(X, y, TEST_SIZE, RANDOM_STATE)
    models, durations = train_models(X_train, y_train, RANDOM_STATE)
    metrics, predictions = evaluate_models(models, X_test, y_test, classes, durations, RESULTS_DIR)
    for name, model in models.items():
        joblib.dump(model, MODELS_DIR / (name.lower().replace(" ", "_") + ".joblib"))
    generate_model_plots(models, X_train, X_test, y_train, y_test, classes, predictions, metrics, PLOTS_DIR)
    split_info = {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "train_samples": len(X_train), "test_samples": len(X_test), "class_labels": classes}
    (RESULTS_DIR / "split_metadata.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    print("\nModel comparison (held-out stratified test split):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nPlots: {PLOTS_DIR}\nResults: {RESULTS_DIR}\nModels: {MODELS_DIR}")


if __name__ == "__main__":
    main()
