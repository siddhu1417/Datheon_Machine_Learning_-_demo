"""Metrics and report serialization."""
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score


def evaluate_models(models, X_test, y_test, classes, training_times, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    rows, predictions = [], {}
    for name, model in models.items():
        pred = model.predict(X_test); predictions[name] = pred
        report = classification_report(y_test, pred, labels=classes, output_dict=True, zero_division=0)
        scores = {
            "accuracy": accuracy_score(y_test, pred),
            "precision_macro": precision_score(y_test, pred, labels=classes, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, pred, labels=classes, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, pred, labels=classes, average="macro", zero_division=0),
            "training_time_seconds": training_times[name],
        }
        rows.append({"model": name, **scores})
        (results_dir / (name.lower().replace(" ", "_") + "_classification_report.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
        cm = confusion_matrix(y_test, pred, labels=classes)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(results_dir / (name.lower().replace(" ", "_") + "_confusion_matrix.csv"))
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(results_dir / "model_comparison.csv", index=False)
    return result_frame, predictions
