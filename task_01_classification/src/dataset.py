"""Dataset loading, validation, and summaries."""
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris, load_wine


def load_dataset(name: str = "iris_csv", csv_path: Path | str | None = None):
    """Return features, target, and class names for CSV, Iris, or Wine."""
    if name == "iris_csv":
        path = Path(csv_path) if csv_path else Path(__file__).resolve().parents[1] / "data" / "iris.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Iris CSV was not found: {path}")
        frame = pd.read_csv(path)
        target_col = "species"
        if target_col not in frame:
            raise ValueError(f"CSV must contain a '{target_col}' target column.")
        X = frame.drop(columns=target_col)
        y = frame[target_col].astype(str)
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
            raise ValueError("All feature columns in the Iris CSV must be numeric.")
        if X.isna().any().any() or y.isna().any():
            raise ValueError("Dataset contains missing values; please clean it before training.")
        classes = sorted(y.unique().tolist())
    elif name in {"sklearn_iris", "wine"}:
        bunch = load_iris(as_frame=True) if name == "sklearn_iris" else load_wine(as_frame=True)
        X, y = bunch.data.copy(), bunch.target.copy()
        y = y.map(dict(enumerate(bunch.target_names)))
        classes = list(bunch.target_names)
        target_col = "species" if name != "wine" else "wine_class"
    else:
        raise ValueError("Unknown dataset. Choose 'iris_csv', 'sklearn_iris', or 'wine'.")
    X = X.reset_index(drop=True)
    y = pd.Series(y, name=target_col).reset_index(drop=True)
    return X, y, classes


def save_dataset_summary(X, y, output_path: Path | str):
    """Save human- and machine-readable dataset metadata."""
    summary = {
        "rows": int(len(X)), "features": int(X.shape[1]),
        "feature_names": list(X.columns), "target_name": y.name,
        "classes": [str(v) for v in sorted(y.unique())],
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "dtypes": {str(k): str(v) for k, v in X.dtypes.items()},
        "missing_values": {str(k): int(v) for k, v in X.isna().sum().items()},
        "duplicate_feature_rows": int(X.duplicated().sum()),
        "summary_statistics": json.loads(X.describe().to_json()),
        "sample_records": json.loads(pd.concat([X, y], axis=1).head(5).to_json(orient="records")),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
