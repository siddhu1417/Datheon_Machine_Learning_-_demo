# Iris Classification Benchmark

This project compares two supervised learning models—Logistic Regression and Decision Tree—for predicting Iris flower species from four measurements: sepal length, sepal width, petal length, and petal width. It uses the supplied `data/iris.csv` dataset.

## Run

Requires Python 3.11 or 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
python -m pytest -v
```

## Workflow

The program summarizes the data, creates exploratory plots, and uses a stratified 80:20 train/test split. Logistic Regression uses a StandardScaler fitted on training data only; Decision Tree uses unscaled features. Both models are evaluated on the same test samples.

## Outputs

Running `main.py` creates plots, model comparison metrics, per-class reports, confusion matrices, and saved models under `outputs/plots/`, `outputs/results/`, and `outputs/models/`. Results depend on the held-out test split. The two-feature decision-boundary plot is an illustrative demo; the benchmark models use all four features.

Change `DATASET` in `config.py` to `sklearn_iris` or `wine` to use a built-in Scikit-learn dataset instead.
