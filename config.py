"""Project configuration."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET = "iris_csv"  # Options: iris_csv, sklearn_iris, wine
CSV_PATH = PROJECT_ROOT / "data" / "iris.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"
MODELS_DIR = OUTPUTS_DIR / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
