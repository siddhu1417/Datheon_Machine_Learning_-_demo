from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
import numpy as np
from src.dataset import load_dataset
from src.preprocessing import split_data
from src.train import train_models
from src.evaluate import evaluate_models


def test_models_train_predict_and_evaluate(tmp_path):
    X, y, classes = load_dataset("iris_csv")
    X_train, X_test, y_train, y_test = split_data(X, y)
    models, times = train_models(X_train, y_train)
    assert isinstance(models["Logistic Regression"], Pipeline)
    scaler = models["Logistic Regression"].named_steps["scaler"]
    assert np.allclose(scaler.mean_, X_train.mean(axis=0).to_numpy())
    for model in models.values():
        assert len(model.predict(X_test)) == len(y_test)
    results, _ = evaluate_models(models, X_test, y_test, classes, times, tmp_path)
    assert results.shape[0] == 2
    assert results[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].apply(lambda c: c.between(0, 1).all()).all()
    assert confusion_matrix(y_test, models["Decision Tree"].predict(X_test), labels=classes).shape == (3, 3)
    assert (tmp_path / "model_comparison.csv").exists()
