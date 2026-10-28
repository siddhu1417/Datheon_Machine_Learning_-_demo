from collections import Counter
from src.dataset import load_dataset
from src.preprocessing import split_data


def test_split_is_80_20_and_stratified():
    X, y, _ = load_dataset("iris_csv")
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_test) == 30 and len(X_train) == 120
    assert Counter(y_test)["setosa"] == 10
    assert Counter(y_test)["versicolor"] == 10
    assert Counter(y_test)["virginica"] == 10
