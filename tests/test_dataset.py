from src.dataset import load_dataset


def test_iris_csv_loads_with_expected_dimensions():
    X, y, classes = load_dataset("iris_csv")
    assert X.shape == (150, 4)
    assert len(y) == len(X)
    assert set(classes) == {"setosa", "versicolor", "virginica"}
