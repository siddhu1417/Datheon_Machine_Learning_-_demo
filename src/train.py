"""Model construction and training."""
from time import perf_counter
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def train_models(X_train, y_train, random_state=42):
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
    }
    durations = {}
    for name, model in models.items():
        start = perf_counter(); model.fit(X_train, y_train)
        durations[name] = perf_counter() - start
    return models, durations
