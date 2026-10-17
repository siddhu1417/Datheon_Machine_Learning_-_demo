"""Model evaluation visualizations."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier


def generate_model_plots(models, X_train, X_test, y_train, y_test, classes, predictions, metrics, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    for name, pred in predictions.items():
        cm_path_name = name.lower().replace(" ", "_")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, pred, labels=classes)
        plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title(f"{name} Confusion Matrix"); plt.xlabel("Predicted class"); plt.ylabel("Actual class")
        plt.tight_layout(); plt.savefig(plots_dir / f"{cm_path_name}_confusion_matrix.png", dpi=160); plt.close()
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    ax = metrics.set_index("model")[metric_names].plot(kind="bar", figsize=(9, 5), ylim=(0, 1.08), color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_title("Classifier Metrics on the Held-Out Test Split"); ax.set_ylabel("Score"); ax.set_xlabel(""); ax.legend(["Accuracy", "Macro precision", "Macro recall", "Macro F1"], loc="lower right")
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(plots_dir / "model_comparison.png", dpi=160); plt.close()
    tree = models["Decision Tree"]
    plt.figure(figsize=(16, 9)); plot_tree(tree, feature_names=list(X_train.columns), class_names=[str(c) for c in classes], filled=True, rounded=True, max_depth=4, fontsize=8)
    plt.title("Decision Tree (display limited to depth 4)"); plt.tight_layout(); plt.savefig(plots_dir / "decision_tree.png", dpi=160); plt.close()
    # A separate two-feature demonstration; the benchmark models above still use all features.
    features = list(X_train.columns[:2])
    class_to_number = {label: number for number, label in enumerate(classes)}
    y_train_numbered = y_train.map(class_to_number)
    demo = DecisionTreeClassifier(random_state=42, max_depth=3).fit(X_train[features], y_train_numbered)
    fig, ax = plt.subplots(figsize=(8, 6))
    DecisionBoundaryDisplay.from_estimator(demo, X_train[features], response_method="predict", alpha=0.22, ax=ax, xlabel=features[0], ylabel=features[1])
    palette = sns.color_palette("Set1", n_colors=len(classes))
    for idx, label in enumerate(classes):
        train_mask = y_train == label; test_mask = y_test == label
        ax.scatter(X_train.loc[train_mask, features[0]], X_train.loc[train_mask, features[1]], color=palette[idx], marker="o", s=35, label=f"Train: {label}")
        ax.scatter(X_test.loc[test_mask, features[0]], X_test.loc[test_mask, features[1]], color=palette[idx], marker="^", s=55, edgecolor="black", label=f"Test: {label}")
    ax.set_title("Two-Feature Decision Boundary Demonstration\n(Simplified; benchmark models use all features)")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(plots_dir / "decision_boundary_two_features.png", dpi=160); plt.close(fig)
