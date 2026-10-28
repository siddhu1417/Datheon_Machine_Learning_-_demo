"""Exploratory analysis plots and dataset summary."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def generate_eda(X, y, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    frame = X.copy()
    frame[y.name] = y.astype(str).values
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 5))
    sns.countplot(data=frame, x=y.name, hue=y.name, legend=False, palette="Set2")
    plt.title("Iris Species Distribution"); plt.xlabel("Species"); plt.ylabel("Sample count")
    plt.tight_layout(); plt.savefig(plots_dir / "class_distribution.png", dpi=160); plt.close()
    grid = sns.pairplot(frame, hue=y.name, diag_kind="hist", corner=True, plot_kws={"alpha": 0.7, "s": 28})
    grid.fig.suptitle("Iris Feature Pairplot", y=1.02)
    grid.savefig(plots_dir / "pairplot.png", dpi=140); plt.close(grid.fig)
    plt.figure(figsize=(8, 6))
    sns.heatmap(X.corr(numeric_only=True), annot=True, cmap="vlag", center=0, fmt=".2f")
    plt.title("Feature Correlation Heatmap"); plt.tight_layout()
    plt.savefig(plots_dir / "correlation_heatmap.png", dpi=160); plt.close()
    axes = X.hist(figsize=(10, 8), bins=16, color="#4C78A8", edgecolor="white")
    plt.suptitle("Feature Distributions"); plt.tight_layout()
    plt.savefig(plots_dir / "feature_distributions.png", dpi=160); plt.close()
