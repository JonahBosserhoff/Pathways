import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_shap(
    shap_path="shap_results/shap_values.ftr",
    output_path="shap_summary.png",
    top_n=30,
):
    shap_data = pd.read_feather(shap_path)
    feature_importance = shap_data.abs().mean().sort_values(ascending=False)
    top_features = feature_importance.head(top_n).sort_values(ascending=True)

    norm = plt.Normalize(top_features.min(), top_features.max())
    plasma_custom = plt.cm.plasma
    truncated_plasma = mcolors.LinearSegmentedColormap.from_list(
        "truncated_plasma",
        plasma_custom(np.linspace(0.0, 0.9, 256)),
    )
    colors = truncated_plasma(norm(top_features.values))

    plt.figure(figsize=(5, 10))
    plt.hlines(
        y=top_features.index,
        xmin=0,
        xmax=top_features.values,
        color=colors,
        linewidth=2,
    )
    plt.scatter(
        top_features.values,
        top_features.index,
        color=colors,
        s=80,
        zorder=3,
    )
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top 30 Feature Importances")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    plot_shap()
