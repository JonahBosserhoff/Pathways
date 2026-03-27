import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

shap_data = pd.read_feather("/workspaces/Pathways/shap_results/shap_values.ftr")

feature_importance = shap_data.abs().mean().sort_values(ascending=False)
top_30 = feature_importance.head(30).sort_values(ascending=True)

norm = plt.Normalize(top_30.min(), top_30.max())

plasma_custom = plt.cm.plasma
truncated_plasma = mcolors.LinearSegmentedColormap.from_list(
    "truncated_plasma",
    plasma_custom(np.linspace(0.0, 0.9, 256))
)

colors = truncated_plasma(norm(top_30.values))

# Plot
plt.figure(figsize=(5, 10))

plt.hlines(
    y=top_30.index,
    xmin=0,
    xmax=top_30.values,
    color=colors,
    linewidth=2
)

plt.scatter(
    top_30.values,
    top_30.index,
    color=colors,
    s=80,
    zorder=3
)

plt.xlabel("Mean |SHAP value|")
plt.title("Top 30 Feature Importances")
plt.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("shap_summary.png")