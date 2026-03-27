import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import (
    roc_curve,
    auc,
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
df_imputed = pd.read_feather("mock_dataset.ftr")
RANDOM_STATE = 42
MODEL_OUTPUT_PATH = "model_30_days.pkl"
 
# ─────────────────────────────────────────────
# Labeling
# ─────────────────────────────────────────────
bins = [-np.inf, -30, 0]
df_imputed["class"] = np.digitize(df_imputed["days_from_death"], bins=bins, right=True)
df_imputed.fillna(0, inplace=True)
 
# ─────────────────────────────────────────────
# Patient-level train/test split
# ─────────────────────────────────────────────
unique_patients = df_imputed["index"].unique()
train_patients, test_patients = train_test_split(
    unique_patients, test_size=0.2, random_state=RANDOM_STATE
)
 
train_df = df_imputed[df_imputed["index"].isin(train_patients)]
test_df  = df_imputed[df_imputed["index"].isin(test_patients)]
 
drop_cols = ["class", "index", "days_from_death"]
X_train, y_train = train_df.drop(columns=drop_cols), train_df["class"]
X_test,  y_test  = test_df.drop(columns=drop_cols),  test_df["class"]
 
# ─────────────────────────────────────────────
# Scaling
# ─────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
 
# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=200, n_jobs=40, random_state=RANDOM_STATE)
model.fit(X_train, y_train)
 
# ─────────────────────────────────────────────
# AUC Curve
# ─────────────────────────────────────────────
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=2)
roc_auc = auc(fpr, tpr)
 
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, lw=2, label=f"Random Forest  (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random Classifier")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.05)
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC / AUC Curve – Random Forest", fontsize=15)
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
plt.savefig("auc.png")
 
# ─────────────────────────────────────────────
# Save model
# ─────────────────────────────────────────────
joblib.dump(model, MODEL_OUTPUT_PATH)
print(f"Model saved to {MODEL_OUTPUT_PATH}", flush=True)