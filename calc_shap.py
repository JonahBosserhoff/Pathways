import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_PATH    = "/workspaces/Pathways/model_30_days.pkl"
DATA_PATH     = "/workspaces/Pathways/mock_dataset.ftr"
OUTPUT_DIR    = "/workspaces/Pathways/shap_results"

N_SHAP_SAMPLES = 100
N_JOBS         = 1
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
DROP_COLS      = ["class", "index", "days_from_death"]

# ─────────────────────────────────────────────
# Load model & data
# ─────────────────────────────────────────────
model      = joblib.load(MODEL_PATH)
df_imputed = pd.read_feather(DATA_PATH)
Path.mkdir(OUTPUT_DIR, exist_ok=True)

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
    unique_patients, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

train_df = df_imputed[df_imputed["index"].isin(train_patients)]
test_df  = df_imputed[df_imputed["index"].isin(test_patients)].reset_index(drop=True)

feature_cols = [c for c in df_imputed.columns if c not in DROP_COLS]

X_train = train_df[feature_cols]
X_test  = test_df[feature_cols].reset_index(drop=True)

y_train = train_df["class"]
y_test  = test_df["class"]

# ─────────────────────────────────────────────
# Scaling
# ─────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# SHAP sampling
# ─────────────────────────────────────────────
n_samples = min(N_SHAP_SAMPLES, len(X_test))
if n_samples < N_SHAP_SAMPLES:
    print(f"Warning: requested {N_SHAP_SAMPLES} samples but test set only has {len(X_test)} rows. Using {n_samples}.", flush=True)

rng            = np.random.default_rng(RANDOM_STATE)
random_indices = rng.choice(len(X_test), size=n_samples, replace=False)

# ─────────────────────────────────────────────
# SHAP computation
# ─────────────────────────────────────────────
explainer = shap.TreeExplainer(model)

def compute_shap(i):
    return explainer.shap_values(X_test[i : i + 1], check_additivity=False)

shap_list  = Parallel(n_jobs=N_JOBS)(
    delayed(compute_shap)(i) for i in tqdm(random_indices)
)

# shap_list: list of arrays with shape (n_classes, 1, n_features)
shap_array = np.concatenate(shap_list, axis=1)   # → (n_classes, n_samples, n_features)
shap_class1 = shap_array[0, :, 1].reshape(N_SHAP_SAMPLES, -1)                # class 1 → (n_samples, n_features)

assert shap_class1.shape == (n_samples, len(feature_cols)), (
    f"Shape mismatch: got {shap_class1.shape}, "
    f"expected ({n_samples}, {len(feature_cols)})"
)

# ─────────────────────────────────────────────
# Build output dataframes
# ─────────────────────────────────────────────
df_shap    = pd.DataFrame(shap_class1, columns=feature_cols)
df_indices = pd.DataFrame({"row_in_test_df": random_indices})

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
test_df.to_feather(   f"{OUTPUT_DIR}/shap_test_df.ftr")
df_indices.to_feather(f"{OUTPUT_DIR}/shap_indices.ftr")
df_shap.to_feather(   f"{OUTPUT_DIR}/shap_values.ftr")

print("Done. Outputs written to:", OUTPUT_DIR, flush=True)