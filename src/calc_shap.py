import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def calculate_shap(
    model_path="model_30_days.pkl",
    data_path="mock_dataset.ftr",
    output_dir="shap_results",
    n_shap_samples=100,
    n_jobs=1,
    random_state=42,
    test_size=0.2,
    drop_cols=None,
):
    if drop_cols is None:
        drop_cols = ["class", "index", "days_from_death"]

    # ─────────────────────────────────────────────
    # Load
    # ─────────────────────────────────────────────
    model = joblib.load(model_path)
    df_imputed = pd.read_feather(data_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────
    # Labeling
    # ─────────────────────────────────────────────
    bins = [-np.inf, -30, 0]
    df_imputed["class"] = np.digitize(df_imputed["days_from_death"], bins=bins, right=True)
    df_imputed.fillna(0, inplace=True)

    # ─────────────────────────────────────────────
    # Split
    # ─────────────────────────────────────────────
    unique_patients = df_imputed["index"].unique()
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=test_size, random_state=random_state
    )

    train_df = df_imputed[df_imputed["index"].isin(train_patients)]
    test_df = df_imputed[df_imputed["index"].isin(test_patients)].reset_index(drop=True)

    feature_cols = [c for c in df_imputed.columns if c not in drop_cols]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols].reset_index(drop=True)

    # ─────────────────────────────────────────────
    # Scaling
    # ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ─────────────────────────────────────────────
    # Sampling
    # ─────────────────────────────────────────────
    n_samples = min(n_shap_samples, len(X_test))
    if n_samples < n_shap_samples:
        print(
            f"Warning: requested {n_shap_samples} samples but test set only has {len(X_test)} rows. Using {n_samples}.",
            flush=True,
        )

    rng = np.random.default_rng(random_state)
    random_indices = rng.choice(len(X_test), size=n_samples, replace=False)

    # ─────────────────────────────────────────────
    # SHAP
    # ─────────────────────────────────────────────
    explainer = shap.TreeExplainer(model)

    def compute_shap(i):
        return explainer.shap_values(X_test[i : i + 1], check_additivity=False)

    shap_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_shap)(i) for i in tqdm(random_indices)
    )

    shap_array = np.concatenate(shap_list, axis=1)

    shap_class1 = shap_array[0, :, 1].reshape(n_samples, -1)

    assert shap_class1.shape == (n_samples, len(feature_cols)), (
        f"Shape mismatch: got {shap_class1.shape}, "
        f"expected ({n_samples}, {len(feature_cols)})"
    )

    # ─────────────────────────────────────────────
    # Output
    # ─────────────────────────────────────────────
    df_shap = pd.DataFrame(shap_class1, columns=feature_cols)
    df_indices = pd.DataFrame({"row_in_test_df": random_indices})

    test_df.to_feather(f"{output_dir}/shap_test_df.ftr")
    df_indices.to_feather(f"{output_dir}/shap_indices.ftr")
    df_shap.to_feather(f"{output_dir}/shap_values.ftr")

    print(f"Done. Outputs written to: {output_dir}", flush=True)

    return df_shap


if __name__ == "__main__":
    calculate_shap()