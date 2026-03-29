import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import roc_curve, auc


def train_model(
    data_path="mock_dataset.ftr",
    model_output_path="model.pkl",
    auc_output_path="auc.png",
    random_state=42,
    test_size=0.2,
    n_estimators=200,
    n_jobs=40,
    cutoff_for_classifier=30
):
    df_imputed = pd.read_feather(data_path)

    bins = [-np.inf, -cutoff_for_classifier, 0]
    df_imputed["class"] = np.digitize(df_imputed["days_from_death"], bins=bins, right=True)
    df_imputed.fillna(0, inplace=True)

    unique_patients = df_imputed["index"].unique()
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=test_size, random_state=random_state
    )

    train_df = df_imputed[df_imputed["index"].isin(train_patients)]
    test_df = df_imputed[df_imputed["index"].isin(test_patients)]

    drop_cols = ["class", "index", "days_from_death"]
    X_train, y_train = train_df.drop(columns=drop_cols), train_df["class"]
    X_test, y_test = test_df.drop(columns=drop_cols), test_df["class"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

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
    fig.savefig(auc_output_path)
    plt.close(fig)

    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}", flush=True)
    return model


if __name__ == "__main__":
    train_model()
