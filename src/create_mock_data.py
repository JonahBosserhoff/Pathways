import numpy as np
import pandas as pd


def create_mock_data(
    output_path="mock_dataset.ftr",
    n_rows=1000,
    n_patients=150,
    seed=42,
):
    np.random.seed(seed)

    patient_ids = np.random.choice(range(1, n_patients + 1), size=n_rows, replace=True)
    patient_ids.sort()

    rows = []
    for pid in patient_ids:
        max_days = np.random.randint(-730, -30)
        day = np.random.randint(max_days, 1)
        rows.append({"index": pid, "days_from_death": day})

    df = pd.DataFrame(rows)

    patient_group = {pid: np.random.choice([0, 1, 2], p=[0.4, 0.35, 0.25])
                     for pid in range(1, n_patients + 1)}
    df["_group"] = df["index"].map(patient_group)  

    group_means = np.array([
        [20, 80, 50, 30, 60, 40, 70, 55, 45, 65],   
        [70, 20, 80, 60, 30, 55, 40, 75, 35, 50],   
        [45, 55, 25, 75, 45, 65, 30, 40, 70, 35],   
    ])

    for i in range(1, 11):
        col = f"feature_{i}"
        std = np.random.uniform(8, 15)
        base = df["_group"].map(lambda g, i=i: group_means[g, i - 1])
        noise = np.random.normal(0, std, n_rows)
        df[col] = np.round(base + noise, 2)

    df["feature_1"] += np.round(-0.03 * df["days_from_death"], 2)
    df["feature_3"] += np.round(0.02 * df["days_from_death"], 2)

    df["feature_5"] = np.round(
        df["feature_5"] + 0.3 * df["feature_2"] - 0.2 * df["feature_4"] +
        np.random.normal(0, 5, n_rows),
        2,
    )

    code_cols = [f"code_{i}" for i in range(1, 6)]
    one_hot = np.zeros((n_rows, 5), dtype=int)
    group_probs = {
        0: [0.5, 0.2, 0.1, 0.1, 0.1],
        1: [0.1, 0.1, 0.5, 0.2, 0.1],
        2: [0.1, 0.1, 0.1, 0.2, 0.5],
    }
    for idx, g in enumerate(df["_group"]):
        active = np.random.choice(5, p=group_probs[g])
        one_hot[idx, active] = 1
    df[code_cols] = one_hot

    df.drop(columns=["_group"], inplace=True)
    df.sort_index(inplace=True)
    df.to_feather(output_path)
    return df


if __name__ == "__main__":
    df = create_mock_data()
    print(df.head(10).to_string())
    print(f"\nShape: {df.shape}")