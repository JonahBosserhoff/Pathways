import numpy as np
import pandas as pd
 
np.random.seed(42)
 
N_ROWS = 1000
N_PATIENTS = 150 
 
patient_ids = np.random.choice(range(1, N_PATIENTS + 1), size=N_ROWS, replace=True)
patient_ids.sort() 
 
rows = []
for pid in patient_ids:
    max_days = np.random.randint(-730, -30)
    day = np.random.randint(max_days, 1)
    rows.append({"index": pid, "days_from_death": day})
 
df = pd.DataFrame(rows)
 
for i in range(1, 11):
    mean = np.random.uniform(0, 100)
    std = np.random.uniform(1, 20)
    df[f"feature_{i}"] = np.round(np.random.normal(mean, std, N_ROWS), 2)
 
code_cols = [f"code_{i}" for i in range(1, 6)]
one_hot = np.zeros((N_ROWS, 5), dtype=int)
active = np.random.randint(0, 5, size=N_ROWS)
one_hot[np.arange(N_ROWS), active] = 1
df[code_cols] = one_hot
 
df.sort_index(inplace=True)

 
 
df.to_feather("mock_dataset.ftr")