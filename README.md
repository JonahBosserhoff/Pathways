# TerminalDecline Workflow

This repository contains a small end-to-end example pipeline for generating mock patient data, training a classification model, computing SHAP values, and generating a summary feature importance plot.

## Project Structure

- `main.py` - orchestrates the entire workflow.
- `create_mock_data.py` - generates a synthetic dataset and saves it as `mock_dataset.ftr`.
- `model.py` - trains a random forest model, saves `auc.png` and `model.pkl`.
- `calc_shap.py` - computes SHAP values and writes them into `shap_results/`.
- `plot_shap.py` - creates a SHAP feature importance summary plot saved as `shap_summary.png`.

## Recommended Usage

1. Install the required Python dependencies.
2. Run the full workflow with:

```bash
pip install -r requirements.txt
python3 main.py
```

This will execute the following steps:

1. `create_mock_data.py` -> generates `mock_dataset.ftr`
2. `model.py` -> trains the model and saves `model.pkl` and `auc.png`
3. `calc_shap.py` -> computes SHAP values and saves them under `shap_results/`
4. `plot_shap.py` -> saves `shap_summary.png`

## Files Produced

- `mock_dataset.ftr` - synthetic input dataset
- `model.pkl` - trained random forest model
- `auc.png` - ROC/AUC curve for the model
- `shap_results/shap_values.ftr` - SHAP values dataframe
- `shap_results/shap_indices.ftr` - selected test row indices
- `shap_results/shap_test_df.ftr` - selected test set rows
- `shap_summary.png` - SHAP feature importance plot

## Notes

- `main.py` defines default paths and parameters for the entire pipeline.
- Update configuration directly in `main.py` if you want to change file names, sample sizes, or other settings.
- If you prefer to run modules individually, each script can also be executed as a standalone script.
