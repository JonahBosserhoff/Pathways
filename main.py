from src.create_mock_data import create_mock_data
from src.model import train_model
from src.calc_shap import calculate_shap
from src.plot_shap import plot_shap


def main():
    config = {
        'mock_data_path': 'mock_dataset.ftr', 
        'model_path': 'model.pkl',
        'shap_output_dir': 'shap_results',
        'auc_output_path': 'auc.png',
        'shap_plot_path': 'shap_summary.png',
        'random_state': 42, # random state for scaler and model
        'test_size': 0.2, # test size for train/test split
        'n_estimators': 200, # number of trees in the random forest
        'model_n_jobs': 40, # number of parallel jobs for model training
        'n_shap_samples': 100, # number of samples to use for SHAP value calculation
        'shap_n_jobs': 1, # number of parallel jobs for SHAP value calculation
        'top_n_features': 10, # number of top features to show in SHAP summary plot
        "cutoff_for_classifier": 30 # cutoff in days for defining the positive class (death within this many days)
    }

    print('Starting mock data generation...')
    create_mock_data(
        output_path=config['mock_data_path'],
        seed=config['random_state'],
    )

    print('Training model...')
    train_model(
        data_path=config['mock_data_path'],
        model_output_path=config['model_path'],
        auc_output_path=config['auc_output_path'],
        random_state=config['random_state'],
        test_size=config['test_size'],
        n_estimators=config['n_estimators'],
        n_jobs=config['model_n_jobs'],
        cutoff_for_classifier=config["cutoff_for_classifier"]
    )

    print('Calculating SHAP values...')
    calculate_shap(
        model_path=config['model_path'],
        data_path=config['mock_data_path'],
        output_dir=config['shap_output_dir'],
        n_shap_samples=config['n_shap_samples'],
        n_jobs=config['shap_n_jobs'],
        random_state=config['random_state'],
        test_size=config['test_size'],
        drop_cols=['class', 'index', 'days_from_death'],
    )

    print('Plotting SHAP summary...')
    plot_shap(
        shap_path=f"{config['shap_output_dir']}/shap_values.ftr",
        output_path=config['shap_plot_path'],
        top_n=config['top_n_features'],
    )

    print('Workflow complete.')


if __name__ == '__main__':
    main()
