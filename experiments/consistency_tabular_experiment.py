"""
Main experimentation pipeline for measuring consistency on tabular datasets.

Things we would like to measure against consistency of explanations:
* Number of runs
* Number of features
"""
import argparse
import logging
import os
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.utils.datasets import get_dataset
from experiments.experiments_common import main, measure_consistency
from experiments.utils.explainers import get_explainer
from experiments.utils.models import get_model


def run_test(config: dict) -> List[float]:
    """
    Run consistency experiment with a given configuration

    Args:
        config (dict): configuration

    Returns:
        (list) Consistency proportion
    """
    # Instantiate model
    model_name = config['model']['name']
    model_params = config['model']['params']
    model = get_model(model_name, model_params)

    # Load dataset
    dataset_name = config['dataset']['name']
    dataset_params = config['dataset'].get('params', {})
    dataset = get_dataset(dataset_name, dataset_params)
    X = dataset['data']

    if type(X) is pd.DataFrame:
        X = X.values

    y = dataset['target']

    if type(y) is pd.DataFrame:
        y = y.values

    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(
        config['explanations']['n_explanations']))
    model.fit(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    logger.info('    Test accuracy: {}'.format(test_acc))

    # Instantiate explainer
    explainer_params = config['explanations']['model_params']
    explainer_params['training_data'] = X_train
    explainer_params['feature_names'] = dataset.get('feature_names', None)

    num_exp_per_sample = config['explanations']['num_exp_per_sample']
    logger.info('    Testing with {} runs'.format(config['experiments']['n_runs']))
    logger.info('    Generating {} explanations per sample'.format(num_exp_per_sample))
    consistencies = []
    for idx in range(int(config['experiments']['n_runs'])):
        logger.info('    Trial {}/{}'.format(idx + 1, config['experiments']['n_runs']))
        explainer = get_explainer(
            name=config['explanations']['type'],
            params=explainer_params
        )

        consistency = measure_consistency(
            model=model,
            X=X_test,
            explainer=explainer,
            inference_params=config['explanations']['inference_params'],
            explainer_type=config['explanations']['type'],
            num_exp_per_sample=num_exp_per_sample,
            data_row_param_name=config['explanations']['data_row_param_name'],
            predict_fn_param_name=config['explanations']['predict_fn_param_name']
        )

        consistencies.append(consistency)

    return consistencies


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_dir', required=True, type=str,
                        help='Directory with config files')
    parser.add_argument('--log_out', required=False, type=str, default='consistency_out.log',
                        help='Place to log output')
    parser.add_argument('--save_dir', required=False, type=str, default='consistency_results',
                        help='Place to save results')
    parser.add_argument('--configs', required=False, type=str, nargs='+',
                        help='Path to individual config files')

    # Parse args
    args = parser.parse_args()
    args = vars(args)
    config_dir = args['config_dir']
    log_out = args['log_out']
    save_dir = args['save_dir']
    configs = args['configs']
    os.makedirs(save_dir, exist_ok=True)

    # Set up logging
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_out)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        handlers=[file_handler, console_handler])
    logger = logging.getLogger(__file__)

    # Run main
    main(config_dir=config_dir, save_dir=save_dir, configs=configs, metric='consistency',
         run_test_fn=run_test)
