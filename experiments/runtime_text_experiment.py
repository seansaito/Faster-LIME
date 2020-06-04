"""
Main experimentation pipeline for measuring runtime on tabular datasets

Things we would like to measure against run-time:
* Number test samples
* Number of synthetic samples
"""
import argparse
import logging
import os
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from experiments.utils.datasets import get_dataset
from experiments.experiments_common import main, measure_time
from experiments.utils.explainers import get_explainer
from experiments.utils.models import get_model


def run_test(config: dict) -> List[float]:
    """
    Run runtime experiment with a given configuration
    Args:
        config (dict): Configuration

    Returns:
        (list) Runtimes in seconds
    """
    # Instantiate model
    model_name = config['model']['name']
    model_params = config['model']['params']
    model = get_model(model_name, model_params)

    # Load dataset
    dataset_name = config['dataset']['name']
    dataset_params = config['dataset'].get('params', {})
    dataset = get_dataset(dataset_name, dataset_params)

    raw_train = dataset['raw_train']
    raw_test = dataset['raw_test']

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(raw_train.data)
    y_train = raw_train.target

    X_test = vectorizer.transform(raw_test.data)
    y_test = raw_test.target

    model.fit(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    logger.info('    Test accuracy: {}'.format(test_acc))

    def predict_fn(instance):
        vec = vectorizer.transform(instance)
        return model.predict_proba(vec)

    # Instantiate explainer
    explainer_params = config['explanations']['model_params']

    # Measure time
    runtimes = []
    logger.info('    Testing with {} runs'.format(config['experiments']['n_runs']))
    for idx in range(int(config['experiments']['n_runs'])):
        logger.info('    Trial {}/{}'.format(idx + 1, config['experiments']['n_runs']))
        explainer = get_explainer(
            name=config['explanations']['type'],
            params=explainer_params
        )
        runtime = measure_time(
            predict_fn=predict_fn,
            X=raw_test.data,
            explainer=explainer,
            inference_params=config['explanations']['inference_params'],
            data_row_param_name=config['explanations']['data_row_param_name'],
            predict_fn_param_name=config['explanations']['predict_fn_param_name']
        )
        runtimes.append(runtime)

    return runtimes


if __name__ == '__main__':
    np.random.seed(123456)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_dir', required=True, type=str,
                        help='Directory with config files')
    parser.add_argument('--log_out', required=False, type=str, default='out.log',
                        help='Place to log output')
    parser.add_argument('--save_dir', required=False, type=str, default='results',
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
    main(config_dir=config_dir, save_dir=save_dir, metric='runtime', configs=configs,
         run_test_fn=run_test)
