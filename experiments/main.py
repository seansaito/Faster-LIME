"""
Main experimentation pipeline.

Things we would like to measure against run-time:
* Number of features
* Number of examples
* Number of synthetic examples generated per explanation
"""
import argparse
import glob
import json
import logging
import os
import pprint
import time
from datetime import datetime
from typing import List
import copy

import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from experiments.datasets import get_dataset
from experiments.explainers import get_explainer
from experiments.models import get_model


def measure(model, X, explainer, inference_params) -> float:
    """
    Measures time to produce explanations for X_test with black-box model

    Args:
        model: Trained model
        X: data
        explainer: explainer model
        inference_params: parameters when generating explanations

    Returns:
        (float) Average time to generate explanations for X
    """
    start = time.perf_counter()
    inference_params['predict_fn'] = model.predict_proba
    for i in tqdm(range(X.shape[0])):
        inference_params['data_row'] = X[i]
        _ = explainer.explain_instance(**inference_params)
    end = time.perf_counter()
    return end - start


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
    X = dataset['data']
    y = dataset['target']

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

    # Measure time
    runtimes = []
    logger.info('    Testing with {} runs'.format(config['experiments']['n_runs']))
    for idx in range(int(config['experiments']['n_runs'])):
        logger.info('    Trial {}/{}'.format(idx + 1, config['experiments']['n_runs']))
        explainer = get_explainer(
            name=config['explanations']['type'],
            params=explainer_params
        )
        runtime = measure(
            model=model,
            X=X_test,
            explainer=explainer,
            inference_params=config['explanations']['inference_params']
        )
        runtimes.append(runtime)

    return runtimes


def save_run(config, runtimes, values, exp_type, path, timestamp):
    """
    Save the experiment
    """
    artefact = {
        'config': config,
        'runtimes': runtimes,
        'values': values,
        'exp_type': exp_type,
        'timestamp': timestamp
    }
    joblib.dump(artefact, path)


def create_save_path(save_dir: str, config_name: str, timestamp: str) -> str:
    """
    Create the path to save results
    """
    t = '{config_name}_{timestamp}.pkl'.format(
        config_name=config_name,
        timestamp=timestamp
    )
    return os.path.join(save_dir, t)


def main(config_dir: str, save_dir: str, configs: list = None):
    """
    Args:
        config_dir (str): Directory to config files
        save_dir (str): Directory to save results
        configs (list): Path to individual config files
    """
    if configs:
        list_config_files = configs
    else:
        list_config_files = glob.glob(os.path.join(config_dir, '*.json'), recursive=False)
    logger.info('{} configs found: {}'.format(
        len(list_config_files), list_config_files))

    list_results = []

    for path in list_config_files:
        logger.info('============================================================')
        try:
            config_f = path.split('/')[-1]
            with open(path, 'r') as fp:
                config = json.load(fp)

            # Run experiment
            logger.info("Experiment config: {}".format(pp.pformat(config)))
            exp_type = config['experiments']['type']
            values = config['experiments']['values']
            logger.info('Experiment type is {} with values {}'.format(exp_type, values))
            list_runtimes = []
            for val in values:
                logger.info('==================================')
                logger.info('    {}: {}'.format(exp_type, val))
                config_copy = copy.deepcopy(config)
                if exp_type == 'n_features':
                    config_copy['dataset']['params']['n_features'] = val
                elif exp_type == 'n_explanations':
                    config_copy['explanations']['n_explanations'] = val
                elif exp_type == 'num_samples':
                    config_copy['explanations']['inference_params']['num_samples'] = val
                else:
                    logger.error('Experiment {} is not supported!'.format(exp_type))
                runtimes = run_test(config_copy)
                logger.info('    Mean runtime: {:.2f}'.format(np.mean(runtimes)))
                logger.info('    Std runtime: {:.2f}'.format(np.std(runtimes)))
                list_runtimes.append(runtimes)

            # Save results
            now = datetime.now()
            timestamp = str(int(datetime.timestamp(now)))
            str_timestamp = str(now)
            save_path = create_save_path(
                save_dir=save_dir,
                config_name=config_f.split('.')[0],
                timestamp=timestamp
            )
            logger.info('Saving results to {}'.format(save_path))
            save_run(config=config,
                     runtimes=list_runtimes,
                     values=values,
                     exp_type=exp_type,
                     path=save_path,
                     timestamp=str_timestamp)

            res = (config_f, exp_type, list(zip(values, list(map(np.mean, list_runtimes)))))
            list_results.append(res)

        except Exception as e:
            logger.error(pp.pformat(e))

    for res in list_results:
        pprint.pprint(res)


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
    pp = pprint.PrettyPrinter()

    # Run main
    main(config_dir, save_dir, configs)
