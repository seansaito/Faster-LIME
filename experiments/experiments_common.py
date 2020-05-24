"""
Common functions used in experiment pipelines
"""
import copy
import glob
import json
import logging
import os
import pprint
import time
from datetime import datetime
from typing import Callable

import numpy as np
from sklearn.externals import joblib
from tqdm import tqdm

from experiments.constants import Explainers

logger = logging.getLogger(__name__)
pp = pprint.PrettyPrinter()


def main(config_dir: str, save_dir: str, metric: str, run_test_fn: Callable, configs: list = None):
    """
    Args:
        config_dir (str): Directory to config files
        save_dir (str): Directory to save results
        configs (list): Path to individual config files,
        run_test_fn (function): Function which the main controller runs for each config
        metric (str): Name of the metric (e.g. runtime, consistency)
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
        config_f = path.split('/')[-1]
        with open(path, 'r') as fp:
            config = json.load(fp)

        # Run experiment
        logger.info("Experiment config: {}".format(pp.pformat(config)))
        exp_type = config['experiments']['type']
        values = config['experiments']['values']
        logger.info('Experiment type is {} with values {}'.format(exp_type, values))
        list_metrics = []
        for val in values:
            logger.info('==================================')
            logger.info('    {}: {}'.format(exp_type, val))
            config_copy = copy.deepcopy(config)
            if exp_type == 'n_features':
                config_copy['dataset']['params']['n_features'] = val
            elif exp_type == 'n_explanations':
                config_copy['explanations']['n_explanations'] = val
            elif exp_type == 'num_exp_per_sample':
                config['explanations']['num_exp_per_sample'] = val
            elif exp_type == 'num_samples':
                config_copy['explanations']['inference_params']['num_samples'] = val
            else:
                logger.error('Experiment {} is not supported!'.format(exp_type))
            metric_values = run_test_fn(config_copy)
            logger.info('    Mean {}: {:.2f}'.format(metric, np.mean(metric_values)))
            logger.info('    Std {}: {:.2f}'.format(metric, np.std(metric_values)))
            list_metrics.append(metric_values)

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
                 metric=metric,
                 metric_values=list_metrics,
                 values=values,
                 exp_type=exp_type,
                 path=save_path,
                 timestamp=str_timestamp)

        res = (config_f, exp_type, list(zip(values, list(map(np.mean, list_metrics)))))
        list_results.append(res)

    for res in list_results:
        pprint.pprint(res)


def create_save_path(save_dir: str, config_name: str, timestamp: str) -> str:
    """
    Create the path to save results
    """
    t = '{config_name}_{timestamp}.pkl'.format(
        config_name=config_name,
        timestamp=timestamp
    )
    return os.path.join(save_dir, t)


def save_run(config, metric, metric_values, values, exp_type, path, timestamp):
    """
    Save the experiment
    """
    artefact = {
        'config': config,
        metric: metric_values,
        'values': values,
        'exp_type': exp_type,
        'timestamp': timestamp
    }
    joblib.dump(artefact, path)


def measure_time(predict_fn, X, explainer, inference_params, data_row_param_name,
                 predict_fn_param_name) -> float:
    """
    Measures time to produce explanations for X_test with black-box model

    Args:
        model: Trained model
        X: data
        explainer: explainer model
        inference_params: parameters when generating explanations
        data_row_param_name: Name of parameter for data when passing to explainer object
        predict_fn_param_name: Name of parameter for prediction function passed to explainer object

    Returns:
        (float) Average time to generate explanations for X
    """
    start = time.perf_counter()
    inference_params[predict_fn_param_name] = predict_fn
    for i in tqdm(range(len(X))):
        inference_params[data_row_param_name] = X[i]
        _ = explainer.explain_instance(**inference_params)
    end = time.perf_counter()
    return end - start


def measure_consistency(model, X, explainer, inference_params, explainer_type,
                        num_exp_per_sample, data_row_param_name,
                        predict_fn_param_name) -> float:
    """
    Measures consistency of explanations

    Args:
        model: Trained model
        X: data
        explainer: explainer model
        inference_params: parameters when generating explanations
        explainer_type (str): Type of explainer
        num_exp_per_sample (int): How many times we generate explanations per sample
        data_row_param_name: Name of parameter for data when passing to explainer object
        predict_fn_param_name: Name of parameter for prediction function passed to explainer object

    Returns:
        (float) mean consistency across test samples
    """
    list_explanations = []
    inference_params[predict_fn_param_name] = model.predict_proba
    for i in tqdm(range(X.shape[0])):
        exp_buffer = set()
        output = np.argmax(model.predict_proba(X[i].reshape(1, -1)))
        inference_params[data_row_param_name] = X[i]
        for i in range(num_exp_per_sample):
            if explainer_type == Explainers.LIMETABULAR:
                inference_params['labels'] = (output,)
                explanations = explainer.explain_instance(**inference_params)
                exp_buffer.add(frozenset(list(map(lambda x: x[0], explanations.as_list(output)))))
            elif explainer_type in [Explainers.NUMPY, Explainers.NUMPYENSEMBLE]:
                inference_params['label'] = output
                explanations = explainer.explain_instance(**inference_params)
                exp_buffer.add(frozenset(list(map(lambda x: x[0], explanations))))

        list_explanations.append(exp_buffer)

    list_consistency = []
    for explanations in list_explanations:
        new_buffer = []
        for exp in explanations:
            if len(new_buffer) == 0:
                new_buffer.append(exp)
            else:
                for idx, exp_ in enumerate(new_buffer):
                    if exp.issubset(exp_):
                        continue
                    elif exp_.issubset(exp):
                        # Replace with larger anchor
                        new_buffer[idx] = exp
                    else:
                        new_buffer.append(exp)

        if len(new_buffer) == 1:
            list_consistency.append(1)
        else:
            list_consistency.append(0)

    return sum(list_consistency) / len(list_consistency)
