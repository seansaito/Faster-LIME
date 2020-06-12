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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from experiments.utils.constants import Explainers

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
                config_copy['explanations']['num_exp_per_sample'] = val
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
        timestamp = datetime.strftime(now, '%Y%m%d%H%M%S')
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
                 timestamp=timestamp)

        res = (config_f, exp_type, list(zip(values, list(map(np.mean, list_metrics)))))
        list_results.append(res)

    for res in list_results:
        pprint.pprint(res)


def create_save_path(save_dir: str, config_name: str, timestamp: str) -> str:
    """
    Create the path to save results
    """
    t = '{config_name}-{timestamp}.pkl'.format(
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
    Measures average time to generate explanations for one sample

    Args:
        model: Trained model
        X: data
        explainer: explainer model
        inference_params: parameters when generating explanations
        data_row_param_name: Name of parameter for data when passing to explainer object
        predict_fn_param_name: Name of parameter for prediction function passed to explainer object

    Returns:
        (float) Average time to generate one explanation
    """
    start = time.perf_counter()
    inference_params[predict_fn_param_name] = predict_fn
    for i in tqdm(range(len(X))):
        inference_params[data_row_param_name] = X[i]
        _ = explainer.explain_instance(**inference_params)
    end = time.perf_counter()
    total_time = end - start
    avg_time = total_time / len(X)
    return avg_time


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
            elif explainer_type in [Explainers.NUMPYTABULAR, Explainers.NUMPYENSEMBLE,
                                    Explainers.NUMPYROBUSTTABULAR]:
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


def measure_precision(model, X, binarizer, feature_names, explainer, inference_params,
                      explainer_type,
                      data_row_param_name, predict_fn_param_name):
    """
    Measures precision of the model across a set of data

    Args:
        model:
        X:
        binarizer:
        feature_names:
        explainer:
        inference_params:
        explainer_type:
        data_row_param_name:
        predict_fn_param_name:

    Returns:
        (float) mean precision across test samples
    """
    precisions = []
    inference_params[predict_fn_param_name] = model.predict_proba
    for i in tqdm(range(X.shape[0])):
        # Get explanations
        inference_params[data_row_param_name] = X[i]
        explanations = explainer.explain_instance(**inference_params)
        if explainer_type == Explainers.LIMETABULAR:
            features_exp = list(map(lambda x: x[0], explanations.as_list(1)))
        elif explainer_type in [Explainers.NUMPYTABULAR, Explainers.NUMPYENSEMBLE,
                                Explainers.NUMPYROBUSTTABULAR]:
            features_exp = list(map(lambda x: x[0], explanations))
        else:
            features_exp = []

        features_used = []
        features_idx = []
        for feature_exp in features_exp:
            for idx, f in enumerate(feature_names):
                if f in feature_exp:
                    features_used.append(f)
                    features_idx.append(idx)
                    break

        features_idx = sorted(features_idx)

        data_row = X[i]
        neighborhood = np.concatenate((X[:i], X[i + 1:]))
        similar = binarizer.fetch_similar(
            data_row=data_row,
            test_data=neighborhood,
            feature_idxes=features_idx
        )

        if similar.shape[0] > 0:
            precision = np.mean(model.predict(similar) ==
                                model.predict(data_row.reshape(1, -1)))
            precisions.append(precision)

    if precisions:
        mean_precision = np.mean(precisions)
    else:
        mean_precision = 0

    return mean_precision


def measure_coverage(model, X, binarizer, feature_names, explainer, inference_params,
                     explainer_type,
                     data_row_param_name, predict_fn_param_name):
    """
    Measures coverage of the model across a set of data

    Args:
        model:
        X:
        binarizer:
        feature_names:
        explainer:
        inference_params:
        explainer_type:
        data_row_param_name:
        predict_fn_param_name:

    Returns:
        (float) mean coverage across test samples
    """
    coverages = []
    inference_params[predict_fn_param_name] = model.predict_proba
    for i in tqdm(range(X.shape[0])):
        # Get explanations
        inference_params[data_row_param_name] = X[i]
        explanations = explainer.explain_instance(**inference_params)
        if explainer_type == Explainers.LIMETABULAR:
            features_exp = list(map(lambda x: x[0], explanations.as_list(1)))
        elif explainer_type in [Explainers.NUMPYTABULAR, Explainers.NUMPYENSEMBLE,
                                Explainers.NUMPYROBUSTTABULAR]:
            features_exp = list(map(lambda x: x[0], explanations))
        else:
            features_exp = []

        features_used = []
        features_idx = []
        for feature_exp in features_exp:
            for idx, f in enumerate(feature_names):
                if f in feature_exp:
                    features_used.append(f)
                    features_idx.append(idx)
                    break

        features_idx = sorted(features_idx)

        data_row = X[i]
        neighborhood = np.concatenate((X[:i], X[i + 1:]))
        similar = binarizer.fetch_similar(
            data_row=data_row,
            test_data=neighborhood,
            feature_idxes=features_idx
        )

        if similar.shape[0] == 0:
            coverage = 0
        else:
            coverage = similar.shape[0] / X.shape[0]

        coverages.append(coverage)

    mean_coverage = np.mean(coverages)
    return mean_coverage


class Binarizer:

    def __init__(self, training_data, feature_names=None,
                 categorical_feature_idxes=None,
                 qs=[25, 50, 75], **kwargs):
        """
        Args:
            training_data (np.ndarray): Training data to measure training data statistics
            feature_names (list): List of feature names
            categorical_feature_idxes (list): List of idxes of features that are categorical
            qs (list): Discretization bins

        Assumptions:
            * Data only contains categorical and/or numerical data
            * Categorical data is already converted to ordinal labels (e.g. via scikit-learn's
                OrdinalEncoder)

        """
        self.training_data = training_data
        self.num_features = self.training_data.shape[1]

        # Parse columns
        if feature_names is not None:
            # TODO input validation
            self.feature_names = list(feature_names)
        else:
            self.feature_names = list(range(self.num_features))
        self.categorical_feature_idxes = categorical_feature_idxes
        if self.categorical_feature_idxes:
            self.categorical_features = [self.feature_names[i] for i in
                                         self.categorical_feature_idxes]
            self.numerical_features = [f for f in self.feature_names if
                                       f not in self.categorical_features]
            self.numerical_feature_idxes = [idx for idx in range(self.num_features) if
                                            idx not in self.categorical_feature_idxes]
        else:
            self.categorical_features = []
            self.numerical_features = self.feature_names
            self.numerical_feature_idxes = list(range(self.num_features))

        # Some book-keeping: keep track of the original indices of each feature
        self.dict_num_feature_to_idx = {feature: idx for (idx, feature) in
                                        enumerate(self.numerical_features)}
        self.dict_feature_to_idx = {feature: idx for (idx, feature) in
                                    enumerate(self.feature_names)}
        self.list_reorder = [self.dict_feature_to_idx[feature] for feature in
                             self.numerical_features + self.categorical_features]

        # Get training data statistics
        # Numerical feature statistics
        if self.numerical_features:
            training_data_num = self.training_data[:, self.numerical_feature_idxes]
            self.sc = StandardScaler(with_mean=False)
            self.sc.fit(training_data_num)
            self.qs = qs
            self.all_bins_num = np.percentile(training_data_num, self.qs, axis=0).T

        # Categorical feature statistics
        if self.categorical_features:
            training_data_cat = self.training_data[:, self.categorical_feature_idxes]
            self.dict_categorical_hist = {
                feature: np.bincount(training_data_cat[:, idx]) / self.training_data.shape[0] for
                (idx, feature) in enumerate(self.categorical_features)
            }

        # Another mapping fr om feature to type
        self.dict_feature_to_type = {
            feature: 'categorical' if feature in self.categorical_features else 'numerical' for
            feature in self.feature_names}

    def discretize(self, X, qs=[25, 50, 75], all_bins=None):
        if all_bins is None:
            all_bins = np.percentile(X, qs, axis=0).T
        return (np.array([np.digitize(a, bins)
                          for (a, bins) in zip(X.T, all_bins)]).T, all_bins)

    def fetch_similar(self, data_row, test_data, feature_idxes):
        """
        Fetch data from test_data which binarized features match those of data_row
        """
        # Scale the data
        data_row = data_row.reshape((1, -1))

        # Split data into numerical and categorical data and process
        list_disc = []
        if self.numerical_features:
            data_num = data_row[:, self.numerical_feature_idxes]
            test_data_num = test_data[:, self.numerical_feature_idxes]

            data_num = np.concatenate((data_num, test_data_num))

            # Discretize
            data_synthetic_num_disc, _ = self.discretize(data_num, self.qs,
                                                         self.all_bins_num)
            list_disc.append(data_synthetic_num_disc)

        if self.categorical_features:
            # Sample from training distribution for each categorical feature
            data_cat = data_row[:, self.categorical_feature_idxes]
            test_data_cat = test_data[:, self.categorical_feature_idxes]
            data_cat = np.concatenate((data_cat, test_data_cat))

            list_disc.append(data_cat)

        # Concatenate the data and reorder the columns
        data_synthetic_disc = np.concatenate(list_disc, axis=1)
        data_synthetic_disc = data_synthetic_disc[:, self.list_reorder]

        data_instance_disc = data_synthetic_disc[0]
        test_data_disc = data_synthetic_disc[1:]

        # Fetch neighbors from real test data where top features are the same
        same_features = np.where(np.all(test_data_disc[:, feature_idxes] ==
                                        data_instance_disc[feature_idxes], axis=1))[0]
        similar_neighbors = test_data[same_features]
        return similar_neighbors
