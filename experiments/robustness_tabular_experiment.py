"""
Main experimentation pipeline for measuring robustness of explainers.

Unlike the other pipelines, we just want to compare the original LIME with its robustified version,
so we do not require a list of configs to run through.

We mainly run three experiments:
* Robustness of original LIME against Fooling LIME attack (surrogate sampler)
* Robustness of CTGAN-LIME against Fooling LIME attack (surrogate sampler)
* Robustness of CTGAN-LIME against Fooling LIME attack with CTGAN sampler (white-box)

We measure the following metrics:
* How often is the biased column (e.g. race) identified as the top feature for a prediction (top-1 accuracy)
* How often is the biased column identified as among the top k features for a prediction (top-k accuracy)
* How often is 'unrelated_column' identified as the top feature for a prediction (success rate)

"""
import argparse
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from experiments.experiments_common import create_save_path
from experiments.utils.adversarial_lime import Adversarial_Lime_Model, one_hot_encode
from experiments.utils.constants import Datasets, Explainers
from experiments.utils.datasets import get_dataset
from experiments.utils.explainers import get_explainer

DATASET_CONFIGS = {
    Datasets.COMPAS: {
        'biased_column': 'race',
        'unrelated_column': 'unrelated_column',
        'use_cat_for_ctgan': True,
        'ctgan_params': {
            'embedding_dim': 512,
            'gen_dim': (256, 256, 256, 256, 256),
            'dis_dim': (256, 256, 256, 256, 256)
        },
        'discriminator_threshold': 0.5
    },
    Datasets.GERMANCREDIT: {
        'biased_column': 'Sex',
        'unrelated_column': 'unrelated_column',
        'use_cat_for_ctgan': False,
        'ctgan_params': {
            'embedding_dim': 512,
            'gen_dim': (256, 256, 256, 256, 256),
            'dis_dim': (256, 256, 256, 256, 256)
        },
        'ctgan_epochs': 300,
        'use_onehot': True,
        'measure_distance': 'raw',
        'discriminator_threshold': 0.5
    },
    Datasets.COMMUNITY: {
        'biased_column': 'racePctWhite numeric',
        'unrelated_column': 'unrelated_column',
        'use_cat_for_ctgan': True,
        'ctgan_params': {
            'embedding_dim': 512,
            'gen_dim': (256, 256, 256, 256, 256),
            'dis_dim': (256, 256, 256, 256, 256),
        },
        'ctgan_epochs': 1000,
        'use_onehot': False,
        'measure_distance': 'raw',
        'discriminator_threshold': 0.01
    }
}


class sexist_model_f:
    """
    For the German credit dataset
    """

    def __init__(self, sex_index):
        self.sex_index = sex_index

    # Decision rule: classify negatively if race is black
    def predict(self, X):
        return np.array([0 if x[self.sex_index] > 0 else 1 for x in X])

    def predict_proba(self, X):
        return one_hot_encode(self.predict(X))

    def score(self, X, y):
        return np.sum(self.predict(X) == y) / len(X)


class innocuous_model_psi_german:

    def __init__(self, unrelated_index):
        self.unrelated_index = unrelated_index

    # Decision rule: classify according to randomly drawn column 'unrelated column'
    def predict(self, X):
        return np.array([0 if x[self.unrelated_index] > 0 else 1 for x in X])

    def predict_proba(self, X):
        return one_hot_encode(self.predict(X))

    def score(self, X, y):
        return np.sum(self.predict(X) == y) / len(X)


class racist_model_f:
    """
    For the COMPAS dataset
    """

    def __init__(self, race_index):
        self.race_index = race_index

    # Decision rule: classify negatively if race is black
    def predict(self, X):
        return np.array([0 if x[self.race_index] > 0 else 1 for x in X])

    def predict_proba(self, X):
        return one_hot_encode(self.predict(X))

    def score(self, X, y):
        return np.sum(self.predict(X) == y) / len(X)


class innocuous_model_psi:

    def __init__(self, unrelated_index):
        self.unrelated_index = unrelated_index

    # Decision rule: classify according to randomly drawn column 'unrelated column'
    def predict(self, X):
        return np.array([0 if x[self.unrelated_index] > 0 else 1 for x in X])

    def predict_proba(self, X):
        return one_hot_encode(self.predict(X))

    def score(self, X, y):
        return np.sum(self.predict(X) == y) / len(X)


def preprocess_robustness_datasets(dataset, params={}):
    data = get_dataset(dataset, params)
    if dataset == Datasets.COMPAS:
        X, y, _ = data['data'], data['target'], data['cols']
        X[DATASET_CONFIGS[Datasets.COMPAS]['unrelated_column']] = np.random.choice([0, 1],
                                                                                   size=X.shape[0])
        features = list(X.columns)
        categorical_feature_name = ['two_year_recid', 'c_charge_degree_F', 'c_charge_degree_M',
                                    'sex_Female', 'sex_Male', 'race', 'unrelated_column']

        categorical_feature_indcs = [features.index(c) for c in categorical_feature_name]

        X = X.values

    elif dataset == Datasets.GERMANCREDIT:
        X, y = data['data'], data['target']
        X = pd.DataFrame(X, columns=data['feature_names'])
        X[DATASET_CONFIGS[Datasets.GERMANCREDIT]['unrelated_column']] = np.random.choice([0, 1],
                                                                                         size=
                                                                                         X.shape[0])
        features = list(X.columns)
        categorical_feature_name = data['categorical_features'] + ['unrelated_column']
        categorical_feature_indcs = [features.index(c) for c in categorical_feature_name]

        X = X.values

    elif dataset == Datasets.ADULT:
        X, y = data['data'], data['target']
        X[DATASET_CONFIGS[Datasets.ADULT]['unrelated_column']] = np.random.choice([0, 1],
                                                                                  size=
                                                                                  X.shape[0])
        features = list(X.columns)
        categorical_feature_name = data['categorical_features'] + ['unrelated_column']
        categorical_feature_indcs = [features.index(c) for c in categorical_feature_name]
        X = X.values

    elif dataset == Datasets.COMMUNITY:
        X, y = data['data'], data['target']
        X[DATASET_CONFIGS[Datasets.COMMUNITY]['unrelated_column']] = np.random.choice(
            [0, 1],
            size=
            X.shape[0]).astype(int)
        features = list(X.columns)
        categorical_feature_name = [DATASET_CONFIGS[Datasets.COMMUNITY]['unrelated_column']]
        categorical_feature_indcs = [features.index(c) for c in categorical_feature_name]
        X = X.values
    else:
        raise KeyError('Dataset {} not available'.format(dataset))

    numerical_features = [f for f in features if f not in categorical_feature_name]
    numerical_feature_indcs = [features.index(c) for c in numerical_features]
    sc = StandardScaler()
    X[:, numerical_feature_indcs] = sc.fit_transform(X[:, numerical_feature_indcs])

    return X, y, features, categorical_feature_name, categorical_feature_indcs


def get_explanations(explainer, X, adv_lime, explainer_name, top_features=3, num_samples=1000):
    list_top_k = []

    for idx in tqdm(range(X.shape[0])):
        label = np.argmax(adv_lime.predict_proba(X[idx].reshape((1, -1))))
        if explainer_name == Explainers.LIMETABULAR:
            exp = explainer.explain_instance(X[idx], adv_lime.predict_proba,
                                             num_features=top_features,
                                             labels=(0, 1)).as_list(label)
        else:
            exp = explainer.explain_instance(X[idx], adv_lime.predict_proba, label=label,
                                             num_samples=num_samples,
                                             num_features=top_features)
        top_k = [e[0] for e in exp]
        list_top_k.append(top_k)

    return list_top_k


def measure_robustness(dataset, top_features=3, params={}):
    X, y, features, categorical_feature_name, categorical_feature_indcs = preprocess_robustness_datasets(
        dataset, params)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    biased_column = DATASET_CONFIGS[dataset]['biased_column']
    unrelated_column = DATASET_CONFIGS[dataset]['unrelated_column']
    biased_index = features.index(biased_column)
    unrelated_index = features.index(unrelated_column)

    # Get original lime model
    logger.info('Initializing original LIME')
    original_lime_params = {
        'training_data': X_train,
        'feature_names': features,
        'discretize_continuous': False,
        'categorical_features': categorical_feature_indcs
    }
    original_lime = get_explainer(Explainers.LIMETABULAR, original_lime_params)

    # Train the adversarial model for LIME with f and psi
    logger.info('Initializing Fooling LIME')
    if dataset in [Datasets.COMPAS, Datasets.COMMUNITY]:
        biased_model = racist_model_f(biased_index)
        innocuous_model = innocuous_model_psi(unrelated_index)
    elif dataset in [Datasets.GERMANCREDIT, Datasets.ADULT]:
        biased_model = sexist_model_f(biased_index)
        innocuous_model = innocuous_model_psi_german(unrelated_index)
    else:
        raise KeyError('Dataset not supported: {}'.format(dataset))

    adv_lime = Adversarial_Lime_Model(
        biased_model,
        innocuous_model).train(X_train, y_train,
                               feature_names=features,
                               categorical_features=categorical_feature_indcs)

    adv_lime_for_explainer = Adversarial_Lime_Model(
        biased_model,
        innocuous_model).train(X_train, y_train,
                               feature_names=features,
                               categorical_features=categorical_feature_indcs)

    # Get robust lime
    logger.info('Initializing CTGAN-LIME')
    robust_lime_params = {
        'training_data': X,
        'feature_names': features,
        'categorical_feature_idxes': categorical_feature_indcs,
        'ctgan_epochs': DATASET_CONFIGS[dataset].get('ctgan_epochs', 300),
        'ctgan_verbose': True,
        'use_cat_for_ctgan': DATASET_CONFIGS[dataset]['use_cat_for_ctgan'],
        'discriminator': adv_lime_for_explainer,
        'discriminator_threshold': DATASET_CONFIGS[dataset].get('discriminator_threshold', 0.5),
        'ctgan_params': DATASET_CONFIGS[dataset]['ctgan_params'],
        'use_onehot': DATASET_CONFIGS[dataset].get('use_onehot', True),
        'measure_distance': DATASET_CONFIGS[dataset].get('measure_distance', 'raw')
    }

    robust_lime = get_explainer(Explainers.NUMPYROBUSTTABULAR, robust_lime_params)

    # Train the adversarial model with CTGAN
    logger.info('Initializing Fooling LIME with CTGAN')
    adv_ctgan_lime = Adversarial_Lime_Model(
        biased_model,
        innocuous_model).train_ctgan(
        X_train, y_train,
        ctgan_sampler=robust_lime.ctgan_sampler,
        feature_names=features,
        categorical_features=categorical_feature_indcs)

    model_pairs = [
        (original_lime, adv_lime),
        (original_lime, adv_ctgan_lime),
        (robust_lime, adv_lime),
        (robust_lime, adv_ctgan_lime)
    ]
    name_pairs = [
        (Explainers.LIMETABULAR, 'Fooling LIME'),
        (Explainers.LIMETABULAR, 'Fooling LIME v2'),
        (Explainers.NUMPYROBUSTTABULAR, 'Fooling LIME'),
        (Explainers.NUMPYROBUSTTABULAR, 'Fooling LIME with CTGAN')
    ]

    list_results = []

    for (explainer, attacker), (e_name, a_name) in zip(model_pairs, name_pairs):
        logger.info('=========================================')
        logger.info('Measuring {} against {}'.format(e_name, a_name))
        top_explanations = get_explanations(
            explainer=explainer,
            X=X_test,
            adv_lime=attacker,
            explainer_name=e_name,
            top_features=top_features
        )
        top_1 = np.mean(list(map(lambda x: biased_column in x[0], top_explanations)))
        top_k = np.mean(list(map(lambda x: any([biased_column in e for e in x]), top_explanations)))
        success_rate = np.mean(list(map(lambda x: unrelated_column in x[0], top_explanations)))
        logger.info('Top 1 accuracy: {:.4f}'.format(top_1))
        logger.info('Top k ({}) accuracy: {:.4f}'.format(top_features, top_k))
        logger.info('Attack success rate: {:.4f}'.format(success_rate))
        list_results.append((top_1, top_k, success_rate))

    return list(zip(name_pairs, list_results))


def main(dataset, num_runs, top_features, params={}):
    dict_results = {}
    logger.info('Testing on dataset: {}'.format(dataset))
    logger.info('Going through {} runs'.format(num_runs))

    for run in range(num_runs):
        logger.info('=============================================')
        logger.info('Run: {}'.format(run))
        list_results = measure_robustness(dataset, top_features=top_features, params=params)
        for (name_pair, result) in list_results:
            e_name, a_name = name_pair
            top_1_acc, top_k_acc, attack_success_rate = result
            key = '{} vs. {}'.format(e_name, a_name)
            if key in dict_results:
                dict_results[key]['top_1_acc'].append(top_1_acc)
                dict_results[key]['top_k_acc'].append(top_k_acc)
                dict_results[key]['attack_success_rate'].append(attack_success_rate)
            else:
                dict_results[key] = {
                    'top_1_acc': [top_1_acc],
                    'top_k_acc': [top_k_acc],
                    'attack_success_rate': [attack_success_rate]
                }

    logger.info('Finished running experiments')
    for key, items in dict_results.items():
        logger.info('Results for {}'.format(key))
        logger.info('Mean top 1 accuracy: {:.4f} (+/- {:.4f})'.format(np.mean(items['top_1_acc']),
                                                                      np.std(items['top_1_acc'])))
        logger.info('Mean top k ({}) accuracy: {:.4f} (+/- {:.4f})'.format(top_features, np.mean(
            items['top_k_acc']),
                                                                           np.std(
                                                                               items['top_k_acc'])))
        logger.info(
            'Mean success rate: {:.4f} (+/- {:.4f})'.format(np.mean(items['attack_success_rate']),
                                                            np.std(items['attack_success_rate'])))

    now = datetime.now()
    timestamp = str(int(datetime.timestamp(now)))
    save_path = create_save_path(
        save_dir=save_dir,
        config_name='{}_robustness_results_top_features_{}'.format(dataset, top_features),
        timestamp=timestamp
    )
    logger.info('Saving results to {}'.format(save_path))
    joblib.dump(dict_results, save_path)
    return dict_results


if __name__ == '__main__':
    np.random.seed(123456)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--log_out', required=False, type=str,
                        default='',
                        help='Place to log output')
    parser.add_argument('--save_dir', required=False, type=str,
                        default='',
                        help='Place to save results')
    parser.add_argument('--dataset', required=False, type=str, default=Datasets.COMPAS,
                        help='Dataset to measure robustness with')
    parser.add_argument('--num_runs', required=False, type=int, default=1,
                        help='Number of trials to run')
    parser.add_argument('--max_features', required=False, type=int, default=3,
                        help='How many features to iterate through?')

    # Parse args
    args = parser.parse_args()
    args = vars(args)
    log_out = args['log_out']
    save_dir = args['save_dir']
    dataset = args['dataset']
    num_runs = int(args['num_runs'])
    max_features = int(args['max_features'])

    if not log_out:
        log_out = 'experiments/logs/{}_robustness.log'.format(dataset)

    if not save_dir:
        save_dir = 'experiments/robustness_{}_results'.format(dataset)

    os.makedirs(save_dir, exist_ok=True)

    # Set up logging
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_out)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        handlers=[file_handler, console_handler])
    logger = logging.getLogger(__file__)

    # Run main
    logger.info('Received following params: {}'.format(args))
    list_all_results = []
    for top_features in range(max_features):
        dict_results = main(
            dataset=dataset,
            num_runs=num_runs,
            top_features=int(top_features)
        )
        list_all_results.append({top_features: dict_results})

    now = datetime.now()
    timestamp = str(int(datetime.timestamp(now)))
    save_path = create_save_path(
        save_dir=save_dir,
        config_name='{}_robustness_results_max_features_{}'.format(dataset, max_features),
        timestamp=timestamp
    )
    logger.info('Saving results to {}'.format(save_path))
    joblib.dump(list_all_results, save_path)
    logger.info('Finished running experiments')
