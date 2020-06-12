import sklearn.datasets as scikit_datasets

from experiments.utils.constants import Datasets
from experiments.utils.dataset_utils import load_german_credit_dataset, \
    get_and_preprocess_compas_data, get_and_preprocess_adult_data, get_and_preprocess_cc

datasets = {
    Datasets.BREASTCANCER: scikit_datasets.load_breast_cancer,
    Datasets.TWENTYNEWSGROUPS: scikit_datasets.fetch_20newsgroups,
    Datasets.SYNTHETIC: scikit_datasets.make_classification,
    Datasets.GERMANCREDIT: load_german_credit_dataset,
    Datasets.COMPAS: get_and_preprocess_compas_data,
    Datasets.ADULT: get_and_preprocess_adult_data,
    Datasets.COMMUNITY: get_and_preprocess_cc
}


def get_dataset(name, params):
    if name in datasets:
        if name in [Datasets.BREASTCANCER, Datasets.GERMANCREDIT, Datasets.ADULT,
                    Datasets.COMMUNITY]:
            data = datasets[name]()
        elif name == Datasets.COMPAS:
            data = datasets[name](**params)
        elif name == Datasets.SYNTHETIC:
            X, y = datasets[name](**params)
            data = {
                'data': X,
                'target': y
            }
        elif name == Datasets.TWENTYNEWSGROUPS:
            raw_train = scikit_datasets.fetch_20newsgroups(subset='train', **params)
            raw_test = scikit_datasets.fetch_20newsgroups(subset='test', **params)
            data = {
                'raw_train': raw_train,
                'raw_test': raw_test
            }
        else:
            data = {}
        return data
    else:
        raise KeyError('Dataset {} not found or is not supported'.format(name))
