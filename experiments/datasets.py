import sklearn.datasets as scikit_datasets

from experiments.constants import Datasets
from experiments.dataset_utils import load_german_credit_dataset

datasets = {
    Datasets.BREASTCANCER: scikit_datasets.load_breast_cancer,
    Datasets.TWENTYNEWSGROUPS: scikit_datasets.fetch_20newsgroups,
    Datasets.SYNTHETIC: scikit_datasets.make_classification,
    Datasets.GERMANCREDIT: load_german_credit_dataset
}


def get_dataset(name, params):
    if name in datasets:
        if name in [Datasets.BREASTCANCER, Datasets.GERMANCREDIT]:
            data = datasets[name]()
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
