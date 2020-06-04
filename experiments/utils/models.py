from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from experiments.utils.constants import Models

models = {
    Models.RANDOMFORESTCLASSIFIER: RandomForestClassifier,
    Models.LOGISTICREGRESSION: LogisticRegression,
    Models.MULTINOMIALNB: MultinomialNB,
    Models.MLPCLASSIFIER: MLPClassifier
}


def get_model(name, params):
    if name in models:
        return models[name](**params)
    else:
        raise KeyError('Model name {} does not exist or is not supported yet'.format(name))
