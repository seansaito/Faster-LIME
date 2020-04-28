from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer

from experiments.constants import Explainers

from faster_lime.explainers.numpy_explainer import NumpyExplainer

explainers = {
    Explainers.LIMETABULAR: LimeTabularExplainer,
    Explainers.LIMETEXT: LimeTextExplainer,
    Explainers.NUMPY: NumpyExplainer
}

def get_explainer(name, params):
    if name in explainers:
        return explainers[name](**params)
    else:
        raise KeyError('Explainer {} not found or is not supported'.format(name))
