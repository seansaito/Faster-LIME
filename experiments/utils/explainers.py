from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer

from experiments.utils.constants import Explainers

from faster_lime.explainers.numpy_tabular_explainer import NumpyTabularExplainer
from faster_lime.explainers.numpy_ensemble_explainer import NumpyEnsembleExplainer
from faster_lime.explainers.numpy_text_explainer import NumpyTextExplainer
from faster_lime.explainers.numpy_robust_tabular_explainer import NumpyRobustTabularExplainer

explainers = {
    Explainers.LIMETABULAR: LimeTabularExplainer,
    Explainers.LIMETEXT: LimeTextExplainer,
    Explainers.NUMPYTABULAR: NumpyTabularExplainer,
    Explainers.NUMPYENSEMBLE: NumpyEnsembleExplainer,
    Explainers.NUMPYTEXT: NumpyTextExplainer,
    Explainers.NUMPYROBUSTTABULAR: NumpyRobustTabularExplainer
}

def get_explainer(name, params):
    if name in explainers:
        return explainers[name](**params)
    else:
        raise KeyError('Explainer {} not found or is not supported'.format(name))
