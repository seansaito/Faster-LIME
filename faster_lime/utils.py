import numpy as np
from sklearn.linear_model import Ridge

dict_disc_to_bin = {
    'quartile': [25, 50, 75],
    'quintile': [20, 40, 60, 80],
    'decile': [10, 20, 30, 40, 50, 60, 70, 80, 90]
}


def ridge_solve(tup):
    data_synthetic_onehot, model_pred, weights = tup
    solver = Ridge(alpha=1, fit_intercept=True)
    solver.fit(data_synthetic_onehot,
               model_pred,
               sample_weight=weights.ravel())
    # Get explanations
    importance = solver.coef_[
        data_synthetic_onehot[0].toarray().ravel() == 1].ravel()
    return importance


def kernel_fn(distances, kernel_width):
    return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))


def discretize(X, percentiles=[25, 50, 75], all_bins=None):
    if all_bins is None:
        all_bins = np.percentile(X, percentiles, axis=0).T
    return (np.array([np.digitize(a, bins)
                      for (a, bins) in zip(X.T, all_bins)]).T, all_bins)
