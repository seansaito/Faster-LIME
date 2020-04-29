import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class NumpyExplainer:

    def __init__(self, training_data, features_names=None, qs=[25, 50, 75], **kwargs):
        self.training_data = training_data
        self.feature_names = features_names
        self.sc = StandardScaler(with_mean=False)
        self.sc.fit(self.training_data)
        self.qs = qs
        self.all_bins = np.percentile(training_data, qs, axis=0).T
        self.oe = OneHotEncoder()
        train_disc = self.discretize(training_data, self.qs, training_data.shape[1])
        self.oe.fit(train_disc)

    def kernel_fn(self, distances, kernel_width):
        return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

    def discretize(self, X, qs=[25, 50, 75], all_bins=None):
        if all_bins is None:
            all_bins = np.percentile(X, qs, axis=0).T
        return (np.array([np.digitize(a, bins)
                          for (a, bins) in zip(X.T, all_bins)]).T, all_bins)

    def explain_instance(self, data_row, predict_fn, num_samples=5000, num_features=10,
                         kernel_width=None, **kwargs):
        # Scale the data
        data_row = data_row.reshape((1, -1))
        data_scaled = self.sc.transform(data_row)

        if kernel_width is None:
            kernel_width = np.sqrt(data_row.shape[1]) * 0.75

        # Create synthetic neighborhood
        X_synthetic = np.tile(data_scaled, (num_samples, 1))
        X_synthetic = X_synthetic + np.random.normal(size=(num_samples, data_row.shape[1]))
        X_synthetic[0] = data_scaled.ravel()
        X_synthetic_orig = self.sc.inverse_transform(X_synthetic)
        X_synthetic_disc, all_bins = self.discretize(X_synthetic_orig, self.qs, self.all_bins)
        X_synthetic_onehot = self.oe.transform(X_synthetic_disc)

        # Get model predictions (i.e. groundtruth)
        model_pred = predict_fn(X_synthetic_orig)

        # Solve
        distances = cdist(X_synthetic[:1], X_synthetic)
        distances = distances.reshape(-1, 1)
        weights = self.kernel_fn(distances, kernel_width=kernel_width).ravel()
        solver = Ridge(alpha=1, fit_intercept=True)
        solver.fit(X_synthetic_onehot, model_pred[:, 0], sample_weight=weights)

        # Get explanations
        importances = solver.coef_[X_synthetic_onehot[0].toarray().ravel() == 1]
        if self.feature_names:
            explanations = sorted(list(zip(self.feature_names, importances)),
                                  key=lambda x: x[1], reverse=True)[:num_features]
        else:
            explanations = sorted(list(zip(range(data_row.shape[1]), importances)),
                                  key=lambda x: x[1], reverse=True)[:num_features]
        return explanations
