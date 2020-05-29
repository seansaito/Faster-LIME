import numpy as np

cimport numpy as np

from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder


DTYPE_INT = np.int
DTYPE_FLOAT = np.float

ctypedef np.int_t DTYPE_INT_t
ctypedef np.float_t DTYPE_FLOAT_t

class CythonExplainer:

    def __init__(self, training_data, feature_names=None,
                 categorical_feature_idxes=None, qs=[25, 50, 75], **kwargs):
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
        cdef int num_features = training_data.shape[1]
        cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] training_data_num
        cdef np.ndarray[DTYPE_INT_t, ndim=2] training_data_cat

        self.training_data = training_data.astype(DTYPE_FLOAT)
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
            training_data_num = self.training_data[:, self.numerical_feature_idxes].astype(DTYPE_FLOAT)
            self.sc = StandardScaler(with_mean=False)
            self.sc.fit(training_data_num)
            self.qs = qs
            self.all_bins_num = np.percentile(training_data_num, self.qs, axis=0).T

        # Categorical feature statistics
        if self.categorical_features:
            training_data_cat = self.training_data[:, self.categorical_feature_idxes].astype(np.int)
            self.dict_categorical_hist = {
                feature: np.bincount(training_data_cat[:, idx]) / self.training_data.shape[0] for
                (idx, feature) in enumerate(self.categorical_features)
            }

        # Another mapping from feature to type
        self.dict_feature_to_type = {
            feature: 'categorical' if feature in self.categorical_features else 'numerical' for
            feature in self.feature_names}

    def kernel_fn(self, np.ndarray[DTYPE_FLOAT_t, ndim=2] distances, float kernel_width):
        return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

    def discretize(self, np.ndarray[DTYPE_FLOAT_t, ndim=2] X, qs=[25, 50, 75], all_bins=None):
        if all_bins is None:
            all_bins = np.percentile(X, qs, axis=0).T
        return (np.array([np.digitize(a, bins)
                          for (a, bins) in zip(X.T, all_bins)]).T.astype(DTYPE_INT), all_bins)

    def explain_instance(self, np.ndarray[long, ndim=1] data_row_1d,
                         predict_fn, label=0, num_samples=5000, num_features=10,
                         kernel_width=None, **kwargs):
        # cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] data_row
        # cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] data_synthetic_original, data_synthetic_disc
        cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] data_num
        cdef np.ndarray[DTYPE_INT_t, ndim=2] data_cat
        cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] data_synthetic_num, data_synthetic_num_original
        cdef np.ndarray[DTYPE_INT_t, ndim=2] data_synthetic_num_disc, data_cat_original, data_cat_disc
        cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] model_pred
        cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] distances
        cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] weights

        # Scale the data
        data_row = data_row_1d.reshape((1, -1))

        # Split data into numerical and categorical data and process
        list_orig = []
        list_disc = []
        if self.numerical_features:
            data_num = data_row[:, self.numerical_feature_idxes].astype(DTYPE_FLOAT)
            data_num = self.sc.transform(data_num)
            data_synthetic_num = np.tile(data_num, (num_samples, 1))
            # Add noise
            data_synthetic_num = data_synthetic_num + np.random.normal(
                size=(num_samples, data_num.shape[1]))
            data_synthetic_num[0] = data_num.ravel()
            # Convert back to original domain
            data_synthetic_num_original = self.sc.inverse_transform(data_synthetic_num)
            # Discretize
            data_synthetic_num_disc, _ = self.discretize(data_synthetic_num_original, self.qs,
                                                         self.all_bins_num)
            list_disc.append(data_synthetic_num_disc)
            list_orig.append(data_synthetic_num_original)

        if self.categorical_features:
            # Sample from training distribution for each categorical feature
            data_cat = data_row[:, self.categorical_feature_idxes].astype(DTYPE_INT)
            list_buf = []
            for feature in self.categorical_features:
                list_buf.append(np.random.choice(a=len(self.dict_categorical_hist[feature]),
                                                 size=(1, num_samples),
                                                 p=self.dict_categorical_hist[feature]))
            data_cat_original = data_cat_disc = np.concatenate(list_buf).T.astype(DTYPE_INT)
            data_cat_original[0] = data_cat.ravel()
            data_cat_disc[0] = data_cat.ravel()
            list_disc.append(data_cat_disc)
            list_orig.append(data_cat_original)

        # Concatenate the data and reorder the columns
        data_synthetic_original = np.concatenate(list_orig, axis=1)
        data_synthetic_disc = np.concatenate(list_disc, axis=1)
        data_synthetic_original = data_synthetic_original[:, self.list_reorder]
        data_synthetic_disc = data_synthetic_disc[:, self.list_reorder]

        # Get model predictions (i.e. groundtruth)
        model_pred = predict_fn(data_synthetic_original)

        # Get distances between original sample and neighbors
        if self.numerical_features:
            distances = cdist(data_synthetic_num[:1], data_synthetic_num).reshape(-1, 1)
        else:
            distances = cdist(data_synthetic_disc[:1], data_synthetic_disc).reshape(-1, 1)

        # Weight distances according to some kernel (e.g. Gaussian)
        if kernel_width is None:
            kernel_width = np.sqrt(data_row.shape[1]) * 0.75
        weights = self.kernel_fn(distances, kernel_width=kernel_width).ravel()

        # Turn discretized data into onehot
        data_synthetic_onehot = OneHotEncoder().fit_transform(data_synthetic_disc)

        # Solve
        solver = Ridge(alpha=1, fit_intercept=True)
        solver.fit(data_synthetic_onehot, model_pred[:, label], sample_weight=weights)

        # Get explanations
        importances = solver.coef_[data_synthetic_onehot[0].toarray().ravel() == 1]
        explanations = sorted(list(zip(self.feature_names, importances)),
                              key=lambda x: x[1], reverse=True)[:num_features]

        # Add '<', '>', '=' etc. to the explanations
        def map_explanations(tup):
            feature, score = tup
            feature_idx = self.dict_feature_to_idx[feature]
            feature_type = self.dict_feature_to_type[feature]
            if feature_type == 'categorical':
                exp = '{} = {}'.format(feature, data_row[0][feature_idx])
            else:
                num_bin = data_synthetic_disc[0][feature_idx]
                bins = self.all_bins_num[self.dict_num_feature_to_idx[feature]]
                if num_bin == 0:
                    exp = '{} < {}'.format(feature, bins[0])
                elif num_bin >= len(bins) - 1:
                    exp = '{} > {}'.format(feature, bins[-1])
                else:
                    exp = '{} <= {} < {}'.format(bins[num_bin - 1], feature, bins[num_bin])

            return exp, score

        explanations = list(map(map_explanations, explanations))

        return explanations


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import datasets

    data = datasets.load_breast_cancer()
    X, y = data['data'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    explainer = CythonExplainer(
        training_data=X_train,
        feature_names=data['feature_names']
    )

    exp = explainer.explain_instance(
        data_row=X_test[0],
        predict_fn=clf.predict_proba
    )
    print(exp)
