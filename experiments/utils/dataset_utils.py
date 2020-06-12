import os

import numpy as np
import pandas as pd
from ctgan import load_demo
from sklearn.preprocessing import LabelEncoder


def get_and_preprocess_adult_data():
    df_data = load_demo()
    feature_names = list(df_data.columns)
    target_col = 'income'
    feature_names.remove(target_col)

    categorical_features = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
    ]

    # Convert categorical features to ordinal
    dict_le = {}
    for cat_col in categorical_features + [target_col]:
        le = LabelEncoder()
        df_data[cat_col] = le.fit_transform(df_data[cat_col])
        dict_le[cat_col] = le

    X, y = df_data[feature_names], df_data[target_col].values

    return {
        'data': X,
        'target': y,
        'feature_names': feature_names,
        'categorical_features': categorical_features
    }


def get_and_preprocess_compas_data(encode=False):
    """
    Handle processing of COMPAS according to: https://github.com/propublica/compas-analysis
    """
    PROTECTED_CLASS = 1
    UNPROTECTED_CLASS = 0
    POSITIVE_OUTCOME = 1
    NEGATIVE_OUTCOME = 0

    path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                        "data/compas-scores-two-years.csv"))
    compas_df = pd.read_csv(path, index_col=0)
    compas_df = compas_df.loc[(compas_df['days_b_screening_arrest'] <= 30) &
                              (compas_df['days_b_screening_arrest'] >= -30) &
                              (compas_df['is_recid'] != -1) &
                              (compas_df['c_charge_degree'] != "O") &
                              (compas_df['score_text'] != "NA")]

    compas_df['length_of_stay'] = (pd.to_datetime(compas_df['c_jail_out']) - pd.to_datetime(
        compas_df['c_jail_in'])).dt.days
    X = compas_df[['age', 'two_year_recid', 'c_charge_degree', 'race', 'sex', 'priors_count',
                   'length_of_stay']]

    # if person has high score give them the _negative_ model outcome
    y = np.array([NEGATIVE_OUTCOME if score == 'High' else POSITIVE_OUTCOME for score in
                  compas_df['score_text']])
    sens = X.pop('race')

    # assign African-American as the protected class
    X = pd.get_dummies(X)
    sensitive_attr = np.array(pd.get_dummies(sens).pop('African-American'))
    X['race'] = sensitive_attr

    # make sure everything is lining up
    assert all((sens == 'African-American') == (X['race'] == PROTECTED_CLASS))
    cols = [col for col in X]

    if encode:
        categorical_features = ["two_year_recid", "c_charge_degree_F", "c_charge_degree_M",
                                "sex_Female", "sex_Male", "race"]
        # Convert categorical features to ordinal
        dict_le = {}
        for cat_col in categorical_features:
            le = LabelEncoder()
            X[cat_col] = le.fit_transform(X[cat_col])
            dict_le[cat_col] = le

        data = {
            'data': X,
            'target': y,
            'feature_names': cols,
            'categorical_features': categorical_features,
            'categorical_encoders': dict_le
        }
    else:
        data = {
            'data': X,
            'target': y,
            'cols': cols,
            'feature_names': cols
        }

    return data


def load_german_credit_dataset():
    path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                     'data/german_credit_data.csv'))
    df = pd.read_csv(path)
    df = df.fillna('None')
    target_col = 'Risk'
    df[df[target_col] == 'good'][target_col] = 1
    df[df[target_col] == 'bad'][target_col] = 0

    numerical_features = ['Age', 'Credit amount', 'Duration']
    categorical_features = ['Sex', 'Job', 'Housing', 'Saving accounts',
                            'Checking account', 'Purpose']
    feature_names = list(df.columns)[:-1]

    # Convert categorical features to ordinal
    dict_le = {}
    for cat_col in categorical_features:
        le = LabelEncoder()
        df[cat_col] = le.fit_transform(df[cat_col])
        dict_le[cat_col] = le

    X, y = df[df.columns[:-1]].to_numpy(), df[target_col].values

    data = {
        'data': X,
        'target': y,
        'feature_names': feature_names,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'categorical_encoders': dict_le
    }

    return data


def get_and_preprocess_cc():
    """"Handle processing of Communities and Crime.  We exclude rows with missing values and predict
    if the violent crime is in the 50th percentile.
    Parameters
    ----------
    params : Params
    Returns:
    ----------
    Pandas data frame X of processed data, np.ndarray y, and list of column names
    """
    PROTECTED_CLASS = 1
    UNPROTECTED_CLASS = 0
    POSITIVE_OUTCOME = 1
    NEGATIVE_OUTCOME = 0

    X = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                 "data/communities_and_crime_new_version.csv"), index_col=0)

    # everything over 50th percentil gets negative outcome (lots of crime is bad)
    high_violent_crimes_threshold = 50
    y_col = 'ViolentCrimesPerPop numeric'

    X = X[X[y_col] != "?"]
    X[y_col] = X[y_col].values.astype('float32')

    # just dump all x's that have missing values
    cols_with_missing_values = []
    for col in X:
        if len(np.where(X[col].values == '?')[0]) >= 1:
            cols_with_missing_values.append(col)

    y = X[y_col]
    y_cutoff = np.percentile(y, high_violent_crimes_threshold)
    X = X.drop(cols_with_missing_values + ['communityname string', 'fold numeric', 'county numeric',
                                           'community numeric', 'state numeric'] + [y_col], axis=1)

    # setup ys
    cols = [c for c in X]
    y = np.array([NEGATIVE_OUTCOME if val > y_cutoff else POSITIVE_OUTCOME for val in y])

    data = {
        'data': X,
        'target': y,
        'feature_names': cols
    }
    return data
