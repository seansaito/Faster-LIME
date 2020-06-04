import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import numpy as np


def get_and_preprocess_compas_data():
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

    data = {
        'data': X,
        'target': y,
        'cols': cols
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
