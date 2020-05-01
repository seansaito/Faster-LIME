import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_german_credit_dataset():
    path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/german_credit_data.csv'))
    print(path)
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
