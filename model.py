import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTENC

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from pickle import dump

RANDOM_STATE = 42
MODEL_PATH = 'data/model/'
DATA_URL = 'https://raw.githubusercontent.com/evgpat/edu_stepik_from_idea_to_mvp/main/datasets/credit_scoring.csv'
TARGET_COL = 'SeriousLate'
CATEGORY_COLS = ['GroupAge']


def open_data(path=DATA_URL):
    df = pd.read_csv(path)

    return df


def rename_columns(df: pd.DataFrame):
    df.rename(columns={
        'SeriousDlqin2yrs': 'SeriousLate',
        'age': 'Age',
        'RevolvingUtilizationOfUnsecuredLines': 'BalanceRate',
        'NumberOfTime30-59DaysPastDueNotWorse': 'Late30',
        'NumberOfTime60-89DaysPastDueNotWorse': 'Late60',
        'NumberOfTimes90DaysLate': 'Late90',
        'NumberOfOpenCreditLinesAndLoans': 'OpenCredits',
        'NumberOfDependents': 'Dependents',
        'NumberRealEstateLoansOrLines': 'Loans',
        'RealEstateLoansOrLines': 'LoansCode'
    }, inplace=True)

    return df


def preprocess_data(df: pd.DataFrame):
    df[CATEGORY_COLS] = df[CATEGORY_COLS].astype('category')

    df.drop(df[df[CATEGORY_COLS[0]] == 'a'].index, inplace=True)

    limits = {
        'BalanceRate': 2,
        'Late30': 10,
        'Late60': 10,
        'Late90': 10,
        'MonthlyIncome': 30000,
        'OpenCredits': 40,
        'DebtRatio': 3
    }

    for key in limits:
        df.drop(df[df[key] > limits[key]].index, inplace=True)

    df.drop(columns=['Age'], inplace=True)
    df.drop(columns=['LoansCode'], inplace=True)

    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].mean())
    df['Dependents'] = df['Dependents'].fillna(float(df['Dependents'].mode()))

    return df


def data_balancing(df: pd.DataFrame, target_col=TARGET_COL):
    over_sampler = SMOTENC(random_state=RANDOM_STATE, categorical_features=[9])
    df_sampled, _ = over_sampler.fit_resample(df, df[TARGET_COL])

    return df_sampled


def split_data(df: pd.DataFrame, target_col=TARGET_COL):
    y = df[target_col]
    X = df.drop(target_col, axis=1)

    return X, y


def encode_categories(X):
    X = pd.get_dummies(X, columns=CATEGORY_COLS, drop_first=True)

    return X


def data_scaling(X):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return scaler, X_scaled


def fit_and_save_model(X, y, scaler, path=MODEL_PATH):
    clf = RandomForestClassifier(n_estimators=1000, max_depth=4, random_state=RANDOM_STATE)
    clf.fit(X, y)

    model_filename = path + 'clf.sav'
    dump(clf, open(model_filename, 'wb'))

    scaler_filename = path + 'scaler.sav'
    dump(scaler, open(scaler_filename, 'wb'))

    test_prediction = clf.predict(X)
    f1_score = metrics.f1_score(test_prediction, y)

    print(f"Model F1-score is {round(f1_score, 4)}")

    print(f"Model was saved to {path}")


if __name__ == "__main__":
    df = open_data()

    df = rename_columns(df)

    df = preprocess_data(df)

    df = data_balancing(df)

    X, y = split_data(df)

    X = encode_categories(X)

    scaler, X = data_scaling(X)

    fit_and_save_model(X, y, scaler)
