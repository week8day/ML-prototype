import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from PIL import Image
from pickle import dump, load


# age gpoup labels
age_group = {
    'от 21 до 34': 'b',
    'от 35 до 49': 'c',
    'от 50 до 64': 'd',
    'от 65': 'e'
}


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    logo_image = Image.open('data/images/logo.png')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Credit score",
        page_icon=logo_image,

    )

    st.image(logo_image)

    st.write(
        """
        # Кредитный скоринг
        прогнозирование просрочки возврата кредита
        """
    )


def write_user_data(df):
    st.write("## Данные клиента")
    st.write(df)


def write_prediction(prediction, prediction_proba):
    st.write("## Скоринг")
    st.write(prediction)

    st.write("## Вероятность невозврата")
    st.write(prediction_proba)


def preprocess_data(df: pd.DataFrame, test=True):
    # features values limitation
    limits = {
        'BalanceRate': 2,
        'Late30': 10,
        'Late60': 10,
        'Late90': 10,
        'MonthlyIncome': 30000,
        'OpenCredits': 40,
        'DebtRatio': 3
    }
    for col in limits:
        df.loc[df[col] > limits[col], col] = limits[col]

    # GroupAge encoding
#    for age_index in age_group:
#        if df['GroupAge'][:1] == age_index:
#            df = pd.concat([df, pd.DataFrame([1], coumns=['GroupAge_' + age_index])], axis=1)
#        else:
#            df = pd.concat([df, pd.DataFrame([0], coumns=['GroupAge_' + age_index])], axis=1)

    df['GroupAge_b'] = 0
    df['GroupAge_c'] = 0
    df['GroupAge_d'] = 0
    df['GroupAge_e'] = 1

    df.drop('GroupAge', axis=1, inplace=True)

    path = "data/model/scaler.sav"
    with open(path, "rb") as file:
        scaler = load(file)

    df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return df


def load_model_and_predict(df, path="data/model/clf_rfc.sav"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]

    prediction_proba = model.predict_proba(df)[0]

    encode_prediction_proba = {
        0: "Вероятность неплатежа",
        1: "Вероятность неплатежа"
    }

    encode_prediction = {
        0: "Платежи будут осуществляться вовремя",
        1: "Ожидается просрочка платежа"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


def process_side_bar_inputs():
    st.sidebar.header('Параметры клиента банка')
    user_input_df = sidebar_input_features()

    write_user_data(user_input_df)

    preprocessed_X_df = preprocess_data(user_input_df)

    user_X_df = preprocessed_X_df[:1]

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)


def sidebar_input_features():
    GroupAge = st.sidebar.selectbox("Возрастная группа", ("от 21 до 34", "от 35 до 49", "от 50 до 64", "от 65"))

    Dependents = st.sidebar.slider("Количество иждивенцев на попечении", min_value=0, max_value=20, value=0, step=1)

    MonthlyIncome = st.sidebar.slider("Ежемесячный доход, тыс. долларов", min_value=0, max_value=10000, value=5000, step=100)

    MonthlyЕxpense = st.sidebar.slider("Ежемесячный расход, тыс. долларов",
        min_value=0, max_value=30000, value=5000, step=100)

    OpenCredits = st.sidebar.slider("Количество активных кредитов", min_value=0, max_value=40, value=0, step=1)

    Balance = st.sidebar.slider("Общий баланс по кредитам, тыс. долларов", min_value=0, max_value=10000000, value=0, step=1000)

    Loans = st.sidebar.slider("Сумма кредитных лимитов, тыс. долларов",
        min_value=0, max_value=20000000, value=0, step=1000)

    Late30 = st.sidebar.slider("Количество просрочек 30-59 дней (за 2 года)", min_value=0, max_value=10, value=0, step=1)

    Late60 = st.sidebar.slider("Количество просрочек 60-89 дней (за 2 года)", min_value=0, max_value=10, value=0, step=1)

    Late90 = st.sidebar.slider("Количество просрочек более 90 дней (за 2 года)", min_value=0, max_value=10, value=0, step=1)

    data = {
        "BalanceRate": Balance / Loans if Loans > 0 else 0,
        "Late30": Late30,
        "DebtRatio": MonthlyЕxpense / MonthlyIncome if MonthlyIncome > 0 else 0,
        "MonthlyIncome": MonthlyIncome,
        "OpenCredits": OpenCredits,
        "Late90": Late90,
        "Late60": Late60,
        "Dependents": Dependents,
        "GroupAge": age_group[GroupAge]
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()
