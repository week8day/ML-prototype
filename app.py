import pandas as pd
import streamlit as st
from PIL import Image
from pickle import load

IMAGE_PATH = 'data/images/'
MODEL_PATH = 'data/model/'
INTERFACE_LANG = 'eng'

# interface lang
interface_lang = INTERFACE_LANG
interface_dict = {
    'app_title': {
        'eng': "Credit score",
        'rus': "Кредитный скоринг"
    },
    'app_slogan': {
        'eng': "Predicting delinquency on loans",
        'rus': "Прогнозирование просрочки платежей по кредитам"
    },
    'form_title': {
        'eng': "Client data",
        'rus': "Параметры клиента"
    },
    'probability': {
        'eng': "Probability of delinquency",
        'rus': "Вероятность просрочки"
    },
    'dataframe_title': {
        'eng': "Parameters",
        'rus': "Параметры"
    },
    'comments_title': {
        'eng': "Comments",
        'rus': "Комментарии"
    },
    'advices_title': {
        'eng': "Advices",
        'rus': "Совет заемщику"
    },
    'prediction_0_text': {
        'eng': "Payments will be made on time",
        'rus': "Платежи будут осуществляться вовремя"
    },
    'prediction_1_text': {
        'eng': "Late payment is expected",
        'rus': "Ожидается просрочка платежа"
    },
    'comments_30': {
        'eng': "There are overdue payments of more than 30 days.",
        'rus': "Есть просрочки платежей более 30 дней."
    },
    'comments_60': {
        'eng': "There are overdue payments of more than 60 days.",
        'rus': "Есть просрочки платежей более 60 дней."
    },
    'comments_90': {
        'eng': "There are overdue payments of more than 90 days.",
        'rus': "Есть просрочки платежей более 90 дней."
    },
    'advice_balance': {
        'eng': "It is desirable that the debts do not exceed the credit limits.",
        'rus': "Желательно, чтобы долги не превышали кредитные лимиты."
    },
    'advice_debt': {
        'eng': "It is desirable that expenses do not exceed income.",
        'rus': "Желательно, чтобы расходы не превышали доходы."
    },
    'form_input_age': {
        'eng': "Age",
        'rus': "Возрастная группа"
    },
    'form_input_dependents': {
        'eng': "Number of dependents",
        'rus': "Количество иждивенцев на попечении"
    },
    'form_input_income': {
        'eng': "Monthly income",
        'rus': "Ежемесячный доход, тыс. долларов"
    },
    'form_input_expense': {
        'eng': "Monthly expense",
        'rus': "Ежемесячный расход, тыс. долларов"
    },
    'form_input_credits': {
        'eng': "Number of open creditlines and loans",
        'rus': "Количество активных кредитов"
    },
    'form_input_balance': {
        'eng': "Total loan balance",
        'rus': "Общий баланс по кредитам, тыс. долларов"
    },
    'form_input_loans': {
        'eng': "Amount of credit limits",
        'rus': "Сумма кредитных лимитов, тыс. долларов"
    },
    'form_input_late30': {
        'eng': "Number of times 30-59 days late",
        'rus': "Количество просрочек 30-59 дней (за 2 года)"
    },
    'form_input_late60': {
        'eng': "Number of times 60-89 days late",
        'rus': "Количество просрочек 60-89 дней (за 2 года)"
    },
    'form_input_late90': {
        'eng': "Number of times 90 days late",
        'rus': "Количество просрочек более 90 дней (за 2 года)"
    }
}

# age group labels
age_group = {
    '21 - 34': 'b',
    '35 - 49': 'c',
    '50 - 64': 'd',
    '> 65': 'e'
}

# features values limits
limits = {
    'BalanceRate': 2,
    'Late30': 10,
    'Late60': 10,
    'Late90': 10,
    'MonthlyIncome': 30000,
    'OpenCredits': 40,
    'DebtRatio': 3
}


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    logo_image = Image.open(IMAGE_PATH + 'logo.png')
    icon_image = Image.open(IMAGE_PATH + 'icon.png')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title=interface_dict['app_title'][interface_lang],
        page_icon=icon_image
    )

    col1, col2 = st.columns((1, 6))
    col1.image(logo_image, width=120)
    col2.title(interface_dict['app_title'][interface_lang])

    st.text(interface_dict['app_slogan'][interface_lang])


def write_user_data(df):
    st.write("## " + interface_dict['dataframe_title'][interface_lang])
    st.write(df)


def write_prediction(prediction, prediction_text, prediction_proba, comments='', advices=''):
    pay_proba = round(prediction_proba * 100)
    not_pay_proba = pay_proba - 100

    scoring_title = "### " + interface_dict['probability'][interface_lang] + ": "
    if prediction == 0:
        scoring_title += ":green["
    else:
        scoring_title += ":red["
    scoring_title += str(pay_proba) + "%]"

    col1, col2 = st.columns((1, 1))

    proba_text = interface_dict['probability'][interface_lang] 
    if prediction == 0:
        progress_text = '' + prediction_text + ''
    else:
        progress_text = '' + prediction_text + ''
    
    col1.write(scoring_title)
    col2.progress(pay_proba, text=progress_text)

    col1, col2 = st.columns((1, 1))

    col1.write("### " + interface_dict['comments_title'][interface_lang])
    col2.write("### " + interface_dict['advices_title'][interface_lang])

    if comments != '':
        col1.text(comments)

    if advices != '':
        col2.text(advices)

def preprocess_data(df: pd.DataFrame, test=True):
    for col in limits:
        df.loc[df[col] > limits[col], col] = limits[col]

    for key in age_group:
        age_index = age_group[key]
        if df['GroupAge'].iloc[0] == age_index:
            df = pd.concat([df, pd.DataFrame([1], columns=['GroupAge_' + age_index])], axis=1)
        else:
            df = pd.concat([df, pd.DataFrame([0], columns=['GroupAge_' + age_index])], axis=1)

    df.drop('GroupAge', axis=1, inplace=True)

    path = MODEL_PATH + "scaler.sav"
    with open(path, "rb") as file:
        scaler = load(file)

    df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return df


def load_model_and_predict(df, path=MODEL_PATH + "clf.sav"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]

    prediction_proba = float(model.predict_proba(df)[0][1])

    return prediction, prediction_proba


def process_side_bar_inputs():
    st.sidebar.header(interface_dict['form_title'][interface_lang])
    user_input_df = sidebar_input_features()

    write_user_data(user_input_df)

    preprocessed_X_df = preprocess_data(user_input_df)

    user_X_df = preprocessed_X_df[:1]

    prediction, prediction_proba = load_model_and_predict(user_X_df)

    encode_prediction = {
        0: interface_dict['prediction_0_text'][interface_lang],
        1: interface_dict['prediction_1_text'][interface_lang]
    }

    prediction_text = encode_prediction[prediction]

    comments, advices = credit_advisor(user_input_df)

    write_prediction(prediction, prediction_text, prediction_proba, comments, advices)


def credit_advisor(user_input_df):
    comments, advices = [], []

    if user_input_df['Late90'][0] > 0:
        comments.append(interface_dict['comments_90'][interface_lang])

    if user_input_df['Late60'][0] > 0:
        comments.append(interface_dict['comments_60'][interface_lang])

    if user_input_df['Late30'][0] > 0:
        comments.append(interface_dict['comments_30'][interface_lang])

    if user_input_df['BalanceRate'][0] > 1:
        advices.append(interface_dict['advice_balance'][interface_lang])

    if user_input_df['DebtRatio'][0] > 1:
        advices.append(interface_dict['advice_debt'][interface_lang])

    return "\n".join(comments), "\n".join(advices)


def sidebar_input_features():
    GroupAge = st.sidebar.selectbox(
        interface_dict['form_input_age'][interface_lang],
        age_group.keys()
    )

    Dependents = st.sidebar.slider(
        interface_dict['form_input_dependents'][interface_lang],
        min_value=0, max_value=20, value=0, step=1
    )

    MonthlyIncome = st.sidebar.slider(
        interface_dict['form_input_income'][interface_lang],
        min_value=0, max_value=10000, value=5000, step=100
    )


    MonthlyЕxpense = st.sidebar.slider(
        interface_dict['form_input_expense'][interface_lang],
        min_value=0, max_value=30000, value=5000, step=100
    )

    OpenCredits = st.sidebar.slider(
        interface_dict['form_input_credits'][interface_lang],
        min_value=0, max_value=40, value=0, step=1
    )

    Balance = st.sidebar.slider(
        interface_dict['form_input_balance'][interface_lang],
        min_value=0, max_value=10000000, value=0, step=1000
    )

    Loans = st.sidebar.slider(
        interface_dict['form_input_loans'][interface_lang],
        min_value=0, max_value=20000000, value=0, step=1000
    )

    Late30 = st.sidebar.slider(
        interface_dict['form_input_late30'][interface_lang],
        min_value=0, max_value=10, value=0, step=1
    )

    Late60 = st.sidebar.slider(
        interface_dict['form_input_late60'][interface_lang],
        min_value=0, max_value=10, value=0, step=1
    )

    Late90 = st.sidebar.slider(
        interface_dict['form_input_late90'][interface_lang],
        min_value=0, max_value=10, value=0, step=1
    )

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
