###
# Code originally written by Harry Wang (https://github.com/harrywang/mini-ml/)
# It was modified for the purpose of teaching how to deploy a machine learning 
# model using Streamlit.
###

import streamlit as st
import pandas as pd
import joblib
import datetime

import hmac

def check_password() -> bool:
    """Returns True if the user has entered the correct password."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.sidebar.subheader("Login")
    password = st.sidebar.text_input("Password", type="password")

    # Read password from Streamlit secrets (recommended for deployment)
    correct_pw = st.secrets.get("APP_PASSWORD", "")

    if password:
        if hmac.compare_digest(password, correct_pw):
            st.session_state.authenticated = True
            st.sidebar.success("Logged in")
            return True
        else:
            st.sidebar.error("Incorrect password")
            return False

    st.sidebar.info("Enter the password to use the app.")
    return False

if not check_password():
    st.stop()  

# load your machine learning model
tree_clf = joblib.load('model_dt.pickle')

### Streamnlit app code starts here

st.title('Titanic Survival Prediction')
with st.expander('Show sample of Titanic data'):
    df = pd.read_csv('titanic.csv') # adjust filename if needed
    st.dataframe(df.head(20))

st.markdown('**Please provide passenger information**:')  # you can use markdown like this

# get inputs
with st.sidebar:
    st.header("Passenger details")

    with st.form("inputs"):
        sex = st.selectbox("Sex", ["female", "male"])

        dob = st.date_input(
            "Date of birth",
            min_value=datetime.date(1900, 1, 1),
            max_value=datetime.date.today(),
            value=datetime.date(2000, 1, 1),
            help="We calculate age automatically."
        )

        sib_sp = st.number_input(
            "Siblings / spouses aboard (SibSp)",
            min_value=0,
            max_value=8,
            value=0,
            step=1
        )

        pclass = st.selectbox(
            "Ticket class (Pclass)",
            [1, 2, 3],
            help="1 = 1st class, 2 = 2nd class, 3 = 3rd class"
        )

        fare = st.number_input(
            "Fare paid",
            min_value=0.0,
            max_value=600.0,
            value=50.0,
            step=1.0
        )

        submitted = st.form_submit_button("Predict")

# this is how to dynamically change text
prediction_state = st.markdown('calculating...')

### Now the inference part starts here

if submitted:
    today = datetime.date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    passenger = pd.DataFrame({
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sib_sp],
        "Pclass": [pclass],
        "Fare": [fare]
    })

    y_pred = tree_clf.predict(passenger)
    proba = tree_clf.predict_proba(passenger)

# Preparing the message to be displayed based on the prediction
    if y_pred[0] == 0:
        st.error("Prediction: Passenger would NOT survive.")
    else:
        st.success("Prediction: Passenger would survive.")

    st.markdown(f"Survival probability: **{proba[0][1]*100:.1f}%**")

