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
from pathlib import Path

# -----------------------------
# Page config (must be first UI call)
# -----------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
)

# -----------------------------
# Professional tech styling (subtle)
# -----------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Sidebar: soft tech gradient */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(14,165,233,0.11), rgba(99,102,241,0.07));
}
[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span{
  font-size: 0.95rem;
}

/* Headline */
h1 { letter-spacing: -0.02em; }

/* Card style */
.card {
  border: 1px solid rgba(148,163,184,0.25);
  border-radius: 16px;
  padding: 16px 16px 8px 16px;
  background: rgba(2,6,23,0.02);
}

/* Primary button: slightly nicer */
div.stButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button[kind="primary"]{
  border-radius: 12px;
  padding: 0.55rem 1rem;
}

/* Metric box */
[data-testid="stMetric"] {
  background: rgba(14,165,233,0.07);
  border: 1px solid rgba(14,165,233,0.18);
  padding: 12px;
  border-radius: 14px;
}

/* Tighten expander spacing a bit */
[data-testid="stExpander"] { border-radius: 14px; }
</style>
""",
    unsafe_allow_html=True,
)


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

# -----------------------------
# Load ML model
# -----------------------------
tree_clf = joblib.load("model_dt.pickle")


# -----------------------------
# Header
# -----------------------------
st.title("🚢 Titanic Survival Predictor")
st.caption("Enter passenger details to estimate survival probability.")
st.divider()

# -----------------------------
# Sidebar: inputs
# -----------------------------
with st.sidebar:
    st.markdown("### 🎛️ Controls")
    st.caption("Set passenger details, then click **Predict**.")
    st.divider()

    st.header("Passenger details")

    with st.form("inputs"):
        sex = st.selectbox("Sex", ["female", "male"])

        dob = st.date_input(
            "Date of birth",
            min_value=datetime.date(1900, 1, 1),
            max_value=datetime.date.today(),
            value=datetime.date(2000, 1, 1),
            help="We calculate age automatically.",
        )

        sib_sp = st.number_input(
            "Siblings / spouses aboard (SibSp)",
            min_value=0,
            max_value=8,
            value=0,
            step=1,
        )

        pclass = st.selectbox(
            "Ticket class (Pclass)",
            [1, 2, 3],
            help="1 = 1st class, 2 = 2nd class, 3 = 3rd class",
        )

        fare = st.number_input(
            "Fare paid",
            min_value=0.0,
            max_value=600.0,
            value=50.0,
            step=1.0,
        )

        submitted = st.form_submit_button("Predict", type="primary")


# -----------------------------
# Main layout
# -----------------------------
st.markdown("### Passenger information")
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.expander("Show sample of Titanic data"):
        df = pd.read_csv("titanic.csv")
        st.dataframe(df.head(20), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction")

    if not submitted:
        st.info("Fill in the details on the left and click **Predict**.")
    else:
        with st.spinner("Calculating survival probability..."):
            today = datetime.date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

            passenger = pd.DataFrame(
                {
                    "Sex": [sex],
                    "Age": [age],
                    "SibSp": [sib_sp],
                    "Pclass": [pclass],
                    "Fare": [fare],
                }
            )

            y_pred = int(tree_clf.predict(passenger)[0])
            proba = float(tree_clf.predict_proba(passenger)[0][1])

        st.metric("Survival probability", f"{proba:.1%}")
        st.progress(int(proba * 100))

        if proba >= 0.50:
            st.balloons()
            video_path = Path(__file__).resolve().parent / "assets" / "survival.mp4"
            st.video(str(video_path))

        if y_pred == 0:
            st.error("❌ Prediction: Passenger would **NOT** survive.")
        else:
            st.success("✅ Prediction: Passenger would **survive**.")

        # Optional: show computed age (nice touch)
        st.caption(f"Computed age from DOB: **{age}**")

    st.markdown("</div>", unsafe_allow_html=True)