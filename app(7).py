import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Text Classification",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"


# ---------------------------------------------------------
# LOAD MODEL AND VECTORIZER
# ---------------------------------------------------------
@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.pkl was not found."
        )

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "vectorizer.pkl was not found."
        )

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)

    return vectorizer, model


try:
    vectorizer, model = load_model()
    model_ready = True

except Exception as error:
    model_ready = False
    load_error = str(error)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

if "input_text" not in st.session_state:
    st.session_state.input_text = ""


# ---------------------------------------------------------
# CLEAR FUNCTION
# ---------------------------------------------------------
def clear_text():

    st.session_state.input_text = ""
    st.session_state.prediction = None
    st.session_state.probabilities = None


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:

    st.title("🤖 AI Classifier")

    st.caption(
        "Created by: Alain Pierre Ombanglil"
    )

    st.divider()

    st.subheader("Model Status")

    if model_ready:
        st.success("Model loaded successfully")
    else:
        st.error("Model could not be loaded")

    st.divider()

    st.subheader("How to Use")

    st.write("""
    1. Enter or paste your text.

    2. Click **Analyze Text**.

    3. Review the predicted category.

    4. Check the confidence score if available.
    """)

    st.divider()

    if model_ready:

        st.subheader("Model Information")

        st.write(
            "Model:",
            type(model).__name__
        )

        st.write(
            "Vectorizer:",
            type(vectorizer).__name__
        )

        if hasattr(model, "classes_"):

            st.write(
                "Number of classes:",
                len(model.classes_)
            )


# ---------------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------------
st.title("🤖 AI Text Classification")

st.subheader(
    "Machine Learning Text Classification Application"
)

st.caption(
    "Created by: Alain Pierre Ombanglil"
)

st.write(
    "Enter text below and the trained machine learning model "
    "will automatically predict its category."
)

st.divider()


# ---------------------------------------------------------
# CHECK MODEL
# ---------------------------------------------------------
if not model_ready:

    st.error(
        "The application could not load the model files."
    )

    st.warning(
        "Make sure model.pkl and vectorizer.pkl "
        "are in the same folder as app.py."
    )

    with st.expander("Technical Details"):
        st.code(load_error)

    st.stop()


# ---------------------------------------------------------
# MAIN COLUMNS
# ---------------------------------------------------------
left_column, right_column = st.columns(
 
