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
        "Created by: REEM FAYYAZ"
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
    [2, 1]
)


# ---------------------------------------------------------
# TEXT INPUT
# ---------------------------------------------------------
with left_column:

    st.header("📝 Enter Your Text")

    text = st.text_area(
        "Text to classify",
        key="input_text",
        placeholder=(
            "Type or paste your text here..."
        ),
        height=250
    )


    # TEXT STATISTICS
    word_count = (
        len(text.split())
        if text.strip()
        else 0
    )

    character_count = len(text)


    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            "Words",
            word_count
        )

    with metric2:
        st.metric(
            "Characters",
            character_count
        )


    st.write("")


    # BUTTONS
    button1, button2 = st.columns(
        [3, 1]
    )

    with button1:

        analyze_button = st.button(
            "🚀 Analyze Text",
            type="primary",
            use_container_width=True
        )

    with button2:

        st.button(
            "🗑️ Clear",
            use_container_width=True,
            on_click=clear_text
        )


# ---------------------------------------------------------
# INFORMATION PANEL
# ---------------------------------------------------------
with right_column:

    st.header("Application Features")

    st.info(
        "✨ Smart text classification using your "
        "trained machine learning model."
    )

    st.info(
        "⚡ Predictions are generated instantly."
    )

    st.info(
        "📊 Confidence scores are displayed when "
        "the model supports predict_proba()."
    )

    st.info(
        "🔒 The model runs directly from your "
        "deployed model files."
    )


# ---------------------------------------------------------
# RUN PREDICTION
# ---------------------------------------------------------
if analyze_button:

    cleaned_text = text.strip()

    if not cleaned_text:

        st.warning(
            "Please enter some text before analyzing."
        )

    elif len(cleaned_text) < 2:

        st.warning(
            "Please enter more text."
        )

    else:

        try:

            with st.spinner(
                "Analyzing text..."
            ):

                # Transform the text
                vectorized_text = vectorizer.transform(
                    [cleaned_text]
                )

                # Predict category
                prediction = model.predict(
                    vectorized_text
                )[0]

                st.session_state.prediction = prediction


                # Probability prediction
                probability_data = None

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probabilities = (
                        model.predict_proba(
                            vectorized_text
                        )[0]
                    )

                    if hasattr(
                        model,
                        "classes_"
                    ):

                        classes = model.classes_

                    else:

                        classes = range(
                            len(probabilities)
                        )


                    probability_data = pd.DataFrame(
                        {
                            "Class": classes,
                            "Probability": probabilities
                        }
                    )

                    probability_data = (
                        probability_data
                        .sort_values(
                            by="Probability",
                            ascending=False
                        )
                        .reset_index(drop=True)
                    )


                st.session_state.probabilities = (
                    probability_data
                )


        except Exception as error:

            st.error(
                "An error occurred while making the prediction."
            )

            with st.expander(
                "Technical Details"
            ):

                st.exception(error)


# ---------------------------------------------------------
# RESULTS SECTION
# ---------------------------------------------------------
if st.session_state.prediction is not None:

    st.divider()

    st.header("📊 Prediction Result")


    result_column1, result_column2 = st.columns(
        2
    )


    with result_column1:

        st.subheader(
            "Predicted Category"
        )

        st.success(
            str(
                st.session_state.prediction
            )
        )


    probability_df = (
        st.session_state.probabilities
    )


    # -----------------------------------------------------
    # CONFIDENCE SCORE
    # -----------------------------------------------------
    if (
        probability_df is not None
        and
        not probability_df.empty
    ):

        top_probability = float(
            probability_df.iloc[0][
                "Probability"
            ]
        )


        with result_column2:

            st.subheader(
                "Confidence"
            )

            st.metric(
                "Top Confidence Score",
                f"{top_probability:.2%}"
            )


        st.progress(
            top_probability
        )


        # -------------------------------------------------
        # CHART
        # -------------------------------------------------
        st.subheader(
            "📈 Prediction Probability"
        )


        chart_data = (
            probability_df
            .set_index("Class")
        )


        st.bar_chart(
            chart_data["Probability"]
        )


        # -------------------------------------------------
        # TABLE
        # -------------------------------------------------
        st.subheader(
            "📋 Detailed Results"
        )


        display_data = (
            probability_df.copy()
        )


        display_data[
            "Probability"
        ] = (
            display_data[
                "Probability"
            ]
            .apply(
                lambda value:
                f"{value:.2%}"
            )
        )


        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )


    else:

        with result_column2:

            st.info(
                "Confidence scores are not available "
                "for this model."
            )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()

st.caption(
    "AI Text Classification Application"
)

st.caption(
    "Created by: Alain Pierre Ombanglil"
)

st.caption(
    "Powered by Python, Streamlit and Scikit-learn"
)
