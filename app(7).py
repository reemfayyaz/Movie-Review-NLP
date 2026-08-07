import streamlit as st
import joblib

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Text Classification App",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# App styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 850px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .app-header {
            text-align: center;
            padding: 1.8rem 1rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
            margin-bottom: 1.8rem;
        }

        .app-header h1 {
            margin-bottom: 0.4rem;
        }

        .app-header p {
            margin: 0;
            opacity: 0.9;
            font-size: 1rem;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.65rem;
        }

        .result-box {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    """Load the trained vectorizer and classification model."""
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("model.pkl")
    return vectorizer, model


try:
    vectorizer, model = load_resources()
except FileNotFoundError as error:
    st.error(
        "Required model files were not found. "
        "Make sure `vectorizer.pkl` and `model.pkl` are in the same folder as `app.py`."
    )
    st.exception(error)
    st.stop()
except Exception as error:
    st.error("The model could not be loaded.")
    st.exception(error)
    st.stop()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🤖 Text Classification</h1>
        <p>Enter text below and let the trained machine-learning model classify it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# User input
# ---------------------------------------------------------
with st.form("prediction_form"):
    text = st.text_area(
        "Enter text",
        placeholder="Type or paste the text you want to classify...",
        height=180,
        help="Enter the text that you want the model to analyze.",
    )

    predict_button = st.form_submit_button("🔍 Predict")

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if predict_button:
    cleaned_text = text.strip()

    if not cleaned_text:
        st.warning("Please enter some text before making a prediction.")
    else:
        try:
            transformed_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(transformed_text)[0]

            st.subheader("Prediction Result")
            st.success(f"Predicted Class: **{prediction}**")

            # Display probabilities when supported by the model
            if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
                probabilities = model.predict_proba(transformed_text)[0]

                probability_data = sorted(
                    zip(model.classes_, probabilities),
                    key=lambda item: item[1],
                    reverse=True,
                )

                st.subheader("Prediction Confidence")

                for class_name, probability in probability_data:
                    st.write(f"**{class_name}** — {probability:.2%}")
                    st.progress(float(probability))

        except Exception as error:
            st.error("An error occurred while making the prediction.")
            st.exception(error)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "This application uses a trained text vectorizer and machine-learning "
        "classification model to predict the category of user-provided text."
    )

    st.divider()

    st.caption("Built with Python, Streamlit, and scikit-learn.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption("🤖 Text Classification App")
