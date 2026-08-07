import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

st.set_page_config(page_title="Text Classification", page_icon="🤖")

st.title("🤖 Text Classification App")
text = st.text_area("Enter Text")

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        st.success(f"Prediction: {pred}")
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vec)[0]
            st.subheader("Prediction Probability")
            for cls, p in zip(model.classes_, probs):
                st.write(f"{cls}: {p:.2%}")
