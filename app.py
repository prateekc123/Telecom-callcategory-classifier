import streamlit as st
import sys
import os

sys.path.append(os.path.abspath("src"))

from predict import predict

st.set_page_config(page_title="Telecom Intent Classifier", layout="centered")

st.title("📞 Telecom Call Intent Classifier")
st.write("Classify customer queries into Billing, Repair, Sales, Retention, etc.")

user_input = st.text_area("Enter customer query:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a query")
    else:
        label, confidence = predict(user_input)

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.2f}")