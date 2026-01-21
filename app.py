import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# ---------------- LOAD MODEL ----------------
with open("xgb_credit_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- LOAD FEATURE NAMES ----------------
with open("columns.json", "r") as f:
    feature_names = json.load(f)

st.title("Credit Risk Prediction App")
st.write("Upload a CSV file or enter feature values manually")

# ---------------- CSV UPLOAD ----------------
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Input data:")
    st.write(data)

    # Safety check
    missing_cols = set(feature_names) - set(data.columns)
    extra_cols = set(data.columns) - set(feature_names)

    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
    elif extra_cols:
        st.error(f"Unexpected columns: {extra_cols}")
    else:
        data = data[feature_names]  # enforce correct order
        preds = model.predict(data)
        probs = model.predict_proba(data)[:, 1]

        st.write("Predictions:")
        st.write(preds)

        st.write("Default Probabilities:")
        st.write(probs)

# ---------------- MANUAL INPUT ----------------
st.write("### Or enter features manually")

input_data = {}

for feature in feature_names:
    input_data[feature] = st.number_input(feature, value=0.0)

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.success(f"Prediction: {pred}")
    st.write(f"Default Probability: **{prob:.4f}**")
