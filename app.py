import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Where your artifacts live when deployed (change to Drive path if running in Colab)
ARTIFACTS_DIR = "artifacts"  # ensure model.joblib, scaler.joblib, features.csv are in this folder

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "model.joblib"))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    features = pd.read_csv(os.path.join(ARTIFACTS_DIR, "features.csv")).iloc[:,0].tolist()
    return model, scaler, features

model, scaler, feature_cols = load_artifacts()

st.title("💡 Customer Churn Predictor")
st.write("Adjust the inputs below and press Predict.")

# Basic inputs (subset of features)
tenure = st.number_input("Tenure (months)", min_value=0, max_value=200, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=5000.0, value=840.0)
contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
internet_service = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
online_security = st.selectbox("Online Security", ["No","Yes"])
tech_support = st.selectbox("Tech Support", ["No","Yes"])
payment_method = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])

def build_feature_row():
    # start zeros
    row = pd.Series(0, index=feature_cols, dtype=float)
    # numeric
    if 'tenure' in row.index:
        row['tenure'] = tenure
    if 'MonthlyCharges' in row.index:
        row['MonthlyCharges'] = monthly_charges
    if 'TotalCharges' in row.index:
        row['TotalCharges'] = total_charges

    # categorical one-hot keys (match your features.csv column names)
    # Contract
    if f"Contract_{contract}" in row.index:
        row[f"Contract_{contract}"] = 1
    # InternetService
    if f"InternetService_{internet_service}" in row.index:
        row[f"InternetService_{internet_service}"] = 1
    # OnlineSecurity
    if f"OnlineSecurity_{online_security}" in row.index:
        row[f"OnlineSecurity_{online_security}"] = 1
    # TechSupport
    if f"TechSupport_{tech_support}" in row.index:
        row[f"TechSupport_{tech_support}"] = 1
    # PaymentMethod
    pm = f"PaymentMethod_{payment_method}"
    if pm in row.index:
        row[pm] = 1

    return row.fillna(0).values.reshape(1, -1)

if st.button("Predict churn"):
    x = build_feature_row()
    try:
        x_scaled = scaler.transform(x)
        prob = model.predict_proba(x_scaled)[0,1]
        pred = model.predict(x_scaled)[0]
        st.metric("Churn probability", f"{prob:.2%}")
        if pred == 1:
            st.error("⚠️ This customer is likely to CHURN")
        else:
            st.success("✅ This customer is likely to STAY")
    except Exception as e:
        st.exception(e)
        st.write("Check that your artifacts (model, scaler, features.csv) match the features used in this app.")
