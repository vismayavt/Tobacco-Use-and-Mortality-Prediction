import streamlit as st
import joblib
import pandas as pd
import requests

# Load the dataset to get unique values for dropdowns
df = pd.read_csv(r"C:\Users\visma\OneDrive\Desktop\Tobacco_Mortality\data\admissions.csv")

# Clean the data
df = df.dropna(subset=["Value"])
df["Sex"] = df["Sex"].fillna("Unknown")
df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})")
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)

# Unique dropdown values
years = sorted(df["Year"].unique())
codes = sorted(df["ICD10 Code"].dropna().unique())
diagnoses = sorted(df["ICD10 Diagnosis"].dropna().unique())
types = sorted(df["Diagnosis Type"].dropna().unique())
metrics = sorted(df["Metric"].dropna().unique())
sexes = sorted(df["Sex"].unique())

# Streamlit app UI
st.title("🧠 Tobacco Mortality Predictor")
st.markdown("Enter the details below to predict the likelihood of mortality based on hospital admission records.")

# Dropdown input fields
year = st.selectbox("Select Year", years)
code = st.selectbox("ICD10 Code", codes)
diagnosis = st.selectbox("ICD10 Diagnosis", diagnoses)
diagnosis_type = st.selectbox("Diagnosis Type", types)
metric = st.selectbox("Metric", metrics)
sex = st.selectbox("Sex", sexes)

# Predict button
if st.button("Predict"):
    input_data = {
        "Year": int(year),
        "ICD10 Code": code,
        "ICD10 Diagnosis": diagnosis,
        "Diagnosis Type": diagnosis_type,
        "Metric": metric,
        "Sex": sex
    }

    st.markdown("📤 Sending the following data to the API:")
    st.json(input_data)

    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Predicted Mortality Class: {result['prediction']}")
            st.metric("📈 Predicted Probability", round(result['probability'], 4))
        else:
            st.error(f"❌ API returned HTTP {response.status_code}")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
