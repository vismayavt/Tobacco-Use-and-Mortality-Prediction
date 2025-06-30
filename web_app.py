import streamlit as st
import requests

st.title("Tobacco Mortality Predictor")

# Input fields (rename based on your actual features)
feature1 = st.number_input("Feature 1: Smoking Rate", value=0.0)
feature2 = st.number_input("Feature 2: Tobacco Exposure (years)", value=0.0)
feature3 = st.number_input("Feature 3: Age Group Encoded", value=0.0)

if st.button("Predict"):
    data = {
        "feature1": feature1,
        "feature2": feature2,
        "feature3": feature3
    }

    st.write("Sending to API:", data)

    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=data)

        if response.status_code == 200:
            result = response.json()
            if 'prediction' in result:
                st.success(f"Predicted Mortality: {result['prediction']:.2f}")
            else:
                st.error(f"API Error: {result.get('error', 'Unknown error')}")
        else:
            st.error(f"HTTP Error {response.status_code}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to Flask API. Please ensure it's running.")
