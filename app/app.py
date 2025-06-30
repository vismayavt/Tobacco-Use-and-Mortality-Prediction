from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and features
model = joblib.load("app/mortality_model.pkl")
feature_columns = joblib.load("app/feature_columns.pkl")

@app.route("/")
def home():
    return "Welcome to the Tobacco Mortality Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Ensure column order and types
        for col in feature_columns:
            if col not in df.columns:
                df[col] = ""  # or a default like "Unknown" for categories, 0 for numbers

        df = df[feature_columns]

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
