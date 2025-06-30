from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained pipeline (StandardScaler + Model)
model = joblib.load('app/mortality_model.pkl')
scaler = joblib.load("app/scaler.pkl")  # if used

@app.route('/')
def home():
    return "Welcome to Tobacco Mortality Predictor API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received input:", data)

        # Make sure keys match model input
        expected_features = ['feature1', 'feature2', 'feature3']
        if not all(f in data for f in expected_features):
            return jsonify({'error': 'Missing one or more required features'})

        # Convert to array for prediction
        features = np.array([[data['feature1'], data['feature2'], data['feature3']]])
        prediction = model.predict(features)

        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)



