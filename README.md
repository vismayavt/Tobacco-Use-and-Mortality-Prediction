# Tobacco Mortality Prediction

This project predicts tobacco-related mortality using hospital admission data from England (2004–2015).It leverages machine learning models like Random Forest for classification.The application includes a Flask API backend and a Streamlit-based user interface.Model interpretability is handled using SHAP visualizations and ROC analysis.It helps analyze public health trends and supports data-driven decision making.

---

## 📂 Project Structure

- `data/` – Dataset (admission.csv)
- `app/` – Flask API and saved model files
- `shap/` – SHAP and ROC visualizations
- `web_app.py` – Streamlit web app
- `train_model.py` – Script for training the model
- `notebooks/` – Jupyter notebook (EDA and modeling)

---

## Technologies Used

- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression, Random Forest)
- SHAP for model explainability
- Flask (for API), Streamlit (for web app)

---

## 🚀 How to Run

1. **Train the model:**
   ```bash
   python train_model.py
2. **Start Flask API:**
   ```bash
   python app/app.py
3. **Launch Streamlit app:**   
   ```bash
   streamlit run web_app.py

---

