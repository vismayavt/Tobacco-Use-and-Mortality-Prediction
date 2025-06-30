# Tobacco Use and Mortality Prediction

This project predicts mortality risk based on hospital admissions data related to tobacco use in England (2004–2015).

## 📁 Project Structure
- `data/` — raw CSV data file
- `notebooks/` — EDA and modeling notebook
- `app/` — Flask API and saved model files
- `shap/` — visual explanation plots

## ⚙️ Technologies
- Python, Pandas, Scikit-learn
- Logistic Regression, Random Forest
- SHAP for interpretability
- Flask for deployment

## 🔮 Model
- Random Forest Classifier
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

## 🚀 Run the Flask API
```bash
python app.py
streamlit web_app.py
```


## 📊 SHAP & ROC Curve
Plots saved to `shap/`
