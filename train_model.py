import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load dataset
df = pd.read_csv(r"C:\Users\visma\Downloads\admissions.csv")
df = df.dropna(subset=["Value"])
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value"])
df["Sex"] = df["Sex"].fillna("Unknown")
df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})")
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)

# ✅ Print for debug
print(f"✅ After cleaning, df shape: {df.shape}")

# Create target column
threshold = df["Value"].median()
df["mortality"] = (df["Value"] > threshold).astype(int)
df = df.drop("Value", axis=1)

# Features and target
X = df.drop("mortality", axis=1)
y = df["mortality"]

print(f"✅ Shape of X: {X.shape}")
print("✅ Sample rows of X:\n", X.head())
print("✅ Sample target values:\n", y.head())

# Define categorical and numerical features
categorical_features = ["ICD10 Code", "ICD10 Diagnosis", "Diagnosis Type", "Metric", "Sex"]
numerical_features = ["Year"]

# Preprocessing
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numerical_transformer, numerical_features),
    ]
)

# Model pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model_pipeline.fit(X_train, y_train)

# Predict
y_pred = model_pipeline.predict(X_test)
y_probs = model_pipeline.predict_proba(X_test)[:, 1]

# Evaluation
print("\n✅ Model Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1 Score:", round(f1_score(y_test, y_pred), 4))
print("ROC AUC:", round(roc_auc_score(y_test, y_probs), 4))

# Save model and features
os.makedirs("app", exist_ok=True)
joblib.dump(model_pipeline, "app/mortality_model.pkl")
joblib.dump(X.columns.tolist(), "app/feature_columns.pkl")

# Save one sample input with mortality = 0
if not df[df["mortality"] == 0].empty:
    sample_zero = df[df["mortality"] == 0].iloc[0].drop("mortality").to_dict()
    sample_zero = {k: (int(v) if isinstance(v, (int, float, np.integer)) else str(v)) for k, v in sample_zero.items()}
    with open("app/sample_zero_input.json", "w") as f:
        json.dump(sample_zero, f, indent=2)
    print("\n🎯 Sample input with mortality = 0 saved:")
    print(sample_zero)
else:
    print("⚠️ No sample with mortality = 0 found!")

# Also save one sample with mortality = 1
if not df[df["mortality"] == 1].empty:
    sample_one = df[df["mortality"] == 1].iloc[0].drop("mortality").to_dict()
    sample_one = {k: (int(v) if isinstance(v, (int, float, np.integer)) else str(v)) for k, v in sample_one.items()}
    with open("app/sample_one_input.json", "w") as f:
        json.dump(sample_one, f, indent=2)
    print("\n📌 Sample input with mortality = 1 saved:")
    print(sample_one)
