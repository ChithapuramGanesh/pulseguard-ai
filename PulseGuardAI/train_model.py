# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("dataset/hypertension_dataset.csv")

print("Dataset shape:", data.shape)

# -----------------------
# Data Cleaning
# -----------------------

# Remove duplicate rows
data = data.drop_duplicates()

# Fill missing values
data = data.fillna(method="ffill")

# -----------------------
# Encode categorical columns
# -----------------------

categorical_cols = [
    "Country",
    "Smoking_Status",
    "Alcohol_Intake",
    "Physical_Activity_Level",
    "Family_History",
    "Diabetes",
    "Gender",
    "Education_Level",
    "Employment_Status",
    "Hypertension"
]

for col in categorical_cols:
    data[col] = data[col].astype("category").cat.codes

# -----------------------
# Feature Selection
# -----------------------

features = [
    "Age",
    "BMI",
    "Cholesterol",
    "Systolic_BP",
    "Diastolic_BP",
    "Smoking_Status",
    "Physical_Activity_Level",
    "Family_History"
]

X = data[features]
y = data["Hypertension"]

print("Feature shape:", X.shape)

# -----------------------
# Train Test Split
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Model Training
# -----------------------

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------
# Model Evaluation
# -----------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# -----------------------
# Save Model
# -----------------------

joblib.dump(model, "model/hypertension_model.pkl")

print("Model saved successfully")