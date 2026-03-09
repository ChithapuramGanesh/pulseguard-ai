"""
PulseGuard AI — Flask Backend
Serves the hypertension risk prediction model via REST API.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ── Load model ────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "hypertension_model.pkl")

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded from {MODEL_PATH}")
else:
    print(f"[WARN] Model not found at {MODEL_PATH}. Train first via train_model.py")


# Encoding maps — must match train_model.py cat.codes encoding
SMOKING_MAP       = {"Never": 2, "Former": 1, "Current": 0}
ACTIVITY_MAP      = {"Low": 1, "Moderate": 2, "High": 0}
FAMILY_MAP        = {"No": 0, "Yes": 1}


def encode_input(data: dict) -> np.ndarray:
    """Convert raw form values to model-ready numeric array."""
    features = [
        float(data["age"]),
        float(data["bmi"]),
        float(data["cholesterol"]),
        float(data["systolic_bp"]),
        float(data["diastolic_bp"]),
        SMOKING_MAP.get(data["smoking_status"], 2),
        ACTIVITY_MAP.get(data["physical_activity_level"], 2),
        FAMILY_MAP.get(data["family_history"], 0),
    ]
    return np.array(features).reshape(1, -1)


# ── Routes ────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Model not loaded. Please train the model first."
        }), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload received."}), 400

        # Validate required fields
        required = [
            "age", "bmi", "cholesterol", "systolic_bp",
            "diastolic_bp", "smoking_status",
            "physical_activity_level", "family_history"
        ]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Encode & predict
        X = encode_input(data)
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])  # P(hypertension=1)

        # Risk tier
        if probability < 0.35:
            risk_tier = "Low"
            risk_color = "#22c55e"
        elif probability < 0.65:
            risk_tier = "Moderate"
            risk_color = "#f59e0b"
        else:
            risk_tier = "High"
            risk_color = "#ef4444"

        # Feature contributions (from Random Forest)
        importances = model.feature_importances_
        feature_names = [
            "Age", "BMI", "Cholesterol", "Systolic BP",
            "Diastolic BP", "Smoking", "Activity", "Family History"
        ]
        contributions = [
            {"feature": name, "importance": round(float(imp) * 100, 1)}
            for name, imp in sorted(
                zip(feature_names, importances),
                key=lambda x: x[1], reverse=True
            )
        ]

        return jsonify({
            "prediction": prediction,
            "probability": round(probability * 100, 1),
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "contributions": contributions,
            "input_summary": {
                "Age": data["age"],
                "BMI": data["bmi"],
                "Systolic BP": data["systolic_bp"],
                "Diastolic BP": data["diastolic_bp"],
            }
        })

    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 422
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)