# PulseGuard AI — Hypertension Risk Predictor

## Project Structure

```
PulseGuardAI/
├── dataset/
│   └── hypertension_dataset.csv   ← Your dataset here
├── model/                         ← Auto-created on training
│   └── hypertension_model.pkl
├── templates/
│   └── index.html                 ← Frontend UI
├── app.py                         ← Flask backend
├── train_model.py                 ← Model training script
└── requirements.txt
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
Make sure your CSV is in `dataset/hypertension_dataset.csv`, then:
```bash
python train_model.py
```

### 3. Start the Flask server
```bash
python app.py
```

### 4. Open in browser
Navigate to: **http://localhost:5000**

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET    | `/`   | Serves the frontend UI |
| POST   | `/api/predict` | Returns risk prediction |
| GET    | `/api/health`  | Health check |

### POST `/api/predict` — Request Body
```json
{
  "age": 45,
  "bmi": 27.5,
  "cholesterol": 210,
  "systolic_bp": 135,
  "diastolic_bp": 88,
  "smoking_status": "Never",
  "physical_activity_level": "Moderate",
  "family_history": "Yes"
}
```

### Response
```json
{
  "prediction": 1,
  "probability": 72.4,
  "risk_tier": "High",
  "risk_color": "#ef4444",
  "contributions": [
    { "feature": "Systolic BP", "importance": 28.3 },
    ...
  ]
}
```
