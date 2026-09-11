from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = joblib.load(ROOT / "models" / "churn_model.joblib")

def predict_customer(customer_dict):
    X = pd.DataFrame([customer_dict])
    probability = float(MODEL.predict_proba(X)[0, 1])

    if probability >= 0.70:
        risk = "High"
    elif probability >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "churn_probability": probability,
        "risk_level": risk
    }
