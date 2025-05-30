import joblib
import json
import pandas as pd
from fastapi import FastAPI

# Load model and feature list at startup
clf = joblib.load("fx_signal_xgb.joblib")
with open("feature_cols.json", "r") as f:
    feature_cols = json.load(f)

app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    """
    Expects a dictionary with all the feature columns as keys and their values for a single prediction.
    Example:
    {
        "Open": 1.1,
        "High": 1.2,
        "Low": 1.0,
        ...
    }
    """
    # Ensure input has all features, fill missing with 0
    X = pd.DataFrame([{col: data.get(col, 0) for col in feature_cols}])
    pred = clf.predict(X)[0]
    prob = clf.predict_proba(X)[0]
    # Map output to readable labels
    label_map = {0: "Sell", 1: "Neutral", 2: "Buy"}
    return {
        "signal": label_map.get(pred, int(pred)),
        "probabilities": {
            "Sell": float(prob[0]),
            "Neutral": float(prob[1]),
            "Buy": float(prob[2])
        }
    }
