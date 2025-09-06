import json
from typing import Dict, Any
import joblib
import pandas as pd
import os

_MODEL = None
_METADATA = {}

def load_model(path: str):
    global _MODEL, _METADATA
    _MODEL = joblib.load(path)
    meta_path = os.path.join(os.path.dirname(path), "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            _METADATA.update(json.load(f))
    return _MODEL

def predict_proba(model, payload: Dict[str, Any]) -> float:
    df = pd.DataFrame([payload])
    # model is a sklearn pipeline with transform + predict_proba
    proba = float(model.predict_proba(df)[0, 1])
    return proba

def explain_top(model, payload: Dict[str, Any]) -> Dict[str, float]:
    """Placeholder: return empty dict to keep inference light.
    You can wire SHAP here later (TreeExplainer for XGBoost/LightGBM)."""
    return {}
