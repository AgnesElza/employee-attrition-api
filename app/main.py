from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from src.schema import EmployeePayload, PredictionResponse
from src.service import load_model, predict_proba, explain_top

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import os

app = FastAPI(title="Attrition Risk API", version="0.1.0")
Instrumentator().instrument(app).expose(app)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
_model = None

@app.on_event("startup")
def _load():
    global _model
    if os.path.exists(MODEL_PATH):
        _model = load_model(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(_model)}

@app.get("/version")
def version():
    return {"model": "v1", "sha": os.getenv("GITHUB_SHA","local")}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: EmployeePayload):
    if _model is None:
        return {"probability": 0.0, "risk": 0, "top_features": {}}
    p = predict_proba(_model, payload.features)
    top = explain_top(_model, payload.features)
    return {"probability": p, "risk": int(p > 0.5), "top_features": top}

# serve docs/ as static & expose /monitor
if os.path.exists("docs"):
    app.mount("/static", StaticFiles(directory="docs"), name="static")

@app.get("/monitor", include_in_schema=False)
def monitor():
    path = "docs/drift_report.html"
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return {"detail": "drift report not found"}