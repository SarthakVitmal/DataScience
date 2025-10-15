from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(title="GPU Revenue Predictor")


class PredictRequest(BaseModel):
    market_share: float = Field(..., example=12.5, description="Market Share Gaming GPU (%)")


MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.pkl")


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}. Run training.py to create it.")
    return joblib.load(path)


@app.on_event("startup")
def startup_event():
    try:
        app.state.model = load_model(MODEL_PATH)
    except Exception as e:
        # Keep startup but make predict return a clear error
        app.state.model = None
        app.state.model_error = str(e)


@app.post("/predict")
def predict(req: PredictRequest):
    """Predict Annual Revenue (Billion USD) from Market Share Gaming GPU (%)"""
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=500, detail=getattr(app.state, "model_error", "Model not loaded"))

    x = np.array([[req.market_share]])
    pred = model.predict(x)
    # Ensure a native Python type for JSON
    return {"prediction": float(pred[0])}
