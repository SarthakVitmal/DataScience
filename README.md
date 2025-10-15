# GPU Revenue Predictor (FastAPI + Docker)

This repository contains a simple linear regression model and a FastAPI app to serve predictions.

Files added:
- `app/main.py` - FastAPI application exposing POST `/predict`.
- `requirements.txt` - Python dependencies.
- `Dockerfile` - Container image to run the API with Uvicorn.

Quick start (Windows PowerShell):

1. Train and save the model (creates `model.pkl`):

```powershell
python training.py
```

2. Run locally without Docker:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

3. Build Docker image and run container:

```powershell
docker build -t gpu-rev-predictor:latest .
docker run -p 8000:8000 gpu-rev-predictor:latest
```

4. Test the endpoint (PowerShell curl):

```powershell
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"market_share": 12.5}'
```

Notes:
- If you already have `model.pkl` in the repo root, the API will load it on startup. If not, run `training.py` to create it.
- The Dockerfile copies the entire repo; `.dockerignore` excludes large files.
