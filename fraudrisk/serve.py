"""uvicorn fraudrisk.serve:app   ->   POST /score scores one transaction with the trained champion."""
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import Field, create_model

from .data import VCOLS

# Same 30 inputs the model was trained on. Time = seconds since the reference transaction.
Txn = create_model("Txn", Time=(float, ...), Amount=(float, Field(..., ge=0)), **{c: (float, ...) for c in VCOLS})
app = FastAPI(title="fraud-risk", description="Calibrated fraud probability and LOW/MEDIUM/HIGH risk band.")
_scorer = None


def reset():
    global _scorer
    _scorer = None


def get_scorer():
    global _scorer
    if _scorer is None:
        path = os.getenv("MODEL_PATH", "artifacts/ulb/model.joblib")
        if not os.path.exists(path):
            raise HTTPException(503, f"model not found at {path}; run `python -m fraudrisk.train` first")
        _scorer = joblib.load(path)
    return _scorer


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _scorer is not None}


@app.post("/score")
def score(txn: Txn):
    s = get_scorer()
    p, band = s.score(pd.DataFrame([txn.model_dump()]))
    return {"fraud_probability": float(p[0]), "risk_band": str(band[0]),
            "thresholds": {"medium": s.thr_medium, "high": s.thr_high}, "model": s.model.kind}
