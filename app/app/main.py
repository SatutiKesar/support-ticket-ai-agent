"""
REST API for the support ticket AI system.

Endpoints:
  GET  /health      - health check + dataset stats
  POST /query        - natural language question answering (LLM -> SQL -> answer)
  GET  /anomalies    - anomaly detection report
  GET  /tickets      - simple filterable ticket listing (bonus, not required)

Run directly with:  uvicorn app.main:app --reload
"""
from typing import Optional
import math

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import anomalies as anomaly_engine
from app import config, data, nl_query

app = FastAPI(
    title="Support Ticket AI System",
    description="NL query + anomaly detection over a support ticket dataset.",
    version="1.0.0",
)

# CORS is wide open here for local/demo use (e.g. calling from the Streamlit UI
# or a tunnelled URL). Tighten CORS_ALLOW_ORIGINS in .env for any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {"df": None}


@app.on_event("startup")
def _startup() -> None:
    try:
        _state["df"] = data.initialize()
    except data.DataLoadError as exc:
        # Fail loudly at startup rather than serving a broken /health as "ok".
        raise RuntimeError(f"Failed to load ticket data: {exc}") from exc


def _get_df() -> pd.DataFrame:
    if _state["df"] is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded yet.")
    return _state["df"]


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    df = _state["df"]
    return {
        "status": "ok" if df is not None else "not_ready",
        "rows_loaded": int(len(df)) if df is not None else 0,
        "llm_provider": config.LLM_PROVIDER,
    }


@app.post("/query")
def query(req: QueryRequest):
    _get_df()  # ensures data is ready
    try:
        return nl_query.answer_question(req.question)
    except nl_query.NLQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface LLM/network errors cleanly
        raise HTTPException(status_code=502, detail=f"Query pipeline failed: {exc}") from exc


@app.get("/anomalies")
def get_anomalies():
    df = _get_df()
    try:
        return anomaly_engine.detect_anomalies(df)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {exc}") from exc


@app.get("/tickets")
def list_tickets(
    category: Optional[str] = None,
    priority: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
):
    df = _get_df()
    out = df
    if category:
        out = out[out["category"].str.lower() == category.lower()]
    if priority:
        out = out[out["priority"].str.lower() == priority.lower()]
    if status:
        out = out[out["status"].str.lower() == status.lower()]
    out = out.head(limit).copy()
    out["created_at"] = out["created_at"].astype(str)
    records = out.to_dict(orient="records")
    # NaN isn't valid JSON (unresolved tickets have no resolution_time_hrs /
    # customer_rating yet). Note: df.where(df.notnull(), None) does NOT work
    # here -- assigning None into a float64 column gets silently re-cast back
    # to NaN by pandas. Sanitizing the already-materialized records instead.
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None
    return {"count": int(len(records)), "tickets": records}
