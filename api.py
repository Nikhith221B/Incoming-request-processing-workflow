"""
Local webhook / intake API for Incoming Request Processing Workflow.

Run (from project root, with venv active):
  uvicorn api:app --reload --port 8000

Then POST a JSON body to:
  http://127.0.0.1:8000/webhook/incoming
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline import process_incoming_request
from src.utils import load_project_env

load_project_env()

app = FastAPI(
    title="Incoming Request Processing Webhook",
    description="Local intake API that classifies requests and runs branch remediation.",
    version="1.0.0",
)


class IncomingRequest(BaseModel):
    request_text: str = Field(..., min_length=1, description="Raw customer/client message")
    requester_name: str = "Webhook User"
    requester_email: str = ""
    account_id: str = ""
    channel: str = "webhook"
    source: Optional[str] = Field(default=None, description="Optional source system label")


@app.get("/health")
def health():
    return {"status": "ok", "service": "incoming-request-webhook"}


@app.post("/webhook/incoming")
def webhook_incoming(payload: IncomingRequest):
    try:
        result = process_incoming_request(
            request_text=payload.request_text,
            requester_name=payload.requester_name,
            requester_email=payload.requester_email,
            account_id=payload.account_id,
            channel=payload.channel or "webhook",
        )
        if payload.source:
            result["case"]["source"] = payload.source
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Processing failed: {type(exc).__name__}") from exc
