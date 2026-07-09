"""Shared request processing pipeline used by Streamlit UI and the local webhook API."""

from __future__ import annotations

from typing import Any, Dict

from .classifier import classify_request
from .logger import insert_log
from .workflows import execute_workflow


def process_incoming_request(
    request_text: str,
    requester_name: str = "Webhook User",
    requester_email: str = "",
    account_id: str = "",
    channel: str = "webhook",
) -> Dict[str, Any]:
    """
    Classify → remediate → audit log.
    Returns a single structured ops case payload suitable for UI or webhook response.
    """
    if not str(request_text or "").strip():
        raise ValueError("request_text is required")

    classification = classify_request(request_text)
    workflow = execute_workflow(request_text, requester_name, requester_email, classification)
    audit = insert_log(
        requester_name=requester_name,
        requester_email=requester_email,
        request_text=request_text,
        classification=classification,
        workflow=workflow,
        account_id=account_id,
        channel=channel,
    )

    return {
        "ok": True,
        "case": {
            "request_id": workflow.request_id,
            "timestamp": workflow.received_at,
            "channel": channel,
            "requester_name": requester_name,
            "requester_email": requester_email,
            "account_id": account_id,
            "request_text": request_text,
            "classification": {
                "request_type": classification.request_type,
                "urgency": classification.urgency,
                "confidence": classification.confidence,
                "sub_topic": classification.sub_topic,
                "rationale": classification.rationale,
                "processing_mode": classification.processing_mode,
            },
            "remediation": {
                "selected_branch": workflow.selected_branch,
                "status": workflow.status,
                "routing_destination": workflow.routing_destination or workflow.routed_team,
                "sla_or_follow_up": workflow.sla_or_follow_up,
                "actions_triggered": workflow.actions_triggered,
                "escalation_flag": workflow.escalation_flag,
                "human_review_required": workflow.human_review_required,
                "generated_response": workflow.generated_response,
                "routing_notification": workflow.routing_notification,
                "processing_mode": workflow.processing_mode,
            },
            "audit_id": audit.get("id"),
        },
    }
