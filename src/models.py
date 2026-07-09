from dataclasses import dataclass, asdict
from typing import Dict, List, Any

@dataclass
class ClassificationResult:
    request_type: str
    urgency: str
    confidence: float
    sub_topic: str
    rationale: str
    extracted_details: Dict[str, Any]
    human_review_required: bool = False
    # Set by classifier to let the UI show which engine produced the result.
    processing_mode: str = "fallback"

@dataclass
class WorkflowResult:
    status: str
    routed_team: str
    sla_or_follow_up: str
    actions_triggered: List[str]
    generated_response: str
    routing_notification: str
    action_summary: str
    human_review_required: bool
    # Canonical operational fields (filled by workflow engine).
    selected_branch: str = ""
    routing_destination: str = ""
    escalation_flag: bool = False
    request_id: str = ""
    received_at: str = ""
    reasoning_summary: str = ""
    processing_mode: str = "fallback"

    def to_dict(self):
        return asdict(self)
