"""Simulated inbox samples covering all remediation branches."""

from __future__ import annotations

from typing import Dict, List, Optional


def get_inbox_samples() -> List[Dict[str, str]]:
    """
    Return 10 realistic inbox items.
    Classification is intentionally NOT pre-labeled — the system must infer it.
    Expected branches (for demo validation, not UI):
      1 Complaint, 2 General Enquiry, 3 Service Request, 4 Escalation / Urgent,
      5 Service Request, 6 Complaint, 7 Needs Human Review (ambiguous),
      8 General Enquiry, 9 Escalation / Urgent, 10 Needs Human Review / unclear
    """
    return [
        {
            "inbox_id": "INB-001",
            "requester_name": "Amit Sharma",
            "requester_email": "amit@example.com",
            "account_id": "ACC-10021",
            "channel": "email",
            "subject": "Duplicate charge on last invoice",
            "request_text": (
                "I was charged twice for my last invoice and nobody has responded. "
                "This is unacceptable and I want this fixed today."
            ),
        },
        {
            "inbox_id": "INB-002",
            "requester_name": "Sara Khan",
            "requester_email": "sara@example.com",
            "account_id": "",
            "channel": "web_form",
            "subject": "Refund policy question",
            "request_text": (
                "Can you explain your refund policy and how long support usually takes to respond?"
            ),
        },
        {
            "inbox_id": "INB-003",
            "requester_name": "Priya Menon",
            "requester_email": "priya@example.com",
            "account_id": "ACC-88210",
            "channel": "email",
            "subject": "Activate account and update billing contact",
            "request_text": (
                "Please activate my new business account and update the billing contact to Priya Menon."
            ),
        },
        {
            "inbox_id": "INB-004",
            "requester_name": "Rahul Verma",
            "requester_email": "rahul@example.com",
            "account_id": "ACC-44102",
            "channel": "shared_inbox",
            "subject": "Launch day outage — need supervisor",
            "request_text": (
                "Our service is down during launch day. I need a supervisor immediately "
                "or we will escalate this legally."
            ),
        },
        {
            "inbox_id": "INB-005",
            "requester_name": "Dana Lee",
            "requester_email": "dana@example.com",
            "account_id": "ACC-33019",
            "channel": "email",
            "subject": "Seat cancellation and plan change",
            "request_text": (
                "I need to cancel one seat from our subscription and move the remaining seats "
                "to the annual plan."
            ),
        },
        {
            "inbox_id": "INB-006",
            "requester_name": "Maya Rao",
            "requester_email": "maya@example.com",
            "account_id": "ACC-21775",
            "channel": "email",
            "subject": "Ignored for three emails",
            "request_text": (
                "Your support team has ignored three emails and the issue is still unresolved. "
                "I am very disappointed."
            ),
        },
        {
            "inbox_id": "INB-007",
            "requester_name": "Omar Ali",
            "requester_email": "omar@example.com",
            "account_id": "",
            "channel": "web_form",
            "subject": "Charge / refund — unclear ask",
            "request_text": (
                "I was charged twice, but I also just want the refund policy and pricing details. "
                "Please advise what I should do next."
            ),
        },
        {
            "inbox_id": "INB-008",
            "requester_name": "Helen Cho",
            "requester_email": "helen@example.com",
            "account_id": "",
            "channel": "email",
            "subject": "SLA / response time enquiry",
            "request_text": (
                "What is your expected response time for general enquiries, and do refund reviews "
                "take longer than that?"
            ),
        },
        {
            "inbox_id": "INB-009",
            "requester_name": "Jason Brooks",
            "requester_email": "jason@example.com",
            "account_id": "ACC-90511",
            "channel": "shared_inbox",
            "subject": "Suspected fraud on account",
            "request_text": (
                "We detected possible fraud on account ACC-90511. Notify a supervisor immediately "
                "and pause automated changes until compliance reviews this."
            ),
        },
        {
            "inbox_id": "INB-010",
            "requester_name": "Unknown Sender",
            "requester_email": "unknown@example.com",
            "account_id": "",
            "channel": "shared_inbox",
            "subject": "Unclear message",
            "request_text": (
                "Need help with the thing we discussed earlier. Please fix ASAP somehow."
            ),
        },
    ]


def get_sample_label_map() -> Dict[str, str]:
    """Short labels for demo convenience only (UI previews). Not used for classification."""
    return {
        "INB-001": "Demo: billing complaint",
        "INB-002": "Demo: refund / response-time enquiry",
        "INB-003": "Demo: account activation request",
        "INB-004": "Demo: outage / legal escalation",
        "INB-005": "Demo: subscription seat change",
        "INB-006": "Demo: unresolved complaint",
        "INB-007": "Demo: ambiguous complaint vs enquiry",
        "INB-008": "Demo: policy / SLA enquiry",
        "INB-009": "Demo: fraud / compliance escalation",
        "INB-010": "Demo: unclear / low-info request",
    }


def get_sample_by_inbox_id(inbox_id: str) -> Optional[Dict[str, str]]:
    for item in get_inbox_samples():
        if item["inbox_id"] == inbox_id:
            return item
    return None
