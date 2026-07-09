import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List, Tuple

from .models import ClassificationResult, WorkflowResult
from .utils import (
    generate_request_id,
    get_gemini_api_key,
    get_gemini_model,
    load_project_env,
    ollama_chat,
    resolve_llm_provider,
    set_last_llm_error,
    utc_now_iso,
)

load_project_env()

CONFIG_PATH = Path(__file__).resolve().parents[1] / "workflow_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

KB_PATH = Path(__file__).resolve().parents[1] / "data" / "knowledge_base.md"


def execute_workflow(request_text: str, requester_name: str, requester_email: str, classification: ClassificationResult) -> WorkflowResult:
    cfg = CONFIG[classification.request_type]
    request_id = generate_request_id()
    received_at = utc_now_iso()

    generated_response, response_mode = _generate_response(
        request_text=request_text,
        requester_name=requester_name,
        requester_email=requester_email,
        classification=classification,
        cfg=cfg,
    )

    routing_notification = _routing_notification(
        requester_name=requester_name,
        requester_email=requester_email,
        classification=classification,
        cfg=cfg,
        request_id=request_id,
        received_at=received_at,
    )
    action_summary = "; ".join(cfg["actions"])

    human_review_required = classification.human_review_required or classification.urgency == "Critical"
    escalation_flag = classification.request_type == "Escalation / Urgent" or classification.urgency == "Critical"

    return WorkflowResult(
        status=cfg["status"],
        routed_team=cfg["routing_team"],
        sla_or_follow_up=cfg["sla"],
        actions_triggered=cfg["actions"],
        generated_response=generated_response,
        routing_notification=routing_notification,
        action_summary=action_summary,
        human_review_required=human_review_required,
        selected_branch=classification.request_type,
        routing_destination=cfg["routing_team"],
        escalation_flag=escalation_flag,
        request_id=request_id,
        received_at=received_at,
        reasoning_summary=classification.rationale,
        processing_mode=response_mode,
    )


def _generate_response(
    request_text: str,
    requester_name: str,
    requester_email: str,
    classification: ClassificationResult,
    cfg: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Returns (generated_response, mode).
    Mode is "llm" only if we successfully call the LLM; otherwise we use deterministic fallback.
    """
    provider = resolve_llm_provider()
    if provider == "gemini":
        try:
            return _generate_response_with_gemini(
                request_text=request_text,
                requester_name=requester_name,
                requester_email=requester_email,
                classification=classification,
                cfg=cfg,
            ), "llm"
        except Exception as exc:
            set_last_llm_error(exc, provider="gemini")
    elif provider == "ollama":
        try:
            return _generate_response_with_ollama(
                request_text=request_text,
                requester_name=requester_name,
                requester_email=requester_email,
                classification=classification,
                cfg=cfg,
            ), "llm"
        except Exception as exc:
            set_last_llm_error(exc, provider="ollama")

    return _generate_response_deterministic(request_text, requester_name, classification, cfg), "fallback"


def _draft_response_prompts(
    request_text: str,
    requester_name: str,
    requester_email: str,
    classification: ClassificationResult,
    cfg: Dict[str, Any],
) -> Tuple[str, str]:
    name = requester_name or "there"
    system_prompt = (
        "You are an operations support assistant. Draft a clear, empathetic customer response. "
        "Do not mention internal workflow steps or audit logs. Keep it under 200 words. "
        "End with a short sign-off."
    )
    user_prompt = (
        f"Requester: {name} <{requester_email}>\n"
        f"Request type: {classification.request_type}\n"
        f"Urgency: {classification.urgency}\n"
        f"Confidence: {classification.confidence:.2f}\n"
        f"Sub-topic: {classification.sub_topic}\n"
        f"Rationale: {classification.rationale}\n\n"
        f"Incoming request:\n{request_text}\n\n"
        f"Routing destination: {cfg['routing_team']}\n"
        f"SLA / follow-up: {cfg['sla']}\n"
        "Generate the customer-facing draft response only."
    )
    return system_prompt, user_prompt


def _generate_response_with_gemini(
    request_text: str,
    requester_name: str,
    requester_email: str,
    classification: ClassificationResult,
    cfg: Dict[str, Any],
) -> str:
    # Lazy import so the app can still run without LLM mode.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=get_gemini_api_key())
    model = get_gemini_model()
    system_prompt, user_prompt = _draft_response_prompts(
        request_text, requester_name, requester_email, classification, cfg
    )

    resp = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
        ),
    )
    return resp.text.strip()


def _generate_response_with_ollama(
    request_text: str,
    requester_name: str,
    requester_email: str,
    classification: ClassificationResult,
    cfg: Dict[str, Any],
) -> str:
    system_prompt, user_prompt = _draft_response_prompts(
        request_text, requester_name, requester_email, classification, cfg
    )
    return ollama_chat(system=system_prompt, user=user_prompt, temperature=0.2)


@lru_cache(maxsize=1)
def _load_kb_sections() -> Dict[str, str]:
    if not KB_PATH.exists():
        return {}
    text = KB_PATH.read_text(encoding="utf-8")

    sections: Dict[str, List[str]] = {}
    current_key: str = ""
    for line in text.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith("## "):
            current_key = line_stripped.replace("## ", "").strip()
            sections[current_key] = []
        elif current_key:
            if line_stripped:
                sections[current_key].append(line_stripped)
            else:
                # Preserve paragraph breaks a bit.
                sections[current_key].append("")

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def _choose_kb_section(request_text: str, classification: ClassificationResult) -> str:
    lower = request_text.lower()
    # Match on stronger phrases first.
    if "refund policy" in lower or "refund" in lower and "policy" in lower:
        return "Refunds"
    if "response time" in lower or "how long" in lower:
        return "Response times"
    if "billing" in lower and ("dispute" in lower or "duplicate" in lower or "charge" in lower):
        return "Billing disputes"
    if "activate" in lower or "account" in lower or "billing contact" in lower:
        return "Account updates"
    # Fallback based on sub_topic.
    if "refund" in classification.sub_topic.lower():
        return "Refunds"
    if "response" in classification.sub_topic.lower():
        return "Response times"
    return "Refunds"


def _generate_response_deterministic(
    request_text: str,
    requester_name: str,
    classification: ClassificationResult,
    cfg: Dict[str, Any],
) -> str:
    name = requester_name or "there"
    sla = cfg.get("sla", "")
    lower = request_text.lower()
    details = classification.extracted_details or {}
    matched_signals = details.get("matched_signals", [])
    account_ids = details.get("account_ids_found", []) or []
    emails = details.get("emails_found", []) or []
    deadline_language = details.get("deadline_language", {}) or {}

    if classification.request_type == "Complaint":
        dispute_hint = "a billing dispute" if any("charged twice" in s or "duplicate charge" in s for s in matched_signals) else "your reported issue"
        return (
            f"Hi {name},\n\n"
            f"Thank you for reaching out. I’m sorry for the frustration this has caused. "
            f"Based on your message, we’ve flagged your case as high priority due to {dispute_hint}. "
            f"We have routed it to a senior support handler and will follow up within {sla} with the next update.\n\n"
            "Regards,\nOperations Support"
        )

    if classification.request_type == "General Enquiry":
        kb_sections = _load_kb_sections()
        kb_key = _choose_kb_section(request_text, classification)
        kb_text = kb_sections.get(kb_key, "")
        if not kb_text:
            kb_text = "General enquiries are handled within our standard support timelines."
        return (
            f"Hi {name},\n\n"
            f"Thanks for your question. Based on our support knowledge base ({kb_key}), here’s what to expect:\n\n"
            f"{kb_text}\n\n"
            "If you share any relevant details (account id or transaction info), we can route it faster.\n\n"
            "Regards,\nCustomer Support"
        )

    if classification.request_type == "Service Request":
        # Use the request text to provide a tailored confirmation; avoid “one size fits all”.
        action = "your requested change"
        if "activate" in lower:
            action = "account activation"
        elif "cancel" in lower:
            action = "subscription cancellation"
        elif "billing contact" in lower:
            action = "a billing contact update"
        elif "seat" in lower or "move" in lower:
            action = "a subscription/seat change"
        confirmation = (
            f"Hi {name},\n\n"
            f"Thanks for the request. We’ve routed this to {cfg['routing_team']} to complete {action}. "
            f"The current SLA is {sla}, and we’ll update you once it’s processed.\n\n"
        )
        if not account_ids:
            confirmation += "To speed things up, please reply with your account ID (if available).\n\n"
        return confirmation + "Regards,\nOperations Support"

    if classification.request_type == "Escalation / Urgent":
        # Use a short, safety-forward acknowledgement.
        paused_phrase = "Automated resolution has been paused"
        if deadline_language.get("mentions_immediately"):
            paused_phrase += " due to the immediate nature of your request"
        return (
            f"Hi {name},\n\n"
            f"We’ve received your urgent request and flagged it for immediate human review by a supervisor. "
            f"{paused_phrase} so a reviewer can assess the case and respond appropriately. "
            f"Our team will coordinate next steps as soon as possible.\n\n"
            "Regards,\nOperations Supervisor Team"
        )

    # Needs Human Review
    clarification_questions: List[str] = []
    if any("refund" in s for s in matched_signals) and any("charged" in s or "duplicate" in s for s in matched_signals):
        clarification_questions.append("Are you disputing a specific charge (e.g., duplicate charge) or asking about the refund policy?")
    if "billing contact" in lower:
        clarification_questions.append("Please confirm the exact billing contact name/email you want to update to.")
    if ("activate" in lower or "cancel" in lower or "subscription" in lower) and not account_ids:
        clarification_questions.append("Please provide your account ID so we can process the service request.")
    if ("policy" in lower or "pricing" in lower) and not emails and not account_ids:
        clarification_questions.append("If applicable, share the relevant account or transaction details to tailor the answer.")
    if not clarification_questions:
        clarification_questions = [
            "Please clarify what outcome you want (refund, activation/update, cancellation, or policy information).",
            "If there’s a deadline, please specify it (date/time) so we can prioritize correctly.",
        ]

    return (
        f"Hi {name},\n\n"
        "Thanks for your message. We’re routing this to a human triage queue because the request is ambiguous and needs safe handling.\n\n"
        "To proceed, please confirm the following:\n"
        + "".join([f"- {q}\n" for q in clarification_questions])
        + f"\nOnce confirmed, we’ll route it to the right team (SLA: {sla}).\n\n"
        "Regards,\nOperations Triage Desk"
    )


def _routing_notification(
    requester_name: str,
    requester_email: str,
    classification: ClassificationResult,
    cfg: Dict[str, Any],
    request_id: str,
    received_at: str,
) -> str:
    return (
        f"Case ID: {request_id}\n"
        f"Received: {received_at}\n"
        f"Route to: {cfg['routing_team']}\n"
        f"Requester: {requester_name} <{requester_email}>\n"
        f"Type: {classification.request_type}\n"
        f"Urgency: {classification.urgency}\n"
        f"Confidence: {classification.confidence:.2f}\n"
        f"Reason: {classification.rationale}\n"
        f"SLA/Follow-up: {cfg['sla']}"
    )
