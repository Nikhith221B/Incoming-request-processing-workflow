import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from .models import ClassificationResult
from .utils import (
    clear_last_llm_error,
    compute_confidence_from_scores,
    extract_account_ids,
    extract_deadline_language,
    extract_emails,
    extract_amounts,
    get_gemini_api_key,
    get_gemini_model,
    is_ambiguous,
    load_project_env,
    ollama_chat,
    resolve_llm_provider,
    safe_float,
    set_last_llm_error,
)

load_project_env()

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = ROOT / "data" / "routing_taxonomy.json"

REQUEST_TYPES = [
    "Complaint",
    "General Enquiry",
    "Service Request",
    "Escalation / Urgent",
    "Needs Human Review",
]
URGENCY_LEVELS = ["Low", "Medium", "High", "Critical"]
REVIEW_CONFIDENCE_THRESHOLD = 0.60

DEFAULT_URGENCY_BY_TYPE: Dict[str, str] = {
    "Complaint": "High",
    "General Enquiry": "Low",
    "Service Request": "Medium",
    "Escalation / Urgent": "Critical",
    "Needs Human Review": "Medium",
}

SYSTEM_PROMPT = """You are an operations triage AI. Classify incoming customer requests.
Return only valid JSON with these keys: request_type, urgency, confidence, sub_topic, rationale, extracted_details.
Allowed request_type values: Complaint, General Enquiry, Service Request, Escalation / Urgent, Needs Human Review.
Allowed urgency values: Low, Medium, High, Critical.
Use Escalation / Urgent for legal threats, supervisor demands, outages, public complaints, safety risk, fraud, compliance risk, or immediate deadlines.
Choose Needs Human Review ONLY if:
- the text is unclear or insufficient to route safely
- confidence is low
- multiple risky categories conflict
"""


def classify_request(text: str) -> ClassificationResult:
    clear_last_llm_error()
    provider = resolve_llm_provider()
    if provider == "gemini":
        try:
            return _classify_with_gemini(text)
        except Exception as exc:
            set_last_llm_error(exc, provider="gemini")
            return _fallback_classify(text)
    if provider == "ollama":
        try:
            return _classify_with_ollama(text)
        except Exception as exc:
            set_last_llm_error(exc, provider="ollama")
            return _fallback_classify(text)
    return _fallback_classify(text)


def _classify_with_gemini(text: str) -> ClassificationResult:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=get_gemini_api_key())
    model = get_gemini_model()
    response = client.models.generate_content(
        model=model,
        contents=(
            "Classify this incoming operations request and return JSON only.\n\n"
            f"Request:\n{text}"
        ),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    raw = response.text or ""
    if not raw.strip():
        raise ValueError("Empty Gemini classification response")
    return _build_classification_from_llm_json(text, json.loads(raw))


def _classify_with_ollama(text: str) -> ClassificationResult:
    raw = ollama_chat(
        system=SYSTEM_PROMPT,
        user=(
            "Classify this incoming operations request and return JSON only.\n\n"
            f"Request:\n{text}"
        ),
        json_format=True,
        temperature=0.1,
    )
    return _build_classification_from_llm_json(text, json.loads(raw))


def _taxonomy_review_triggers(text: str) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Reuse fallback taxonomy signals to catch ambiguous/conflicting cases after LLM classification."""
    lower = text.lower()
    taxonomy = _load_routing_taxonomy()
    scores, matched = _score_against_taxonomy(lower, taxonomy)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_type, top_score = ordered[0]
    second_type, second_score = ordered[1] if len(ordered) > 1 else ("", 0.0)

    taxonomy_confidence = compute_confidence_from_scores(top_score, second_score)
    ambiguous = is_ambiguous(top_score, second_score)
    conflict_rules = _matching_conflict_rules(lower, scores, taxonomy)
    insufficient = _is_insufficient_for_auto_route(lower, scores)

    triggers: List[str] = []
    if conflict_rules:
        triggers.append(f"taxonomy_conflict={conflict_rules}")
    if ambiguous and top_score >= 2.0 and second_score >= 2.0:
        triggers.append(
            f"taxonomy_ambiguous ({top_type}={top_score:.1f}, {second_type}={second_score:.1f})"
        )
    if taxonomy_confidence < REVIEW_CONFIDENCE_THRESHOLD:
        triggers.append(f"taxonomy_low_confidence={taxonomy_confidence:.2f}")
    if insufficient:
        triggers.append("insufficient_operational_detail")

    context = {
        "scores": scores,
        "matched": matched,
        "top_type": top_type,
        "second_type": second_type,
        "conflict_rules": conflict_rules,
    }
    return bool(triggers), triggers, context


def _is_insufficient_for_auto_route(lower: str, scores: Dict[str, float]) -> bool:
    """Detect vague low-detail requests that should not be auto-routed (e.g. INB-010)."""
    vague_markers = [
        "thing we discussed",
        "discussed earlier",
        "somehow",
        "please fix",
        "need help with",
        "as discussed",
        "that issue",
        "the issue from before",
    ]
    has_vague = any(marker in lower for marker in vague_markers)
    weak_signals = max(scores.values()) <= 4.0
    lacks_account_context = not any(
        token in lower for token in ["account", "acct", "invoice", "subscription", "billing", "order"]
    )
    word_count = len(lower.split())
    return has_vague and weak_signals and lacks_account_context and word_count <= 25


def _build_classification_from_llm_json(text: str, data: Dict[str, Any]) -> ClassificationResult:
    extracted_details = data.get("extracted_details", {})
    if not isinstance(extracted_details, dict):
        extracted_details = {}

    request_type = _safe_value(data.get("request_type"), REQUEST_TYPES, "General Enquiry")
    urgency = _safe_value(data.get("urgency"), URGENCY_LEVELS, "Low")
    confidence = safe_float(data.get("confidence"), 0.75)
    confidence = max(0.0, min(1.0, confidence))

    rationale = str(data.get("rationale") or "Classified by LLM.")
    sub_topic = str(data.get("sub_topic") or "General")

    result = ClassificationResult(
        request_type=request_type,
        urgency=urgency,
        confidence=confidence,
        sub_topic=sub_topic,
        rationale=rationale,
        extracted_details=extracted_details,
        processing_mode="llm",
    )

    if result.confidence < REVIEW_CONFIDENCE_THRESHOLD:
        result.request_type = "Needs Human Review"
        result.urgency = "Critical" if result.urgency == "Critical" else "Medium"
        result.rationale = f"{result.rationale} (Low confidence; routing to Needs Human Review.)"

    result.human_review_required = (
        result.request_type in {"Needs Human Review", "Escalation / Urgent"} or result.urgency == "Critical"
    )
    if (
        result.urgency == "Critical"
        and result.request_type not in {"Escalation / Urgent", "Needs Human Review"}
    ):
        result.request_type = "Escalation / Urgent"

    should_review, triggers, review_context = _taxonomy_review_triggers(text)
    if should_review and result.request_type not in {"Escalation / Urgent"}:
        result.request_type = "Needs Human Review"
        result.urgency = "Critical" if result.urgency == "Critical" else "Medium"
        result.human_review_required = True
        result.rationale = (
            f"{result.rationale} (Taxonomy safety gate: {'; '.join(triggers)}. "
            "Routing to Needs Human Review.)"
        )
        details = result.extracted_details if isinstance(result.extracted_details, dict) else {}
        details["taxonomy_safety_triggers"] = triggers
        details["conflict_rules"] = review_context.get("conflict_rules", [])
        result.extracted_details = details
    else:
        result.human_review_required = (
            result.request_type in {"Needs Human Review", "Escalation / Urgent"}
            or result.urgency == "Critical"
        )

    return result


def _load_routing_taxonomy() -> Dict[str, Any]:
    """Load fallback routing taxonomy from data/routing_taxonomy.json."""
    if not TAXONOMY_PATH.exists():
        raise FileNotFoundError(f"Missing fallback routing taxonomy: {TAXONOMY_PATH}")
    data = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    categories = data.get("categories", {})
    missing = [
        cat for cat in DEFAULT_URGENCY_BY_TYPE if cat not in categories and cat != "Needs Human Review"
    ]
    if missing:
        raise ValueError(f"Routing taxonomy missing categories: {missing}")
    return data


def _score_against_taxonomy(text: str, taxonomy: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """Return weighted category scores and matched operational signals."""
    scores: Dict[str, float] = {}
    matched: Dict[str, List[str]] = {}

    for category, cfg in taxonomy.get("categories", {}).items():
        scores[category] = 0.0
        matched[category] = []
        for signal in cfg.get("signals", []):
            weight = float(signal.get("weight", 0.0))
            label = str(signal.get("label", "signal"))
            phrases = signal.get("phrases", []) or []
            for phrase in phrases:
                phrase_l = str(phrase).lower().strip()
                if phrase_l and phrase_l in text:
                    scores[category] += weight
                    matched[category].append(f"{label}: {phrase_l}")
                    break

    return scores, matched


def _matches_phrase_group(text: str, phrases: List[str]) -> bool:
    return any(str(p).lower().strip() in text for p in phrases if str(p).strip())


def _matching_conflict_rules(text: str, scores: Dict[str, float], taxonomy: Dict[str, Any]) -> List[str]:
    """Detect taxonomy-defined conflicts that require manual review."""
    matched_rules: List[str] = []
    for rule in taxonomy.get("conflict_rules", []) or []:
        min_scores = rule.get("min_scores", {}) or {}
        score_ok = all(scores.get(cat, 0.0) >= float(min_score) for cat, min_score in min_scores.items())
        phrase_groups = rule.get("required_phrase_groups", []) or []
        phrase_ok = all(_matches_phrase_group(text, group) for group in phrase_groups)
        if score_ok and phrase_ok:
            matched_rules.append(str(rule.get("name", "unnamed_conflict")))
    return matched_rules


def _fallback_classify(text: str) -> ClassificationResult:
    lower = text.lower()
    extracted = _extract_details(text)
    taxonomy = _load_routing_taxonomy()

    scores, matched = _score_against_taxonomy(lower, taxonomy)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_type, top_score = ordered[0]
    second_type, second_score = ordered[1] if len(ordered) > 1 else ("", 0.0)

    confidence = compute_confidence_from_scores(top_score, second_score)
    ambiguous = is_ambiguous(top_score, second_score)
    conflict_rules = _matching_conflict_rules(lower, scores, taxonomy)

    risk_score = scores.get("Escalation / Urgent", 0.0)
    if confidence < REVIEW_CONFIDENCE_THRESHOLD or ambiguous or conflict_rules:
        request_type = "Needs Human Review"
        urgency = "Critical" if risk_score >= 5.0 else DEFAULT_URGENCY_BY_TYPE[request_type]
        rationale = (
            f"Fallback routing is uncertain or conflicting (confidence={confidence:.2f}, "
            f"ambiguous={ambiguous}, conflict_rules={conflict_rules or 'none'}). "
            f"Top signals: {top_type} score={top_score:.1f}, {second_type} score={second_score:.1f}. "
            "Routing to Needs Human Review."
        )
        matched_signals = list(dict.fromkeys(matched.get(top_type, []) + matched.get(second_type, [])))
        extracted["matched_signals"] = matched_signals
        extracted["conflict_rules"] = conflict_rules
    else:
        request_type = top_type
        urgency = DEFAULT_URGENCY_BY_TYPE[request_type]
        rationale = (
            f"Fallback scoring selected {request_type} because its weighted signal score "
            f"({top_score:.1f}) was safely above the next best route ({second_type}: {second_score:.1f}); "
            f"confidence={confidence:.2f}. Matched signals: {matched.get(top_type, [])[:5]}."
        )
        extracted["matched_signals"] = matched.get(top_type, [])[:8]

    result = ClassificationResult(
        request_type=request_type,
        urgency=urgency,
        confidence=confidence,
        sub_topic=_infer_sub_topic(request_type, extracted),
        rationale=rationale,
        extracted_details=extracted,
        human_review_required=request_type in {"Needs Human Review", "Escalation / Urgent"} or urgency == "Critical",
        processing_mode="fallback",
    )
    return result


def _infer_sub_topic(request_type: str, extracted_details: Dict[str, Any]) -> str:
    details_str = " ".join(map(str, extracted_details.get("matched_signals", []))).lower()

    if request_type == "Complaint":
        if any(k in details_str for k in ["charged twice", "duplicate charge", "double charged", "billing_dispute", "billing dispute"]):
            return "Billing dispute / duplicate charge"
        if any(k in details_str for k in ["refund dispute", "reversal", "refund_dispute"]):
            return "Refund reversal / dispute"
        return "Customer dissatisfaction / unresolved issue"

    if request_type == "General Enquiry":
        if "refund policy" in details_str or "refund_policy" in details_str:
            return "Refund policy & review timelines"
        if "response time" in details_str or "response_time" in details_str:
            return "Support response times"
        if "pricing" in details_str:
            return "Pricing & plan availability"
        return "General information request"

    if request_type == "Service Request":
        if "billing contact" in details_str or "billing_contact" in details_str:
            return "Update billing contact"
        if any(k in details_str for k in ["activate", "account activation", "activation"]):
            return "Account activation / onboarding"
        if "cancel" in details_str:
            return "Subscription cancellation"
        if "seat" in details_str or "plan_change" in details_str:
            return "Seat / subscription plan change"
        return "Operational account/service change"

    if request_type == "Escalation / Urgent":
        if any(k in details_str for k in ["outage", "service down", "is down", "launch day", "launch_incident"]):
            return "Outage / launch incident"
        if any(k in details_str for k in ["legal", "lawsuit", "fraud", "compliance", "safety"]):
            return "Legal/compliance/safety escalation"
        if "supervisor" in details_str:
            return "Supervisor intervention requested"
        return "Urgent escalation"

    return "Ambiguous request; needs triage"


def _safe_value(value: Any, allowed: list, default: str) -> str:
    return value if value in allowed else default


def _extract_details(text: str) -> Dict[str, Any]:
    return {
        "emails_found": extract_emails(text),
        "amounts_found": extract_amounts(text),
        "account_ids_found": extract_account_ids(text),
        "deadline_language": extract_deadline_language(text),
    }
