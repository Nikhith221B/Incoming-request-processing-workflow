"""Shared helpers: IDs, extraction, confidence scoring, and LLM mode (no secrets)."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from math import isnan
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
_LAST_LLM_ERROR: Optional[str] = None


def load_project_env() -> None:
    """Load `.env` from project root regardless of cwd."""
    load_dotenv(ROOT / ".env", override=False)


def get_gemini_api_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or "").strip()


def get_gemini_model() -> str:
    return (os.getenv("GEMINI_MODEL") or "gemini-2.0-flash").strip()


def get_llm_provider_setting() -> str:
    return (os.getenv("LLM_PROVIDER") or "auto").strip().lower()


def get_ollama_base_url() -> str:
    return (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip()


def get_ollama_model() -> str:
    return (os.getenv("OLLAMA_MODEL") or "llama3.2").strip()


def resolve_llm_provider() -> Optional[str]:
    """Return active provider ('gemini' | 'ollama') or None for fallback-only mode."""
    setting = get_llm_provider_setting()
    if setting in {"none", "fallback", "off", "disabled"}:
        return None
    if setting == "gemini":
        return "gemini" if get_gemini_api_key() else None
    if setting == "ollama":
        return "ollama"
    # auto: prefer Gemini when a key is present, otherwise try local Ollama.
    if get_gemini_api_key():
        return "gemini"
    return "ollama"


def is_llm_enabled() -> bool:
    return resolve_llm_provider() is not None


def has_llm_key() -> bool:
    """Backward-compatible alias: True when an LLM provider would be attempted."""
    return is_llm_enabled()


def ollama_chat(
    *,
    system: str,
    user: str,
    json_format: bool = False,
    temperature: float = 0.1,
    timeout: int = 120,
) -> str:
    """Call a local Ollama instance via its native /api/chat endpoint."""
    import urllib.error
    import urllib.request

    url = f"{get_ollama_base_url().rstrip('/')}/api/chat"
    payload: Dict[str, Any] = {
        "model": get_ollama_model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_format:
        payload["format"] = "json"

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Ollama request failed at {get_ollama_base_url()} ({exc}). "
            "Is Ollama running and is the model pulled?"
        ) from exc

    content = (data.get("message") or {}).get("content") or ""
    if not content.strip():
        raise ValueError("Empty Ollama response")
    return content.strip()


def set_last_llm_error(
    exc: Optional[BaseException] = None,
    message: Optional[str] = None,
    provider: str = "gemini",
) -> None:
    global _LAST_LLM_ERROR
    if message:
        _LAST_LLM_ERROR = message
        return
    if exc is None:
        _LAST_LLM_ERROR = None
        return
    text = str(exc)
    lowered = text.lower()
    label = "Ollama" if provider == "ollama" else "Gemini"
    if provider == "ollama" and (
        "connection refused" in lowered
        or "urlerror" in lowered
        or "ollama request failed" in lowered
        or "actively refused" in lowered
        or "timed out" in lowered
    ):
        _LAST_LLM_ERROR = (
            f"Ollama is not reachable at {get_ollama_base_url()} "
            f"(model: {get_ollama_model()}). Using Fallback Demo Mode."
        )
    elif "429" in text or "resource_exhausted" in lowered or "quota" in lowered:
        _LAST_LLM_ERROR = f"{label} API quota/rate limit exceeded (429). Using Fallback Demo Mode."
    elif "401" in text or "403" in text or "unauthorized" in lowered or "permission" in lowered:
        _LAST_LLM_ERROR = f"{label} API key rejected or unauthorized. Using Fallback Demo Mode."
    elif "api key" in lowered:
        _LAST_LLM_ERROR = f"{label} API authentication failed. Using Fallback Demo Mode."
    else:
        _LAST_LLM_ERROR = f"{label} unavailable ({type(exc).__name__}). Using Fallback Demo Mode."


def clear_last_llm_error() -> None:
    global _LAST_LLM_ERROR
    _LAST_LLM_ERROR = None


def get_last_llm_error() -> Optional[str]:
    return _LAST_LLM_ERROR


def mode_banner_state() -> Tuple[str, str]:
    """Returns (level, message) for Streamlit banner. Never includes secrets."""
    setting = get_llm_provider_setting()
    provider = resolve_llm_provider()
    if provider is None:
        if setting == "gemini" and not get_gemini_api_key():
            return (
                "info",
                "Mode: Fallback Demo Mode (LLM_PROVIDER=gemini but no GEMINI_API_KEY). "
                "Deterministic classification + templated ops outputs.",
            )
        if setting in {"none", "fallback", "off", "disabled"}:
            return (
                "info",
                "Mode: Fallback Demo Mode (LLM disabled). Deterministic classification + templated ops outputs.",
            )
        return (
            "info",
            "Mode: Fallback Demo Mode (no LLM configured). Deterministic classification + templated ops outputs.",
        )

    err = get_last_llm_error()
    if err:
        return ("warning", f"Mode: Fallback Demo Mode ({provider} call failed). {err}")

    if provider == "gemini":
        return (
            "success",
            f"Mode: LLM Mode (Gemini · {get_gemini_model()}). "
            "Uses Gemini when calls succeed; falls back automatically on errors.",
        )
    return (
        "success",
        f"Mode: LLM Mode (Ollama · {get_ollama_model()} @ {get_ollama_base_url()}). "
        "Uses local Ollama when calls succeed; falls back automatically on errors.",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def generate_request_id(prefix: str = "REQ") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10].upper()}"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        if isnan(f):
            return default
        return f
    except Exception:
        return default


def safe_json_loads(raw: str) -> Dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object")
    return parsed


def extract_emails(text: str) -> List[str]:
    return re.findall(r"[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}", text)


def extract_amounts(text: str) -> List[str]:
    return re.findall(
        r"(?:\$|INR|Rs\.?\s*)\s?\d+(?:,\d{3})*(?:\.\d{2})?",
        text,
        flags=re.IGNORECASE,
    )


def extract_account_ids(text: str) -> List[str]:
    ids: List[str] = []
    for m in re.findall(
        r"\b(?:acct|account|customer)\b\s*(?:id|#|number|:)?\s*([A-Za-z0-9][A-Za-z0-9\-_]{2,})",
        text,
        flags=re.IGNORECASE,
    ):
        if m:
            ids.append(m)
    seen = set()
    out: List[str] = []
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def extract_deadline_language(text: str) -> Dict[str, Any]:
    lower = text.lower()
    flags = {
        "mentions_today": "today" in lower,
        "mentions_immediately": any(w in lower for w in ["immediately", "right now", "now", "asap"]),
        "mentions_launch_day": "launch day" in lower or ("launch" in lower and "day" in lower),
        "deadline_phrases": [],
    }
    for phrase in ["today", "immediately", "asap", "right now", "launch day"]:
        if phrase in lower:
            flags["deadline_phrases"].append(phrase)
    return flags


def compute_confidence_from_scores(top_score: float, second_score: float, min_strength: float = 2.0) -> float:
    top = max(0.0, top_score)
    second = max(0.0, second_score)
    margin = max(0.0, top - second)
    margin_factor = margin / (margin + 2.5)
    strength_factor = top / (top + 3.0)
    base = 0.15 + 0.70 * (0.6 * margin_factor + 0.4 * strength_factor)
    if top < min_strength:
        base *= 0.7
    return clamp(base, 0.05, 0.99)


def is_ambiguous(top_score: float, second_score: float, margin_threshold: float = 1.5, min_strength: float = 2.0) -> bool:
    margin = top_score - second_score
    return (top_score < min_strength) or (margin < margin_threshold)
