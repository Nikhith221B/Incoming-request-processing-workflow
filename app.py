"""Incoming Request Processing Workflow — Streamlit ops dashboard."""

from __future__ import annotations

import json
import html
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.classifier import classify_request
from src.dashboard import (
    case_log_table,
    export_csv_bytes,
    filter_case_log,
    get_dashboard_metrics,
    load_case_log,
    unique_options,
    render_horizontal_bar_chart,
    value_counts_series,
    workflow_coverage_summary,
)
from src.file_parser import parse_uploaded_file
from src.logger import clear_logs, insert_log
from src.pipeline import process_incoming_request
from src.sample_data import get_inbox_samples, get_sample_label_map
from src.utils import load_project_env, mode_banner_state
from src.workflows import execute_workflow

load_project_env()

st.set_page_config(
    page_title="OpsFlow AI Case Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV_ITEMS: List[Dict[str, str]] = [
    {"id": "single", "label": "Single Request Intake", "icon": "📥", "hint": "Create and triage one case"},
    {"id": "batch", "label": "File Upload / Batch", "icon": "📁", "hint": "CSV, TXT, PDF, DOCX intake"},
    {"id": "inbox", "label": "Simulated Inbox", "icon": "📬", "hint": "10 demo messages, all branches"},
    {"id": "webhook", "label": "Webhook Intake", "icon": "🔗", "hint": "Local API push intake"},
    {"id": "dashboard", "label": "Dashboard / Audit Log", "icon": "📊", "hint": "Metrics, filters, export"},
]
SECTIONS = [item["label"] for item in NAV_ITEMS]

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
  /* Dark chrome: sidebar + header (always) */
  --ops-chrome-bg: #0a0e17;
  --ops-chrome-elevated: #121a2a;
  --ops-chrome-text: #f8fafc;
  --ops-chrome-muted: #94a3b8;
  --ops-chrome-border: #243044;
  --ops-chrome-hover: rgba(255, 255, 255, 0.06);
  --ops-chrome-active: rgba(59, 130, 246, 0.22);
  --ops-brand: #3b82f6;
  --ops-brand-bright: #60a5fa;
  /* Workspace tokens — light defaults (dark overridden below) */
  --ops-surface: #ffffff;
  --ops-muted: #f4f6fb;
  --ops-text: #0f172a;
  --ops-text-muted: #64748b;
  --ops-border: #dbe3ef;
  --ops-shadow: rgba(15, 23, 42, 0.06);
  --ops-ticket-bg: #eef4ff;
  --ops-success: #10b981;
  --ops-warning: #f59e0b;
}

/* Dark / system-resolved-dark — workspace matches sidebar chrome */
.stApp[data-theme="dark"],
html[data-theme="dark"] .stApp {
  --ops-surface: #121a2a;
  --ops-muted: #0a0e17;
  --ops-text: #f8fafc;
  --ops-text-muted: #94a3b8;
  --ops-border: #243044;
  --ops-shadow: rgba(0, 0, 0, 0.35);
  --ops-ticket-bg: #121a2a;
}

/* Light theme workspace */
.stApp[data-theme="light"],
html[data-theme="light"] .stApp {
  --ops-surface: #ffffff;
  --ops-muted: #f4f6fb;
  --ops-text: #0f172a;
  --ops-text-muted: #64748b;
  --ops-border: #dbe3ef;
  --ops-shadow: rgba(15, 23, 42, 0.06);
  --ops-ticket-bg: #eef4ff;
}

.block-container {
  padding-top: 0.65rem;
  padding-bottom: 1.25rem;
  max-width: 1180px;
}

/* Dark chrome header — matches sidebar */
.ops-hero {
  border: 1px solid var(--ops-chrome-border);
  border-radius: 10px;
  padding: 11px 16px;
  margin-bottom: 14px;
  background: linear-gradient(180deg, var(--ops-chrome-elevated) 0%, var(--ops-chrome-bg) 100%);
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.22);
  color: var(--ops-chrome-text);
}
.ops-hero-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.ops-hero-main h1 {
  margin: 0;
  font-size: 1.12rem;
  line-height: 1.2;
  font-weight: 700;
  color: var(--ops-chrome-text);
}
.ops-hero-main p {
  color: var(--ops-chrome-muted);
  margin: 3px 0 0 0;
  font-size: 0.78rem;
}
.ops-eyebrow {
  color: var(--ops-brand-bright);
  font-size: 0.62rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 700;
  margin-bottom: 2px;
}
.mode-pill, .status-pill {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.7rem;
  font-weight: 600;
  border: 1px solid var(--ops-chrome-border);
  background: rgba(255, 255, 255, 0.06);
  color: var(--ops-chrome-text);
  white-space: nowrap;
}
.mode-success {
  color: #6ee7b7;
  background: rgba(16, 185, 129, 0.15);
  border-color: rgba(16, 185, 129, 0.35);
}
.mode-warning {
  color: #fcd34d;
  background: rgba(245, 158, 11, 0.15);
  border-color: rgba(245, 158, 11, 0.35);
}
.mode-info {
  color: var(--ops-brand-bright);
  background: rgba(59, 130, 246, 0.15);
  border-color: rgba(59, 130, 246, 0.35);
}

/* Case panels */
.case-panel {
  border: 1px solid var(--ops-border);
  border-radius: 12px;
  padding: 14px 16px;
  margin: 10px 0;
  background: var(--ops-surface);
  box-shadow: 0 1px 4px var(--ops-shadow);
  color: var(--ops-text);
}
.case-title-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 10px;
}
.case-title-row h3 { margin: 2px 0 0 0; color: var(--ops-text); }
.case-panel .ops-eyebrow { color: var(--ops-brand-bright); }
.case-subtitle { color: var(--ops-text-muted); font-size: 0.8rem; }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(96px, 1fr));
  gap: 8px;
  margin: 10px 0 8px;
}
.metric-card {
  border: 1px solid var(--ops-border);
  border-radius: 8px;
  padding: 9px 10px;
  background: var(--ops-muted);
}
.metric-card span {
  color: var(--ops-text-muted);
  font-size: 0.68rem;
  display: block;
}
.metric-card b { display: block; margin-top: 3px; font-size: 0.88rem; color: var(--ops-text); }
.ticket-body {
  border-left: 3px solid var(--ops-brand);
  background: var(--ops-ticket-bg);
  border-radius: 8px;
  padding: 11px 13px;
  white-space: pre-wrap;
  font-size: 0.86rem;
  line-height: 1.5;
  color: var(--ops-text);
}
.step-card {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  border: 1px solid var(--ops-border);
  border-radius: 8px;
  padding: 9px 11px;
  margin-bottom: 7px;
  background: var(--ops-surface);
  color: var(--ops-text);
}
.step-index {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: color-mix(in srgb, var(--ops-brand) 14%, var(--ops-surface));
  color: var(--ops-brand);
  font-weight: 700;
  font-size: 0.68rem;
  flex: 0 0 auto;
}
.info-card {
  border: 1px solid var(--ops-border);
  border-radius: 10px;
  padding: 12px;
  background: var(--ops-surface);
  color: var(--ops-text);
}
.info-card h4 { margin: 0 0 8px 0; color: var(--ops-text); font-size: 0.92rem; }
.queue-note {
  color: var(--ops-text-muted);
  font-size: 0.8rem;
  margin-bottom: 4px;
}

/* Sidebar — always dark chrome, white content */
[data-testid="stSidebar"] {
  background: var(--ops-chrome-bg) !important;
  border-right: 1px solid var(--ops-chrome-border) !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 0.85rem;
  padding-bottom: 1rem;
}
[data-testid="stSidebar"] .stCaption {
  color: var(--ops-chrome-muted) !important;
}
.sidebar-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 2px 2px 12px;
  margin-bottom: 10px;
  border-bottom: 1px solid var(--ops-chrome-border);
}
.sidebar-logo {
  width: 34px;
  height: 34px;
  border-radius: 9px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(145deg, #3b82f6, #1d4ed8);
  color: #fff;
  font-weight: 800;
  font-size: 0.95rem;
  flex-shrink: 0;
  box-shadow: 0 2px 10px rgba(59, 130, 246, 0.35);
}
.sidebar-brand-title {
  font-size: 0.98rem;
  font-weight: 700;
  color: var(--ops-chrome-text) !important;
  margin: 0;
  line-height: 1.15;
}
.sidebar-brand-sub {
  font-size: 0.7rem;
  color: var(--ops-chrome-muted) !important;
  margin: 2px 0 0 0;
}
.sidebar-section-label {
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: #94a3b8 !important;
  margin: 0.35rem 0 0.45rem 0.1rem;
}
.nav-desc {
  font-size: 0.68rem;
  color: var(--ops-chrome-muted) !important;
  margin: -0.2rem 0 0.5rem 0.15rem;
  line-height: 1.25;
}
.health-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  margin-top: 2px;
}
.health-stat {
  border: 1px solid var(--ops-chrome-border);
  border-radius: 8px;
  padding: 8px 6px;
  background: var(--ops-chrome-elevated);
  text-align: center;
}
.health-stat span {
  display: block;
  font-size: 0.62rem;
  color: var(--ops-chrome-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.health-stat b {
  display: block;
  margin-top: 2px;
  font-size: 1rem;
  color: var(--ops-chrome-text);
}
.sidebar-footnote {
  font-size: 0.68rem;
  color: var(--ops-chrome-muted) !important;
  line-height: 1.4;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid var(--ops-chrome-border);
}

/* Sidebar nav buttons — force light text on dark chrome */
[data-testid="stSidebar"] .stButton {
  margin-bottom: 0;
}
[data-testid="stSidebar"] .stButton > button {
  width: 100%;
  min-height: 38px;
  padding: 0.45rem 0.7rem !important;
  border-radius: 8px !important;
  border: 1px solid transparent !important;
  box-shadow: none !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  text-align: left !important;
  justify-content: flex-start !important;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}
[data-testid="stSidebar"] .stButton > button * {
  color: inherit !important;
  -webkit-text-fill-color: inherit !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[kind="primary"] *,
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] * {
  background: transparent !important;
  background-color: var(--ops-chrome-active) !important;
  border-color: rgba(59, 130, 246, 0.45) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  font-weight: 600 !important;
  box-shadow: inset 3px 0 0 var(--ops-brand) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"],
[data-testid="stSidebar"] .stButton > button[kind="secondary"] *,
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"],
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"] * {
  background: transparent !important;
  background-color: transparent !important;
  color: #e2e8f0 !important;
  -webkit-text-fill-color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover *,
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"]:hover,
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"]:hover * {
  background: var(--ops-chrome-hover) !important;
  background-color: var(--ops-chrome-hover) !important;
  border-color: var(--ops-chrome-border) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

/* Main workspace */
[data-testid="stMain"],
[data-testid="stAppViewContainer"] > section.main {
  background: var(--ops-muted) !important;
}
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] .stCaption,
[data-testid="stMain"] [data-testid="stMarkdownContainer"],
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li {
  color: var(--ops-text);
}
[data-testid="stMain"] .stAlert {
  border-radius: 8px;
}

/* Case status pills inside workspace cards */
.case-panel .status-pill {
  background: var(--ops-muted);
  border-color: var(--ops-border);
  color: var(--ops-text);
}
.case-panel .mode-success {
  color: #6ee7b7;
  background: rgba(16, 185, 129, 0.15);
  border-color: rgba(16, 185, 129, 0.35);
}
.case-panel .mode-warning {
  color: #fcd34d;
  background: rgba(245, 158, 11, 0.15);
  border-color: rgba(245, 158, 11, 0.35);
}
.stApp[data-theme="dark"] .case-panel .mode-success {
  color: #6ee7b7;
}
.stApp[data-theme="dark"] .case-panel .mode-warning {
  color: #fcd34d;
}

/* Streamlit widgets in dark workspace */
.stApp[data-theme="dark"] [data-testid="stMain"] input,
.stApp[data-theme="dark"] [data-testid="stMain"] textarea,
.stApp[data-theme="dark"] [data-testid="stMain"] [data-baseweb="select"],
html[data-theme="dark"] .stApp [data-testid="stMain"] input,
html[data-theme="dark"] .stApp [data-testid="stMain"] textarea,
html[data-theme="dark"] .stApp [data-testid="stMain"] [data-baseweb="select"] {
  color: var(--ops-text);
  background-color: var(--ops-surface);
  border-color: var(--ops-border);
}
.stApp[data-theme="dark"] [data-testid="stMain"] [data-testid="stCode"],
html[data-theme="dark"] .stApp [data-testid="stMain"] [data-testid="stCode"] {
  background: var(--ops-surface);
  color: var(--ops-text);
}
.stApp[data-theme="dark"] [data-testid="stMain"] .stTextArea textarea[disabled],
html[data-theme="dark"] .stApp [data-testid="stMain"] .stTextArea textarea[disabled] {
  color: var(--ops-text) !important;
  -webkit-text-fill-color: var(--ops-text) !important;
  opacity: 1 !important;
  background-color: var(--ops-surface) !important;
}

/* ── Dark / system: force unified chrome background + readable text ── */
.stApp[data-theme="dark"],
html[data-theme="dark"] .stApp,
html[data-theme="dark"] body {
  background-color: #0a0e17 !important;
}
.stApp[data-theme="dark"] [data-testid="stAppViewContainer"],
html[data-theme="dark"] [data-testid="stAppViewContainer"],
.stApp[data-theme="dark"] [data-testid="stAppViewContainer"] > section.main,
html[data-theme="dark"] [data-testid="stAppViewContainer"] > section.main,
.stApp[data-theme="dark"] [data-testid="stMain"],
html[data-theme="dark"] [data-testid="stMain"],
.stApp[data-theme="dark"] [data-testid="stMain"] > div,
html[data-theme="dark"] [data-testid="stMain"] > div,
.stApp[data-theme="dark"] .block-container,
html[data-theme="dark"] .block-container {
  background-color: #0a0e17 !important;
  color: #f8fafc !important;
}
.stApp[data-theme="dark"] [data-testid="stMain"] h1,
.stApp[data-theme="dark"] [data-testid="stMain"] h2,
.stApp[data-theme="dark"] [data-testid="stMain"] h3,
.stApp[data-theme="dark"] [data-testid="stMain"] h4,
.stApp[data-theme="dark"] [data-testid="stMain"] p,
.stApp[data-theme="dark"] [data-testid="stMain"] label,
.stApp[data-theme="dark"] [data-testid="stMain"] .stMarkdown,
.stApp[data-theme="dark"] [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
.stApp[data-theme="dark"] [data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
.stApp[data-theme="dark"] [data-testid="stMain"] .stCaption,
html[data-theme="dark"] [data-testid="stMain"] h1,
html[data-theme="dark"] [data-testid="stMain"] h2,
html[data-theme="dark"] [data-testid="stMain"] h3,
html[data-theme="dark"] [data-testid="stMain"] h4,
html[data-theme="dark"] [data-testid="stMain"] p,
html[data-theme="dark"] [data-testid="stMain"] label,
html[data-theme="dark"] [data-testid="stMain"] .stMarkdown,
html[data-theme="dark"] [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
html[data-theme="dark"] [data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
html[data-theme="dark"] [data-testid="stMain"] .stCaption {
  color: #f8fafc !important;
}
.stApp[data-theme="dark"] [data-testid="stMain"] label[data-testid="stWidgetLabel"],
html[data-theme="dark"] [data-testid="stMain"] label[data-testid="stWidgetLabel"] {
  color: #cbd5e1 !important;
}
.stApp[data-theme="dark"] .info-card,
html[data-theme="dark"] .info-card {
  background: #121a2a !important;
  border-color: #243044 !important;
  color: #f8fafc !important;
}
.stApp[data-theme="dark"] .info-card h4,
.stApp[data-theme="dark"] .queue-note,
html[data-theme="dark"] .info-card h4,
html[data-theme="dark"] .queue-note {
  color: #f8fafc !important;
}
html[data-theme="dark"] .queue-note {
  color: #94a3b8 !important;
}

/* System theme fallback when OS is dark */
@media (prefers-color-scheme: dark) {
  html:not([data-theme="light"]) .stApp,
  html:not([data-theme="light"]) [data-testid="stAppViewContainer"],
  html:not([data-theme="light"]) [data-testid="stMain"],
  html:not([data-theme="light"]) .block-container {
    background-color: #0a0e17 !important;
  }
  html:not([data-theme="light"]) .stApp {
    --ops-surface: #121a2a;
    --ops-muted: #0a0e17;
    --ops-text: #f8fafc;
    --ops-text-muted: #94a3b8;
    --ops-border: #243044;
    --ops-ticket-bg: #121a2a;
  }
  html:not([data-theme="light"]) [data-testid="stMain"] h1,
  html:not([data-theme="light"]) [data-testid="stMain"] h2,
  html:not([data-theme="light"]) [data-testid="stMain"] h3,
  html:not([data-theme="light"]) [data-testid="stMain"] p,
  html:not([data-theme="light"]) [data-testid="stMain"] label,
  html:not([data-theme="light"]) [data-testid="stMain"] [data-testid="stMarkdownContainer"] p {
    color: #f8fafc !important;
  }
  html:not([data-theme="light"]) .info-card {
    background: #121a2a !important;
    border-color: #243044 !important;
    color: #f8fafc !important;
  }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def _mode_class(level: str) -> str:
    return {"success": "mode-success", "warning": "mode-warning"}.get(level, "mode-info")


def render_shell_header() -> None:
    level, banner = mode_banner_state()
    st.markdown(
        f"""
        <div class="ops-hero">
          <div class="ops-hero-row">
            <div class="ops-hero-main">
              <div class="ops-eyebrow">OpsFlow · AI Triage</div>
              <h1>Case Command Center</h1>
              <p>Intake → classify → remediate → audit</p>
            </div>
            <span class="mode-pill {_mode_class(level)}">{_escape(banner)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_playbook_card() -> None:
    st.markdown(
        """
        <div class="info-card">
          <h4>Routing playbook</h4>
          <div class="queue-note">Complaint → Senior Customer Support · 2-hour follow-up</div>
          <div class="queue-note">General Enquiry → Knowledge Desk · auto-resolved when safe</div>
          <div class="queue-note">Service Request → Account/Ops queue · SLA timer</div>
          <div class="queue-note">Escalation → Supervisor review · auto-resolution paused</div>
          <div class="queue-note">Low confidence / conflict → Manual triage</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def process_one(
    requester_name: str,
    requester_email: str,
    request_text: str,
    account_id: str = "",
    channel: str = "web_form",
):
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
    return classification, workflow, audit


def _parse_actions(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            return [part.strip() for part in raw.split(";") if part.strip()]
    return []


def case_dict_from_live(classification, workflow, audit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    audit = audit or {}
    return {
        "request_id": workflow.request_id or audit.get("request_id"),
        "timestamp": workflow.received_at or audit.get("timestamp"),
        "requester_name": audit.get("requester_name", ""),
        "requester_email": audit.get("requester_email", ""),
        "account_id": audit.get("account_id", ""),
        "channel": audit.get("channel", ""),
        "request_text": audit.get("request_text", ""),
        "request_type": classification.request_type,
        "urgency": classification.urgency,
        "confidence": classification.confidence,
        "sub_topic": classification.sub_topic,
        "rationale": classification.rationale,
        "selected_branch": workflow.selected_branch or classification.request_type,
        "status": workflow.status,
        "routed_team": workflow.routed_team,
        "routing_destination": workflow.routing_destination or workflow.routed_team,
        "sla_or_follow_up": workflow.sla_or_follow_up,
        "actions_triggered": workflow.actions_triggered,
        "generated_response": workflow.generated_response,
        "routing_notification": workflow.routing_notification,
        "human_review_required": workflow.human_review_required,
        "escalation_flag": workflow.escalation_flag,
        "processing_mode": workflow.processing_mode or classification.processing_mode,
        "extracted_details": classification.extracted_details,
    }


def render_case_detail(case: Dict[str, Any], title: str = "Case detail"):
    """Full ops-style case view used by Single / Inbox / Batch / Audit drill-down."""
    request_id = case.get("request_id") or "—"
    request_type = case.get("request_type") or "—"
    urgency = case.get("urgency") or "—"
    confidence = case.get("confidence")
    try:
        conf_txt = f"{float(confidence):.2f}"
    except Exception:
        conf_txt = str(confidence or "—")
    status = case.get("status") or "—"
    esc = bool(case.get("escalation_flag"))
    hr = bool(case.get("human_review_required"))
    mode_label = str(case.get("processing_mode") or "fallback").upper()
    queue = case.get("routing_destination") or case.get("routed_team") or "—"
    branch = case.get("selected_branch") or request_type

    badge_class = "mode-warning" if esc or hr or status in {"Escalated", "Human Review Required", "Pending Review"} else "mode-success"
    st.markdown(
        f"""
        <div class="case-panel">
          <div class="case-title-row">
            <div>
              <div class="ops-eyebrow">{_escape(title)}</div>
              <h3>{_escape(request_id)}</h3>
              <div class="case-subtitle">Requester: {_escape(case.get('requester_name') or '—')} · Channel: {_escape(case.get('channel') or '—')} · Account: {_escape(case.get('account_id') or '—')}</div>
            </div>
            <span class="status-pill {badge_class}">{_escape(status)}</span>
          </div>
          <div class="metric-grid">
            <div class="metric-card"><span>Classification</span><b>{_escape(request_type)}</b></div>
            <div class="metric-card"><span>Urgency</span><b>{_escape(urgency)}</b></div>
            <div class="metric-card"><span>Confidence</span><b>{_escape(conf_txt)}</b></div>
            <div class="metric-card"><span>Assigned queue</span><b>{_escape(queue)}</b></div>
            <div class="metric-card"><span>SLA / follow-up</span><b>{_escape(case.get('sla_or_follow_up') or '—')}</b></div>
          </div>
          <div class="case-subtitle">Branch: <b>{_escape(branch)}</b> · Engine: <b>{_escape(mode_label)}</b> · Escalation: <b>{'ON' if esc else 'OFF'}</b> · Human review: <b>{'YES' if hr else 'NO'}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if esc or request_type == "Escalation / Urgent":
        st.error(
            "ESCALATION ACTIVE — Supervisor notification created (simulated). "
            "Auto-resolution paused. Case is in human-review queue."
        )
    elif request_type == "Needs Human Review" or hr:
        st.warning("NEEDS HUMAN REVIEW — Ambiguous/low-confidence case queued for manual triage.")
    elif status == "Escalated":
        st.warning("Priority complaint escalation path executed (senior support routing logged).")

    level, banner = mode_banner_state()
    if level == "warning":
        st.warning(banner)

    st.subheader("1) Incoming request")
    meta = (
        f"**Requester:** {case.get('requester_name') or '—'}  ·  "
        f"**Email:** {case.get('requester_email') or '—'}  ·  "
        f"**Channel:** {case.get('channel') or '—'}  ·  "
        f"**Account:** {case.get('account_id') or '—'}"
    )
    st.markdown(meta)
    request_html = _escape(case.get("request_text") or "(no request text stored)").replace("\n", "<br>")
    st.markdown(f'<div class="ticket-body">{request_html}</div>', unsafe_allow_html=True)

    st.subheader("2) Classification decision")
    st.write(case.get("rationale") or "—")
    if case.get("sub_topic"):
        st.caption(f"Sub-topic: {case.get('sub_topic')}")

    st.subheader("3) Remediation workflow (downstream steps)")
    actions = _parse_actions(case.get("actions_triggered"))
    if not actions and case.get("action_summary"):
        actions = _parse_actions(case.get("action_summary"))
    if actions:
        for idx, action in enumerate(actions, start=1):
            st.markdown(
                f'<div class="step-card"><span class="step-index">{idx}</span><div><b>Step {idx}</b><br>{_escape(action)}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No downstream actions recorded for this case.")

    st.subheader("4) Case timeline")
    timeline = [
        f"Received at {case.get('timestamp') or '—'}",
        f"Classified as {request_type} / {urgency} (confidence {conf_txt})",
        f"Routed to {case.get('routing_destination') or case.get('routed_team') or '—'}",
        f"Follow-up / SLA set: {case.get('sla_or_follow_up') or '—'}",
        f"Status updated to: {status}",
    ]
    if esc or request_type == "Escalation / Urgent":
        timeline.append("Escalation path: supervisor notified (simulated); auto-resolution paused")
    if request_type == "Needs Human Review" or hr:
        timeline.append("Human-in-the-loop: clarification / manual triage queue")
    timeline.append("Outcome logged to audit trail")
    for step in timeline:
        st.write(f"- {step}")

    left, right = st.columns(2)
    with left:
        st.subheader("5) Draft customer response")
        st.text_area(
            "Editable draft",
            value=case.get("generated_response") or "—",
            height=240,
            disabled=True,
            label_visibility="collapsed",
            key=f"draft_{request_id}",
        )
    with right:
        st.subheader("6) Internal routing note")
        st.text_area(
            "Internal note",
            value=case.get("routing_notification") or "—",
            height=240,
            disabled=True,
            label_visibility="collapsed",
            key=f"route_{request_id}",
        )

    extracted = case.get("extracted_details")
    if extracted:
        st.subheader("Extracted fields")
        if isinstance(extracted, str):
            try:
                extracted = json.loads(extracted)
            except Exception:
                pass
        st.json(extracted)

    with st.expander("Full case record (audit fields)"):
        st.json(case)


def render_result(classification, workflow, audit=None):
    render_case_detail(case_dict_from_live(classification, workflow, audit), title="Case created / updated")


@st.fragment
def _audit_case_drilldown(filtered: pd.DataFrame) -> None:
    """Isolated drill-down so toggling detail does not rerun sidebar navigation."""
    if filtered.empty:
        st.info("No cases match the current filters.")
        return

    detail_df = filtered.copy()
    if "id" in detail_df.columns:
        detail_df = detail_df.sort_values("id", ascending=False)

    options: List[str] = []
    option_map: Dict[str, Dict[str, Any]] = {}
    for _, row in detail_df.iterrows():
        rid = str(row.get("request_id") or row.get("id") or "CASE")
        label = (
            f"{rid} · {row.get('request_type')} · {row.get('urgency')} · {row.get('status')} · "
            f"{str(row.get('requester_name') or '')}"
        )
        options.append(label)
        option_map[label] = row.to_dict()

    chosen = st.selectbox("Select a case from the filtered log", options, key="audit_case_pick")
    show = st.checkbox(
        "Show full remediation detail for selected audit case",
        value=False,
        key="audit_show_detail",
    )
    if show:
        render_case_detail(option_map[chosen], title="Audit case detail")
    else:
        st.caption("Case list is enough for scanning. Enable the checkbox only when you need the full case file.")


def _init_workspace() -> None:
    if "ops_active_section" not in st.session_state:
        legacy = st.session_state.get("workspace_section")
        st.session_state.ops_active_section = legacy if legacy in SECTIONS else SECTIONS[0]


def _navigate_to(section_label: str) -> None:
    st.session_state.ops_active_section = section_label


def render_sidebar() -> str:
    """Rich vertical nav (Jira / ServiceNow style) with queue health snapshot."""
    _init_workspace()
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-logo">O</div>
          <div>
            <p class="sidebar-brand-title">OpsFlow</p>
            <p class="sidebar-brand-sub">Case Command Center</p>
          </div>
        </div>
        <p class="sidebar-section-label">Workspace</p>
        """,
        unsafe_allow_html=True,
    )

    for item in NAV_ITEMS:
        is_active = st.session_state.ops_active_section == item["label"]
        st.button(
            f"{item['icon']}  {item['label']}",
            key=f"nav_{item['id']}",
            width="stretch",
            type="primary" if is_active else "secondary",
            on_click=_navigate_to,
            args=(item["label"],),
        )
        if is_active:
            st.markdown(f'<p class="nav-desc">{_escape(item["hint"])}</p>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section-label">Queue health</p>', unsafe_allow_html=True)
    try:
        _sidebar_log = load_case_log(200)
        if _sidebar_log.empty:
            st.caption("No cases processed yet.")
        else:
            _metrics = get_dashboard_metrics(_sidebar_log)
            st.markdown(
                f"""
                <div class="health-grid">
                  <div class="health-stat"><span>Cases</span><b>{_metrics["total"]}</b></div>
                  <div class="health-stat"><span>Review</span><b>{_metrics["human_review"]}</b></div>
                  <div class="health-stat"><span>Escalated</span><b>{_metrics["escalated"]}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception:
        st.caption("Queue health unavailable.")

    st.markdown(
        '<p class="sidebar-footnote">AI-assisted triage for shared-services ops: classify, route, remediate, and audit every case.</p>',
        unsafe_allow_html=True,
    )
    return str(st.session_state.ops_active_section)


with st.sidebar:
    section = render_sidebar()

render_shell_header()

if section == "Single Request Intake":
    st.header("Create intake case")
    st.markdown(
        "Create a new operations ticket from a customer email, web-form message, or shared-inbox note. "
        "The user never selects the category manually — the workflow classifies and routes it."
    )

    case_form_col, playbook_col = st.columns([1.45, 0.8])
    with playbook_col:
        render_playbook_card()
    with case_form_col:
        st.markdown("#### Intake details")

        c_a, c_b = st.columns(2)
        with c_a:
            requester_name = st.text_input("Requester name", value="Demo User", key="sr_name")
            account_id = st.text_input("Account ID (optional)", value="", key="sr_account")
        with c_b:
            requester_email = st.text_input("Requester email", value="demo@example.com", key="sr_email")
            channel = st.selectbox(
                "Channel",
                ["web_form", "email", "shared_inbox", "chat", "phone_note"],
                index=0,
                key="sr_channel",
            )

        request_text = st.text_area(
            "Incoming request text (auto-classified)",
            height=160,
            placeholder="Paste any customer email / form message here…",
            key="sr_text",
        )

        if st.button("Create case and run workflow", type="primary", key="process_single", width="stretch"):
            if not str(request_text or "").strip():
                st.error("Please paste request text first.")
            else:
                classification, workflow, audit = process_one(
                    requester_name=requester_name or "Demo User",
                    requester_email=requester_email or "",
                    request_text=request_text,
                    account_id=account_id or "",
                    channel=channel,
                )
                st.session_state["last_single_result"] = (classification, workflow, audit)

    if "last_single_result" in st.session_state:
        classification, workflow, audit = st.session_state["last_single_result"]
        render_result(classification, workflow, audit)

elif section == "File Upload / Batch":
    st.header("File upload / batch processing")
    st.write(
        "Upload **CSV, TXT, PDF, or DOCX**. CSV should include a request text column "
        "(e.g. `request_text`). After processing, select any Case ID below to open the full remediation view."
    )
    uploaded = st.file_uploader("Upload file", type=["csv", "txt", "pdf", "docx"], key="batch_upload")
    if uploaded:
        try:
            records = parse_uploaded_file(uploaded, uploaded.name)
            st.success(f"Parsed {len(records)} request(s) from `{uploaded.name}`.")
            st.dataframe(pd.DataFrame(records), width="stretch")
            if st.button("Process all parsed requests", type="primary", key="process_batch"):
                outputs = []
                for rec in records:
                    classification, workflow, audit = process_one(
                        requester_name=rec.get("requester_name", "Unknown"),
                        requester_email=rec.get("requester_email", ""),
                        request_text=rec["request_text"],
                        account_id=rec.get("account_id", ""),
                        channel=rec.get("channel", "file_upload"),
                    )
                    outputs.append(case_dict_from_live(classification, workflow, audit))
                st.session_state["last_batch_cases"] = outputs
                st.session_state["last_batch_df"] = pd.DataFrame(outputs)

            if "last_batch_df" in st.session_state:
                out_df = st.session_state["last_batch_df"]
                summary_cols = [
                    c
                    for c in [
                        "request_id",
                        "requester_name",
                        "request_type",
                        "urgency",
                        "confidence",
                        "status",
                        "routing_destination",
                        "sla_or_follow_up",
                        "escalation_flag",
                        "human_review_required",
                    ]
                    if c in out_df.columns
                ]
                st.dataframe(out_df[summary_cols] if summary_cols else out_df, width="stretch")
                st.download_button(
                    "Download processed CSV",
                    out_df.to_csv(index=False),
                    "processed_requests.csv",
                    "text/csv",
                    key="batch_download",
                )

                cases = st.session_state.get("last_batch_cases") or out_df.to_dict(orient="records")
                if cases:
                    labels = {
                        f"{c.get('request_id')} · {c.get('request_type')} · {c.get('status')}": c for c in cases
                    }
                    pick = st.selectbox(
                        "Choose a batch case",
                        list(labels.keys()),
                        key="batch_case_pick",
                    )
                    show = st.checkbox(
                        "Show full remediation detail for selected batch case",
                        value=False,
                        key="batch_show_detail",
                    )
                    if show:
                        render_case_detail(labels[pick], title="Batch case detail")
                    else:
                        st.caption("Summary table shown above. Enable the checkbox to open one full case.")
        except Exception as exc:
            st.error(f"Could not parse file: {exc}")

elif section == "Simulated Inbox":
    st.header("Simulated inbox")
    st.markdown(
        "Demo-only sample messages for when you have no live data. "
        "Process one or all — the system **auto-classifies** each message. "
        "Every processed item is written to **Dashboard / Audit Log**."
    )
    inbox = get_inbox_samples()
    demo_labels = get_sample_label_map()
    preview = pd.DataFrame(
        [
            {
                "inbox_id": i["inbox_id"],
                "demo_intent": demo_labels.get(i["inbox_id"], ""),
                "from": i["requester_name"],
                "email": i["requester_email"],
                "channel": i["channel"],
                "subject": i["subject"],
                "preview": i["request_text"][:80] + ("…" if len(i["request_text"]) > 80 else ""),
            }
            for i in inbox
        ]
    )
    st.dataframe(preview, width="stretch", hide_index=True)

    choices = {f"{i['inbox_id']} — {i['subject']}": i for i in inbox}
    selected_label = st.selectbox(
        "Select one inbox item to preview / process",
        list(choices.keys()),
        key="inbox_item_select",
    )
    selected = choices[selected_label]

    st.markdown("**Message body**")
    st.code(selected["request_text"], language="text")

    col_x, col_y = st.columns(2)
    with col_x:
        process_one_clicked = st.button(
            "Process selected message", type="primary", key="process_inbox_one"
        )
    with col_y:
        process_all_clicked = st.button("Process all inbox messages", key="process_inbox_all")

    if process_one_clicked:
        classification, workflow, audit = process_one(
            requester_name=selected["requester_name"],
            requester_email=selected["requester_email"],
            request_text=selected["request_text"],
            account_id=selected.get("account_id", ""),
            channel=selected.get("channel", "shared_inbox"),
        )
        st.session_state["last_inbox_result"] = (classification, workflow, audit)
        st.session_state.pop("last_inbox_batch", None)
        st.session_state.pop("last_inbox_cases", None)
        st.session_state.pop("inbox_batch_case_pick", None)

    if process_all_clicked:
        rows = []
        cases = []
        for item in inbox:
            classification, workflow, audit = process_one(
                requester_name=item["requester_name"],
                requester_email=item["requester_email"],
                request_text=item["request_text"],
                account_id=item.get("account_id", ""),
                channel=item.get("channel", "shared_inbox"),
            )
            case = case_dict_from_live(classification, workflow, audit)
            case["inbox_id"] = item["inbox_id"]
            cases.append(case)
            rows.append(
                {
                    "inbox_id": item["inbox_id"],
                    "request_id": workflow.request_id,
                    "request_type": classification.request_type,
                    "urgency": classification.urgency,
                    "confidence": round(classification.confidence, 2),
                    "status": workflow.status,
                    "routed_team": workflow.routed_team,
                    "escalation_flag": workflow.escalation_flag,
                    "human_review_required": workflow.human_review_required,
                    "processing_mode": workflow.processing_mode,
                }
            )
        st.session_state["last_inbox_batch"] = pd.DataFrame(rows)
        st.session_state["last_inbox_cases"] = cases
        st.session_state.pop("last_inbox_result", None)
        st.session_state["inbox_show_detail"] = False

    if "last_inbox_result" in st.session_state and "last_inbox_batch" not in st.session_state:
        classification, workflow, audit = st.session_state["last_inbox_result"]
        render_result(classification, workflow, audit)

    if "last_inbox_batch" in st.session_state:
        st.dataframe(st.session_state["last_inbox_batch"], width="stretch", hide_index=True)
        st.success(
            f"Processed {len(st.session_state['last_inbox_batch'])} inbox messages into the audit trail. "
            "Use the selector below only when you want one case’s full remediation detail."
        )
        cases = st.session_state.get("last_inbox_cases") or []
        if cases:
            labels = {
                f"{c.get('inbox_id')} · {c.get('request_id')} · {c.get('request_type')}": c for c in cases
            }
            pick = st.selectbox(
                "Choose a case",
                list(labels.keys()),
                key="inbox_batch_case_pick",
            )
            show = st.checkbox("Show full remediation detail for selected case", value=False, key="inbox_show_detail")
            if show:
                render_case_detail(labels[pick], title="Inbox case detail")
            else:
                st.caption("Detail hidden to avoid repeating the summary. Tick the checkbox to expand one case.")

elif section == "Webhook Intake":
    st.header("Local webhook intake")
    st.markdown(
        "This demo endpoint shows how an email/form/shared-inbox system could **push** requests into the same "
        "classify → remediation → audit pipeline.\n\n"
        "Start the API in another terminal:\n"
        "`uvicorn api:app --reload --port 8000`"
    )

    wh_name = st.text_input("Requester name", value="Webhook Demo", key="wh_name")
    wh_email = st.text_input("Requester email", value="webhook@example.com", key="wh_email")
    wh_account = st.text_input("Account ID (optional)", value="", key="wh_account")
    wh_channel = st.selectbox(
        "Channel",
        ["webhook", "email", "web_form", "shared_inbox"],
        index=0,
        key="wh_channel",
    )
    wh_text = st.text_area(
        "request_text",
        height=140,
        key="wh_text",
        value="Our service is down during launch day. I need a supervisor immediately or we will escalate this legally.",
    )

    c1, c2 = st.columns(2)
    with c1:
        use_http = st.checkbox("Send via local HTTP webhook (port 8000)", value=False, key="wh_use_http")
    with c2:
        st.caption("If HTTP is off, the request is processed in-process (same code path).")

    if st.button("Send to webhook pipeline", type="primary", key="wh_send"):
        if not str(wh_text or "").strip():
            st.error("request_text is required.")
        elif use_http:
            try:
                import urllib.request

                body = json.dumps(
                    {
                        "request_text": wh_text,
                        "requester_name": wh_name,
                        "requester_email": wh_email,
                        "account_id": wh_account,
                        "channel": wh_channel,
                        "source": "streamlit_webhook_tester",
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:8000/webhook/incoming",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                st.session_state["last_webhook_payload"] = payload
                st.success("Webhook accepted and processed.")
            except Exception as exc:
                st.error(
                    f"Could not reach local webhook API: {exc}. "
                    "Start it with: uvicorn api:app --reload --port 8000"
                )
        else:
            try:
                payload = process_incoming_request(
                    request_text=wh_text,
                    requester_name=wh_name,
                    requester_email=wh_email,
                    account_id=wh_account,
                    channel=wh_channel,
                )
                st.session_state["last_webhook_payload"] = payload
                st.success("Processed via shared pipeline (in-process).")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")

    st.code(
        '''curl -X POST http://127.0.0.1:8000/webhook/incoming ^
  -H "Content-Type: application/json" ^
  -d "{\\"request_text\\":\\"Please activate my new business account.\\",\\"requester_name\\":\\"Priya\\",\\"channel\\":\\"webhook\\"}"''',
        language="bash",
    )

    if "last_webhook_payload" in st.session_state:
        payload = st.session_state["last_webhook_payload"]
        st.subheader("Webhook response")
        st.json(payload)
        case = payload.get("case") or {}
        cls = case.get("classification") or {}
        rem = case.get("remediation") or {}
        flat = {
            "request_id": case.get("request_id"),
            "timestamp": case.get("timestamp"),
            "requester_name": case.get("requester_name"),
            "requester_email": case.get("requester_email"),
            "account_id": case.get("account_id"),
            "channel": case.get("channel"),
            "request_text": case.get("request_text"),
            "request_type": cls.get("request_type"),
            "urgency": cls.get("urgency"),
            "confidence": cls.get("confidence"),
            "sub_topic": cls.get("sub_topic"),
            "rationale": cls.get("rationale"),
            "selected_branch": rem.get("selected_branch"),
            "status": rem.get("status"),
            "routing_destination": rem.get("routing_destination"),
            "sla_or_follow_up": rem.get("sla_or_follow_up"),
            "actions_triggered": rem.get("actions_triggered"),
            "generated_response": rem.get("generated_response"),
            "routing_notification": rem.get("routing_notification"),
            "human_review_required": rem.get("human_review_required"),
            "escalation_flag": rem.get("escalation_flag"),
            "processing_mode": rem.get("processing_mode") or cls.get("processing_mode"),
        }
        show = st.checkbox("Show full remediation detail", value=True, key="wh_show_detail")
        if show:
            render_case_detail(flat, title="Webhook case detail")

else:
    st.header("Dashboard / audit trail")
    st.caption("All processed requests from Single Intake, File Upload, and Simulated Inbox appear here.")

    log_df = load_case_log(500)
    if log_df.empty:
        st.info("No requests processed yet. Use Single Request, File Upload, or Simulated Inbox first.")
    else:
        metrics = get_dashboard_metrics(log_df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total processed", metrics["total"])
        c2.metric("Human review", metrics["human_review"])
        c3.metric("Escalation flags", metrics["escalated"])
        c4.metric("Critical", metrics["by_urgency"].get("Critical", 0))

        coverage = workflow_coverage_summary(log_df)
        covered = [k for k, ok in coverage.items() if ok]
        missing = [k for k, ok in coverage.items() if not ok]
        if missing:
            st.warning(
                f"Demo coverage: {len(covered)}/5 branches in log. Missing: {', '.join(missing)}. "
                "Tip: use Simulated Inbox → Process all."
            )
        else:
            st.success("Demo coverage: all 5 workflow branches are present in the case log.")

        chart_l, chart_m, chart_r = st.columns(3)
        with chart_l:
            st.subheader("By request type")
            render_horizontal_bar_chart(value_counts_series(log_df, "request_type"))
        with chart_m:
            st.subheader("By urgency")
            render_horizontal_bar_chart(value_counts_series(log_df, "urgency"))
        with chart_r:
            st.subheader("By status")
            render_horizontal_bar_chart(value_counts_series(log_df, "status"))

        st.subheader("Filter case log")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            type_opts = unique_options(log_df, "request_type")
            selected_types = st.multiselect("Request type", type_opts, default=type_opts, key="flt_types")
        with f2:
            urg_opts = unique_options(log_df, "urgency")
            selected_urg = st.multiselect("Urgency", urg_opts, default=urg_opts, key="flt_urg")
        with f3:
            status_opts = unique_options(log_df, "status")
            selected_status = st.multiselect("Status", status_opts, default=status_opts, key="flt_status")
        with f4:
            human_only = st.checkbox("Human review only", value=False, key="flt_human")
            esc_only = st.checkbox("Escalation only", value=False, key="flt_esc")
            show_details = st.checkbox("Show detail columns", value=False, key="flt_details")

        filtered = filter_case_log(
            log_df,
            request_types=selected_types,
            urgencies=selected_urg,
            statuses=selected_status,
            human_review_only=human_only,
            escalation_only=esc_only,
        )

        st.caption(f"Showing {len(filtered)} of {len(log_df)} cases")
        st.dataframe(case_log_table(filtered, include_details=show_details), width="stretch")

        st.download_button(
            "Download audit log CSV",
            export_csv_bytes(5000),
            "audit_log.csv",
            "text/csv",
            key="download_audit_csv",
        )

        st.subheader("Open case (full remediation detail)")
        _audit_case_drilldown(filtered)

    if st.button("Clear audit log", key="clear_audit"):
        clear_logs()
        for key in (
            "last_single_result",
            "last_batch_df",
            "last_batch_cases",
            "last_inbox_result",
            "last_inbox_batch",
            "last_inbox_cases",
            "last_webhook_payload",
            "inbox_show_detail",
            "batch_show_detail",
            "audit_show_detail",
            "ops_active_section",
            "workspace_section",
        ):
            st.session_state.pop(key, None)
        st.success("Audit log cleared. Switch workspace section or refresh to reload.")
