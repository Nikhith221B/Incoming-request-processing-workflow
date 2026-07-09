"""SQLite audit trail + CSV export for processed requests."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import ClassificationResult, WorkflowResult
from .utils import utc_now_iso

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "audit_log.sqlite"
CASE_LOG_CSV = ROOT / "data" / "case_log.csv"

# Columns for stable CSV export / dashboard display.
EXPORT_COLUMNS = [
    "id",
    "timestamp",
    "request_id",
    "requester_name",
    "requester_email",
    "account_id",
    "channel",
    "request_text",
    "request_type",
    "urgency",
    "confidence",
    "sub_topic",
    "rationale",
    "selected_branch",
    "status",
    "routed_team",
    "routing_destination",
    "sla_or_follow_up",
    "actions_triggered",
    "generated_response",
    "routing_notification",
    "human_review_required",
    "escalation_flag",
    "processing_mode",
]


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                request_id TEXT,
                requester_name TEXT,
                requester_email TEXT,
                account_id TEXT,
                channel TEXT,
                request_text TEXT NOT NULL,
                request_type TEXT NOT NULL,
                urgency TEXT NOT NULL,
                confidence REAL NOT NULL,
                sub_topic TEXT,
                rationale TEXT,
                selected_branch TEXT,
                status TEXT,
                routed_team TEXT,
                routing_destination TEXT,
                sla_or_follow_up TEXT,
                actions_triggered TEXT,
                generated_response TEXT,
                routing_notification TEXT,
                human_review_required INTEGER,
                escalation_flag INTEGER,
                processing_mode TEXT
            )
            """
        )
        # Migrate older DBs created before Phase 2.
        existing = {row[1] for row in conn.execute("PRAGMA table_info(audit_log)").fetchall()}
        migrations = {
            "request_id": "TEXT",
            "account_id": "TEXT",
            "channel": "TEXT",
            "selected_branch": "TEXT",
            "routing_destination": "TEXT",
            "escalation_flag": "INTEGER",
            "processing_mode": "TEXT",
        }
        for col, col_type in migrations.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE audit_log ADD COLUMN {col} {col_type}")
        conn.commit()


def insert_log(
    requester_name: str,
    requester_email: str,
    request_text: str,
    classification: ClassificationResult,
    workflow: WorkflowResult,
    account_id: str = "",
    channel: str = "web_form",
) -> Dict[str, Any]:
    """Persist a processed case and return the stored audit record."""
    init_db()
    timestamp = workflow.received_at or utc_now_iso()
    row = {
        "timestamp": timestamp,
        "request_id": workflow.request_id,
        "requester_name": requester_name,
        "requester_email": requester_email,
        "account_id": account_id,
        "channel": channel,
        "request_text": request_text,
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
        "actions_triggered": json.dumps(workflow.actions_triggered),
        "generated_response": workflow.generated_response,
        "routing_notification": workflow.routing_notification,
        "human_review_required": 1 if workflow.human_review_required else 0,
        "escalation_flag": 1 if workflow.escalation_flag else 0,
        "processing_mode": workflow.processing_mode or classification.processing_mode,
    }

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO audit_log (
                timestamp, request_id, requester_name, requester_email, account_id, channel,
                request_text, request_type, urgency, confidence, sub_topic, rationale,
                selected_branch, status, routed_team, routing_destination, sla_or_follow_up,
                actions_triggered, generated_response, routing_notification,
                human_review_required, escalation_flag, processing_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["timestamp"],
                row["request_id"],
                row["requester_name"],
                row["requester_email"],
                row["account_id"],
                row["channel"],
                row["request_text"],
                row["request_type"],
                row["urgency"],
                row["confidence"],
                row["sub_topic"],
                row["rationale"],
                row["selected_branch"],
                row["status"],
                row["routed_team"],
                row["routing_destination"],
                row["sla_or_follow_up"],
                row["actions_triggered"],
                row["generated_response"],
                row["routing_notification"],
                row["human_review_required"],
                row["escalation_flag"],
                row["processing_mode"],
            ),
        )
        conn.commit()
        row["id"] = cur.lastrowid

    _append_case_log_csv(row)
    return row


def fetch_logs(limit: int = 100) -> List[Dict[str, Any]]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def clear_logs() -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM audit_log")
        conn.commit()
    if CASE_LOG_CSV.exists():
        CASE_LOG_CSV.unlink()


def export_logs_csv(path: Optional[Path] = None, limit: int = 5000) -> Path:
    """Write recent logs to CSV and return the path."""
    dest = path or CASE_LOG_CSV
    dest.parent.mkdir(parents=True, exist_ok=True)
    logs = fetch_logs(limit=limit)
    # Reverse so CSV is chronological oldest→newest.
    logs = list(reversed(logs))

    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for item in logs:
            writer.writerow({col: item.get(col, "") for col in EXPORT_COLUMNS})
    return dest


def logs_to_csv_string(limit: int = 5000) -> str:
    logs = list(reversed(fetch_logs(limit=limit)))
    if not logs:
        return ",".join(EXPORT_COLUMNS) + "\n"

    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for item in logs:
        writer.writerow({col: item.get(col, "") for col in EXPORT_COLUMNS})
    return buf.getvalue()


def compute_metrics(logs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    rows = logs if logs is not None else fetch_logs(5000)
    if not rows:
        return {
            "total": 0,
            "by_type": {},
            "by_urgency": {},
            "by_status": {},
            "human_review": 0,
            "escalated": 0,
        }

    by_type: Dict[str, int] = {}
    by_urgency: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    human_review = 0
    escalated = 0
    for r in rows:
        by_type[r.get("request_type") or "Unknown"] = by_type.get(r.get("request_type") or "Unknown", 0) + 1
        by_urgency[r.get("urgency") or "Unknown"] = by_urgency.get(r.get("urgency") or "Unknown", 0) + 1
        by_status[r.get("status") or "Unknown"] = by_status.get(r.get("status") or "Unknown", 0) + 1
        if r.get("human_review_required"):
            human_review += 1
        if r.get("escalation_flag"):
            escalated += 1

    return {
        "total": len(rows),
        "by_type": by_type,
        "by_urgency": by_urgency,
        "by_status": by_status,
        "human_review": human_review,
        "escalated": escalated,
    }


def _append_case_log_csv(row: Dict[str, Any]) -> None:
    CASE_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CASE_LOG_CSV.exists()
    with CASE_LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in EXPORT_COLUMNS})
