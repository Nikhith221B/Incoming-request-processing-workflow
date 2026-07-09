"""Dashboard metrics, filters, and case-log helpers for the Streamlit ops UI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import altair as alt
import pandas as pd
import streamlit as st

from .logger import compute_metrics, fetch_logs, logs_to_csv_string

CASE_LOG_DISPLAY_COLUMNS: List[str] = [
    "timestamp",
    "request_id",
    "requester_name",
    "channel",
    "request_type",
    "urgency",
    "confidence",
    "status",
    "routing_destination",
    "sla_or_follow_up",
    "escalation_flag",
    "human_review_required",
    "processing_mode",
]

DETAIL_COLUMNS: List[str] = [
    "request_text",
    "sub_topic",
    "rationale",
    "actions_triggered",
    "generated_response",
    "routing_notification",
]


def load_case_log(limit: int = 500) -> pd.DataFrame:
    logs = fetch_logs(limit=limit)
    if not logs:
        return pd.DataFrame()
    return pd.DataFrame(logs)


def get_dashboard_metrics(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    if df is None or df.empty:
        return compute_metrics([])
    records = df.to_dict(orient="records")
    return compute_metrics(records)


def filter_case_log(
    df: pd.DataFrame,
    request_types: Optional[Sequence[str]] = None,
    urgencies: Optional[Sequence[str]] = None,
    statuses: Optional[Sequence[str]] = None,
    human_review_only: bool = False,
    escalation_only: bool = False,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    if request_types:
        out = out[out["request_type"].isin(list(request_types))]
    if urgencies and "urgency" in out.columns:
        out = out[out["urgency"].isin(list(urgencies))]
    if statuses and "status" in out.columns:
        out = out[out["status"].isin(list(statuses))]
    if human_review_only and "human_review_required" in out.columns:
        out = out[out["human_review_required"].fillna(0).astype(int) == 1]
    if escalation_only and "escalation_flag" in out.columns:
        out = out[out["escalation_flag"].fillna(0).astype(int) == 1]
    return out


def value_counts_series(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype=int)
    return df[column].fillna("Unknown").value_counts()


def render_horizontal_bar_chart(series: pd.Series, height_per_row: int = 30) -> None:
    """Render a readable horizontal bar chart (category labels read left-to-right)."""
    if series.empty:
        st.caption("No data yet.")
        return

    chart_df = series.reset_index()
    chart_df.columns = ["label", "count"]
    chart_height = max(180, height_per_row * len(chart_df))

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#3b82f6")
        .encode(
            x=alt.X("count:Q", title="Cases", axis=alt.Axis(format="d")),
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            tooltip=["label:N", "count:Q"],
        )
        .properties(height=chart_height)
    )
    st.altair_chart(chart, width="stretch")


def case_log_table(df: pd.DataFrame, include_details: bool = False) -> pd.DataFrame:
    if df.empty:
        return df
    cols = list(CASE_LOG_DISPLAY_COLUMNS)
    if include_details:
        cols.extend(DETAIL_COLUMNS)
    present = [c for c in cols if c in df.columns]
    return df[present]


def unique_options(df: pd.DataFrame, column: str) -> List[str]:
    if df.empty or column not in df.columns:
        return []
    return sorted({str(v) for v in df[column].dropna().unique().tolist()})


def export_csv_bytes(limit: int = 5000) -> str:
    return logs_to_csv_string(limit=limit)


def workflow_coverage_summary(df: pd.DataFrame) -> Dict[str, bool]:
    """Quick demo check: which branches appear in the current log."""
    expected = [
        "Complaint",
        "General Enquiry",
        "Service Request",
        "Escalation / Urgent",
        "Needs Human Review",
    ]
    present = set(df["request_type"].dropna().unique()) if not df.empty and "request_type" in df.columns else set()
    return {name: name in present for name in expected}
