"""Backward-compatible re-exports; prefer src.logger going forward. """

from .logger import clear_logs, compute_metrics, export_logs_csv, fetch_logs, init_db, insert_log, logs_to_csv_string

__all__ = [
    "init_db",
    "insert_log",
    "fetch_logs",
    "clear_logs",
    "export_logs_csv",
    "logs_to_csv_string",
    "compute_metrics",
]
