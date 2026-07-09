"""Parse uploaded CSV / TXT / PDF / DOCX into normalized request records."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

FileLike = Union[BinaryIO, bytes, str, Path]


def parse_uploaded_file(
    file_obj: Any,
    filename: str,
    default_channel: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Normalize an uploaded file into request records with keys:
      requester_name, requester_email, account_id, channel, request_text, source_file
    """
    name = (filename or "").lower()
    channel = default_channel or _channel_from_filename(name)

    if name.endswith(".csv"):
        return parse_csv(file_obj, filename=filename, channel=channel)
    if name.endswith(".txt"):
        return parse_txt(file_obj, filename=filename, channel=channel)
    if name.endswith(".pdf"):
        return parse_pdf(file_obj, filename=filename, channel=channel)
    if name.endswith(".docx"):
        return parse_docx(file_obj, filename=filename, channel=channel)
    raise ValueError(f"Unsupported file type: {filename}. Use CSV, TXT, PDF, or DOCX.")


def parse_csv(file_obj: Any, filename: str = "upload.csv", channel: str = "csv_upload") -> List[Dict[str, str]]:
    raw = _read_text(file_obj)
    reader = csv.DictReader(io.StringIO(raw))
    if not reader.fieldnames:
        raise ValueError("CSV has no header row.")

    # Accept flexible column names.
    headers = {h.strip().lower(): h for h in reader.fieldnames if h}
    text_col = _pick(headers, ["request_text", "request", "message", "body", "text", "content"])
    if not text_col:
        raise ValueError(
            "CSV must include a request text column such as request_text, message, body, or text."
        )
    name_col = _pick(headers, ["requester_name", "name", "customer_name", "from_name"])
    email_col = _pick(headers, ["requester_email", "email", "customer_email", "from_email"])
    account_col = _pick(headers, ["account_id", "account", "acct_id", "customer_id"])

    records: List[Dict[str, str]] = []
    for row in reader:
        text = (row.get(text_col) or "").strip()
        if not text:
            continue
        records.append(
            {
                "requester_name": (row.get(name_col) or "Unknown").strip() if name_col else "Unknown",
                "requester_email": (row.get(email_col) or "").strip() if email_col else "",
                "account_id": (row.get(account_col) or "").strip() if account_col else "",
                "channel": channel,
                "request_text": text,
                "source_file": filename,
            }
        )
    if not records:
        raise ValueError("No request rows found in CSV.")
    return records


def parse_txt(file_obj: Any, filename: str = "upload.txt", channel: str = "txt_upload") -> List[Dict[str, str]]:
    raw = _read_text(file_obj)
    chunks = [c.strip() for c in re.split(r"\n\s*\n+", raw) if c.strip()]
    if not chunks:
        raise ValueError("TXT file is empty.")

    # Merge metadata-only blocks with the following body block when split by blank lines.
    merged: List[str] = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        name, email, account_id, body = _extract_header_fields(chunk)
        looks_like_header_only = bool(name or email or account_id) and body.strip() == chunk.strip() and len(chunk.splitlines()) <= 4
        # If header extract left body equal to whole chunk but no long prose, try pairing with next chunk.
        if (name or email or account_id) and i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            next_name, _, _, next_body = _extract_header_fields(next_chunk)
            # Next chunk looks like body (no new name header) → merge.
            if not next_name and len(next_body.split()) >= 3:
                merged.append(chunk + "\n\n" + next_chunk)
                i += 2
                continue
        if looks_like_header_only and i + 1 < len(chunks):
            merged.append(chunk + "\n\n" + chunks[i + 1])
            i += 2
            continue
        merged.append(chunk)
        i += 1

    records: List[Dict[str, str]] = []
    for idx, chunk in enumerate(merged, start=1):
        name, email, account_id, body = _extract_header_fields(chunk)
        records.append(
            {
                "requester_name": name or f"TXT Requester {idx}",
                "requester_email": email,
                "account_id": account_id,
                "channel": channel,
                "request_text": body,
                "source_file": filename,
            }
        )
    return records


def parse_pdf(file_obj: Any, filename: str = "upload.pdf", channel: str = "pdf_upload") -> List[Dict[str, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install pypdf to parse PDF uploads: pip install pypdf") from exc

    data = _read_bytes(file_obj)
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError("Could not extract text from PDF.")

    # Treat blank-page/line breaks as multiple requests when present; else one request.
    chunks = [c.strip() for c in re.split(r"\n\s*\n+", text) if c.strip()]
    if len(chunks) <= 1:
        chunks = [text]

    records: List[Dict[str, str]] = []
    for i, chunk in enumerate(chunks, start=1):
        name, email, account_id, body = _extract_header_fields(chunk)
        records.append(
            {
                "requester_name": name or f"PDF Requester {i}",
                "requester_email": email,
                "account_id": account_id,
                "channel": channel,
                "request_text": body,
                "source_file": filename,
            }
        )
    return records


def parse_docx(file_obj: Any, filename: str = "upload.docx", channel: str = "docx_upload") -> List[Dict[str, str]]:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install python-docx to parse DOCX uploads: pip install python-docx") from exc

    data = _read_bytes(file_obj)
    doc = Document(io.BytesIO(data))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    if not paragraphs:
        raise ValueError("DOCX file has no text content.")

    # Split on blank paragraphs already removed — group by double newlines in joined text.
    joined = "\n".join(paragraphs)
    chunks = [c.strip() for c in re.split(r"\n\s*\n+", joined) if c.strip()]
    if not chunks:
        chunks = ["\n".join(paragraphs)]

    records: List[Dict[str, str]] = []
    for i, chunk in enumerate(chunks, start=1):
        name, email, account_id, body = _extract_header_fields(chunk)
        records.append(
            {
                "requester_name": name or f"DOCX Requester {i}",
                "requester_email": email,
                "account_id": account_id,
                "channel": channel,
                "request_text": body,
                "source_file": filename,
            }
        )
    return records


def _pick(headers: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in headers:
            return headers[c]
    return None


def _channel_from_filename(name: str) -> str:
    if name.endswith(".csv"):
        return "csv_upload"
    if name.endswith(".txt"):
        return "txt_upload"
    if name.endswith(".pdf"):
        return "pdf_upload"
    if name.endswith(".docx"):
        return "docx_upload"
    return "file_upload"


def _read_bytes(file_obj: Any) -> bytes:
    if isinstance(file_obj, (bytes, bytearray)):
        return bytes(file_obj)
    if isinstance(file_obj, Path):
        return file_obj.read_bytes()
    if isinstance(file_obj, str):
        # Path string vs raw text — if it exists as a path, read; else treat as text.
        p = Path(file_obj)
        if p.exists() and p.is_file():
            return p.read_bytes()
        return file_obj.encode("utf-8")
    if hasattr(file_obj, "read"):
        data = file_obj.read()
        if isinstance(data, str):
            return data.encode("utf-8")
        return data
    raise TypeError("Unsupported file object for binary read.")


def _read_text(file_obj: Any) -> str:
    if isinstance(file_obj, str) and not Path(file_obj).exists():
        return file_obj
    data = _read_bytes(file_obj)
    return data.decode("utf-8", errors="replace")


def _extract_header_fields(chunk: str) -> tuple[str, str, str, str]:
    """
    Optional leading metadata lines:
      Name: ...
      Email: ...
      Account: ...
    Remaining text is the body.
    """
    lines = chunk.splitlines()
    name = ""
    email = ""
    account_id = ""
    body_start = 0
    for i, line in enumerate(lines[:8]):
        lower = line.strip().lower()
        if lower.startswith("name:"):
            name = line.split(":", 1)[1].strip()
            body_start = i + 1
        elif lower.startswith("email:"):
            email = line.split(":", 1)[1].strip()
            body_start = i + 1
        elif lower.startswith("account:") or lower.startswith("account id:"):
            account_id = line.split(":", 1)[1].strip()
            body_start = i + 1
        elif lower.startswith("from:"):
            # From: Name <email@x.com>
            from_val = line.split(":", 1)[1].strip()
            m = re.search(r"([^<]+)<([^>]+)>", from_val)
            if m:
                name = name or m.group(1).strip()
                email = email or m.group(2).strip()
            else:
                name = name or from_val
            body_start = i + 1
        else:
            # Stop if we hit non-metadata content after scanning consecutive headers.
            if body_start > 0 and i >= body_start:
                break
            if not any(lower.startswith(p) for p in ("name:", "email:", "account", "from:")):
                break

    body = "\n".join(lines[body_start:]).strip() or chunk.strip()
    return name, email, account_id, body
