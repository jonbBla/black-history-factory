"""Small shared helpers: atomic JSON I/O, ID generation, slugify, timestamps."""

from __future__ import annotations
import json
import os
import re
import tempfile
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: str, data) -> None:
    """Write JSON via a temp file + rename so a Colab crash mid-write can
    never leave a half-written status/checkpoint file behind."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dir_ = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")


def next_job_id(used_ids: list) -> str:
    """BH000001, BH000002, ... based on the highest existing id."""
    max_n = 0
    for jid in used_ids:
        m = re.match(r"BH(\d+)", jid)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"BH{max_n + 1:06d}"
