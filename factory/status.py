"""Writes 08_STATUS/current.json and history.json, and (optionally) pushes
copies of both to the GitHub dashboard repo so the phone-facing GitHub Pages
site can show live progress. The dashboard only ever reads these two files —
it never talks to Colab directly.
"""

from __future__ import annotations
from .utils import read_json, write_json_atomic, now_iso


def update_current(paths, checkpoint, *, colab_status: str = "running") -> dict:
    total = checkpoint.total or 0
    percent = round((checkpoint.completed / total) * 100, 1) if total else 0.0
    payload = {
        "status": checkpoint.status,
        "job_id": checkpoint.job_id,
        "title": checkpoint.title,
        "stage": checkpoint.stage,
        "current": checkpoint.completed,
        "total": total,
        "percent": percent,
        "colab_status": colab_status,   # running | paused
        "error": checkpoint.error,
        "last_update": now_iso(),
    }
    write_json_atomic(paths.status_current, payload)
    return payload


def append_history(paths, *, job_id: str, title: str, video_path: str,
                    thumbnail_path: str) -> list:
    history = read_json(paths.status_history, default=[])
    history = [h for h in history if h.get("id") != job_id]
    history.insert(0, {
        "id": job_id,
        "title": title,
        "date": now_iso()[:10],
        "video": video_path,
        "thumbnail": thumbnail_path,
    })
    write_json_atomic(paths.status_history, history)
    return history


def mark_idle(paths) -> None:
    write_json_atomic(paths.status_current, {
        "status": "idle",
        "job_id": None,
        "title": None,
        "stage": None,
        "current": 0,
        "total": 0,
        "percent": 0.0,
        "colab_status": "paused",
        "error": None,
        "last_update": now_iso(),
    })
