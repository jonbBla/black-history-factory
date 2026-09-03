from __future__ import annotations

import os

from .utils import read_json, write_json_atomic, now_iso


def _maybe_publish(paths, data):
    try:
        import time
        cfg = read_json(paths("00_CONFIG", "config.json"), {}) or {}
        repo = cfg.get("github_repo", "")
        token = os.environ.get("GH_TOKEN", "")
        if not repo or not token:
            return
        last = float(data.get("_last_github_push", 0) or 0)
        interval = float(cfg.get("github_update_seconds", 60))
        if time.time() - last < interval:
            return
        from .github import push_status
        history = read_json(paths.status_history, []) or []
        config_obj = type(
            "C",
            (),
            {
                "github_repo": repo,
                "github_dashboard_path": cfg.get(
                    "github_dashboard_path",
                    "dashboard/data",
                ),
            },
        )()
        push_status(config_obj, data, history, token)
        data["_last_github_push"] = time.time()
        write_json_atomic(paths.status_current, data)
    except Exception as e:
        data.setdefault("warnings", []).append(
            f"GitHub publish warning: {e}"
        )


def set_processor(
    paths,
    name,
    status,
    job_id="",
    stage="",
    detail="",
    completed=0,
    total=0,
):
    data = read_json(paths.status_current, {}) or {}
    data.setdefault("processors", {})[name] = {
        "status": status,
        "job_id": job_id,
        "stage": stage,
        "detail": detail,
        "completed": completed,
        "total": total,
        "percent": round((completed / total) * 100, 1) if total else 0,
        "updated_at": now_iso(),
    }
    data["updated_at"] = now_iso()
    write_json_atomic(paths.status_current, data)
    _maybe_publish(paths, data)
    return data


def update_status(paths, job_id, stage, message=""):
    """Compatibility API for older callers such as qwen_pipeline."""
    return set_processor(
        paths,
        "qwen",
        "running" if stage != "error" else "error",
        job_id=job_id,
        stage=stage,
        detail=message,
    )


def append_history(paths, job_id, title, video_path, thumbnail_path, seconds=None):
    history = read_json(paths.status_history, []) or []
    history = [x for x in history if x.get("job_id") != job_id]
    item = {
        "job_id": job_id,
        "title": title,
        "video": video_path,
        "thumbnail": thumbnail_path,
        "date": now_iso(),
    }
    if seconds is not None:
        item["seconds"] = seconds
    history.insert(0, item)
    write_json_atomic(paths.status_history, history)
    return history


def mark_idle(paths, name="qwen"):
    return set_processor(paths, name, "idle")
