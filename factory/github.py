"""Pushes 08_STATUS/current.json and history.json from Drive to the
dashboard repo's dashboard/data/ folder, via the GitHub Contents API, so
the GitHub Pages dashboard reflects live progress without Colab needing
git installed or SSH keys -- just a fine-grained PAT with Contents:
read/write on this one repo, read from Colab's Secrets panel as GH_TOKEN.
"""

from __future__ import annotations
import base64
import json
import requests


def _put_file(repo: str, path: str, content, token: str, message: str) -> None:
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    resp = requests.get(url, headers=headers, timeout=15)
    sha = resp.json().get("sha") if resp.status_code == 200 else None

    body = {
        "message": message,
        "content": base64.b64encode(json.dumps(content, indent=2).encode("utf-8")).decode("ascii"),
    }
    if sha:
        body["sha"] = sha
    put = requests.put(url, headers=headers, json=body, timeout=15)
    put.raise_for_status()


def push_status(config, current: dict, history: list, token: str) -> None:
    if not config.github_repo or not token:
        return  # not configured -- status stays local to Drive only
    base = config.github_dashboard_path.rstrip("/")
    _put_file(config.github_repo, f"{base}/current.json", current, token,
              f"status: {current.get('stage')} ({current.get('percent')}%)")
    _put_file(config.github_repo, f"{base}/history.json", history, token,
              "update history")
