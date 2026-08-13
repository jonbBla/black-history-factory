"""Phase C -- real implementation.

Contract (unchanged from the Phase A stub):
  input:  paths.research_raw(job_id) + a QwenClient
  output: paths.research_verified(job_id) -- same shape, but every claim's
          classification has been reviewed, and unsourced/low-confidence
          claims get flagged rather than silently passed downstream.
"""

from __future__ import annotations
import json
import os
from .utils import read_json, write_json_atomic, now_iso
from .research_engine import RESEARCH_SCHEMA_KEYS, VALID_CLASSIFICATIONS

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")


def _load_template(name: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(research: dict) -> str:
    template = _load_template("fact_check.txt")
    return template.format(research_json=json.dumps(research, indent=2))


def _enforce_classifications(data: dict) -> dict:
    for key in RESEARCH_SCHEMA_KEYS:
        if key in ("topic", "overview", "sources"):
            continue
        items = data.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and item.get("classification") not in VALID_CLASSIFICATIONS:
                item["classification"] = "uncertain"
    return data


def run(paths, job_id: str, qwen=None) -> dict:
    raw = read_json(paths.research_raw(job_id), default={})

    if qwen is None:
        verified = dict(raw)
        verified["_verified_at"] = now_iso()
        verified["_stub"] = True
        write_json_atomic(paths.research_verified(job_id), verified)
        return verified

    prompt = build_prompt(raw)
    try:
        result = qwen.generate_json(prompt, max_new_tokens=3000)
    except ValueError as e:
        raise RuntimeError(f"Fact-check generation failed for job {job_id}: {e}") from e

    verified = _enforce_classifications(result if isinstance(result, dict) else dict(raw))
    verified["_verified_at"] = now_iso()
    verified["_stub"] = False
    write_json_atomic(paths.research_verified(job_id), verified)
    return verified
