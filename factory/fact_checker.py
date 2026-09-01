from __future__ import annotations
import json, os
from .utils import read_json, write_json_atomic, now_iso
from .research_engine import RESEARCH_SCHEMA_KEYS, VALID_CLASSIFICATIONS

PROMPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")

def run(paths, job_id, qwen):
    raw = read_json(paths.research(job_id), {}) or {}
    template = open(os.path.join(PROMPTS, "fact_check.txt"), encoding="utf8").read()
    prompt = template.format(research_json=json.dumps(raw, indent=2, ensure_ascii=False))
    result = qwen.generate_json(prompt, max_new_tokens=2200)
    if not isinstance(result, dict):
        raise ValueError("Fact checker returned non-object JSON")
    for key, value in raw.items():
        result.setdefault(key, value)
    for key in RESEARCH_SCHEMA_KEYS:
        if isinstance(result.get(key), list):
            for item in result[key]:
                if isinstance(item, dict) and item.get("classification") not in VALID_CLASSIFICATIONS:
                    item["classification"] = "uncertain"
    result["_verified_at"] = now_iso()
    result["_external_verification"] = False
    write_json_atomic(paths.verified(job_id), result)
    return result
