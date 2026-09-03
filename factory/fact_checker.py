from __future__ import annotations

import json
import os

from .utils import read_json, write_json_atomic, now_iso
from .research_engine import RESEARCH_SCHEMA_KEYS, VALID_CLASSIFICATIONS

PROMPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "prompts",
)


def run(paths, job_id, topic=None, research=None, config=None, qwen=None):
    if qwen is None:
        raise ValueError("Qwen client is required for fact checking.")

    if research is None:
        research = read_json(paths.research(job_id), {}) or {}

    template = open(
        os.path.join(PROMPTS, "fact_check.txt"),
        encoding="utf8",
    ).read()

    prompt = template.format(
        research_json=json.dumps(
            research,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"[FACT_CHECK] {job_id} | QWEN3-4B | VALIDATING DOSSIER")

    result = qwen.generate_json(
        prompt,
        max_new_tokens=2600,
        retries=3,
    )

    if not isinstance(result, dict):
        raise ValueError("Fact checker returned non-object JSON.")

    # Never allow the checker to silently delete dossier sections.
    for key in RESEARCH_SCHEMA_KEYS:
        if key not in result:
            result[key] = research.get(
                key,
                "" if key in ("topic", "overview") else [],
            )

    # Sources are evidence records; preserve the actual searched sources.
    if not result.get("sources"):
        result["sources"] = research.get("sources", [])

    for key in RESEARCH_SCHEMA_KEYS:
        value = result.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "classification" in item:
                    if item.get("classification") not in VALID_CLASSIFICATIONS:
                        item["classification"] = "uncertain"

    result["_verified_at"] = now_iso()
    result["_external_verification"] = False

    write_json_atomic(paths.verified(job_id), result)
    return result
