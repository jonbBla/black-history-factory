"""Real research engine implementation.

Contract:
  input:  a Topic (see topic_engine.Topic) + a QwenClient
  output: writes paths.research_raw(job_id) with keys: topic, overview,
    timeline, people, architecture, technology, daily_life, religion,
    mythology, trade, art, lesser_known_facts, archaeological_evidence,
    scholarly_debates, sources.

Every list item that represents a claim gets a "classification" field, one
of: established_fact, archaeological_evidence, scholarly_interpretation,
oral_tradition, mythology, uncertain. If the model omits a classification
(or returns one that isn't in this set), it defaults to "uncertain" rather
than being silently treated as established fact.

External source cross-checking (the source hierarchy in the spec: academic
papers, museums, universities, archaeological institutions, ...) is not
wired in yet -- `sources` is currently whatever the model reports citing,
which needs human spot-checking until a retrieval step is added.
"""

from __future__ import annotations
import json
import os
from .utils import write_json_atomic, now_iso

RESEARCH_SCHEMA_KEYS = [
    "topic", "overview", "timeline", "people", "architecture", "technology",
    "daily_life", "religion", "mythology", "trade", "art",
    "lesser_known_facts", "archaeological_evidence", "scholarly_debates",
    "sources",
]

VALID_CLASSIFICATIONS = {
    "established_fact", "archaeological_evidence", "scholarly_interpretation",
    "oral_tradition", "mythology", "uncertain",
}

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")


def _load_template(name: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(topic) -> str:
    template = _load_template("research.txt")
    return template.format(
        topic_title=topic.title,
        category=topic.category,
        region=topic.region,
        period=topic.period,
        description=getattr(topic, "description", "") or "(no additional angle specified)",
    )


def _as_text(value) -> str:
    """Coerce a value that's supposed to be plain text into an actual
    string, even if Qwen returned a nested object or list instead of a
    plain string for a "text" field -- this happens in practice (the model
    doesn't always respect the schema), and letting a dict/list silently
    flow downstream as "text" causes hard-to-diagnose crashes later (e.g.
    slicing a dict raises a confusing KeyError, not a clear TypeError, on
    Python 3.12+). Better to flatten it to readable text once, here, than
    let every downstream consumer defend against it separately."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _normalize(data: dict, topic) -> dict:
    """Fills any keys the model omitted so downstream code never KeyErrors
    on a missing field, normalizes classifications, and coerces "overview"
    to an actual string regardless of what shape the model returned it in."""
    out = {}
    for key in RESEARCH_SCHEMA_KEYS:
        default = "" if key in ("topic", "overview") else []
        out[key] = data.get(key, default) if isinstance(data, dict) else default
    out["topic"] = topic.title  # authoritative from the Topic object -- no need to trust the model's echo of it
    out["overview"] = _as_text(out["overview"])

    for key in RESEARCH_SCHEMA_KEYS:
        if key in ("topic", "overview", "sources"):
            continue
        items = out[key]
        if not isinstance(items, list):
            out[key] = []
            continue
        for item in items:
            if isinstance(item, dict):
                if item.get("classification") not in VALID_CLASSIFICATIONS:
                    item["classification"] = "uncertain"
    return out


def run(paths, job_id: str, topic, qwen=None) -> dict:
    """qwen: a QwenClient (see qwen_client.py), loaded once in Colab Cell 4.
    If qwen is None, falls back to a clearly-labeled placeholder so the rest
    of the pipeline (checkpointing, resume, scene planning, etc.) can still
    be exercised and tested without a loaded model."""
    if qwen is None:
        placeholder = _normalize({}, topic)
        placeholder["overview"] = f"[NO MODEL LOADED] Research not generated for '{topic.title}'."
        placeholder["_generated_at"] = now_iso()
        placeholder["_stub"] = True
        write_json_atomic(paths.research_raw(job_id), placeholder)
        return placeholder

    prompt = build_prompt(topic)
    try:
        raw = qwen.generate_json(prompt, max_new_tokens=3000)
    except ValueError as e:
        raise RuntimeError(f"Research generation failed for {topic.title}: {e}") from e

    normalized = _normalize(raw if isinstance(raw, dict) else {}, topic)
    normalized["_generated_at"] = now_iso()
    normalized["_stub"] = False
    write_json_atomic(paths.research_raw(job_id), normalized)
    return normalized
