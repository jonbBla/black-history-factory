"""Phase C — real implementation.

Contract (unchanged from the Phase A stub):
  input:  a Topic (see topic_engine.Topic) + a QwenClient
  output: writes paths.research_raw(job_id) with keys: topic, overview,
    timeline, people, architecture, technology, daily_life, religion,
    mythology, trade, art, lesser_known_facts, archaeological_evidence,
    scholarly_debates, sources.

Every list item that represents a claim gets a "classification" field, one
of: established_fact, archaeological_evidence, scholarly_interpretation,
oral_tradition, mythology, uncertain. If the model omits a classification
(or returns one that isn't in this set), it defaults to "uncertain" rather
than being silently treated as established fact -- this matters a lot for
this subject matter, where oral tradition and mythology are common and
should never get flattened into "fact" by an engineering shortcut.

External source cross-checking (the source hierarchy in the spec: academic
papers, museums, universities, archaeological institutions, ...) is not
wired in yet -- that's a distinct follow-up (would need a web-search-capable
tool call available to Qwen, or a separate retrieval step before this
prompt). For now `sources` is whatever the model reports citing; treat it
as a starting point for manual verification, not a guarantee.
"""

from __future__ import annotations
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


def _normalize(data: dict, topic) -> dict:
    """Fills any keys the model omitted so downstream code never KeyErrors
    on a missing field, and normalizes classifications."""
    out = {}
    for key in RESEARCH_SCHEMA_KEYS:
        default = "" if key in ("topic", "overview") else []
        out[key] = data.get(key, default) if isinstance(data, dict) else default
    out["topic"] = out["topic"] or topic.title

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
        # Fail loudly rather than writing garbage the fact-checker would
        # then have to somehow make sense of.
        raise RuntimeError(f"Research generation failed for {topic.title}: {e}") from e

    normalized = _normalize(raw if isinstance(raw, dict) else {}, topic)
    normalized["_generated_at"] = now_iso()
    normalized["_stub"] = False
    write_json_atomic(paths.research_raw(job_id), normalized)
    return normalized
