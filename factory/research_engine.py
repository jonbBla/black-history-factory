from .utils import write_json_atomic, now_iso

VALID_CLASSIFICATIONS = {
    "established_fact",
    "archaeological_evidence",
    "scholarly_interpretation",
    "oral_tradition",
    "mythology",
    "uncertain",
}

RESEARCH_SCHEMA_KEYS = [
    "topic", "overview", "timeline", "people", "architecture",
    "technology", "daily_life", "religion", "mythology", "trade",
    "art", "lesser_known_facts", "archaeological_evidence",
    "scholarly_debates", "sources"
]

def build_prompt(topic):
    return f"""Research this topic for a historically responsible short documentary.

Topic: {topic.title}
Category: {topic.category}
Region: {topic.region}
Period: {topic.period}
Angle: {topic.description}

Return ONLY one valid JSON object with these keys:
{RESEARCH_SCHEMA_KEYS}

Every claim item must include a classification from:
{sorted(VALID_CLASSIFICATIONS)}

Never invent sources, citations, dates, people, artifacts, buildings, technologies, or quotations.
Clearly separate mythology and oral tradition from established history.
If evidence is weak or disputed, say so.
Avoid making colonization the central framing.
"""

def normalize(d, topic):
    d = d if isinstance(d, dict) else {}
    out = {}
    for k in RESEARCH_SCHEMA_KEYS:
        default = "" if k in ("topic", "overview") else []
        out[k] = d.get(k, default)
    out["topic"] = topic.title

    for key, value in out.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("classification") not in VALID_CLASSIFICATIONS:
                    item["classification"] = "uncertain"
    return out

def run(paths, job_id, topic, qwen):
    result = qwen.generate_json(
        build_prompt(topic),
        max_new_tokens=3200,
    )
    out = normalize(result, topic)
    out["_generated_at"] = now_iso()
    write_json_atomic(paths.research(job_id), out)
    write_json_atomic(paths.sources(job_id), out.get("sources", []))
    return out
