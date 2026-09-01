from .utils import write_json_atomic, now_iso

VALID_CLASSIFICATIONS = {
    "established_fact", "archaeological_evidence", "scholarly_interpretation",
    "oral_tradition", "mythology", "uncertain",
}
RESEARCH_SCHEMA_KEYS = [
    "topic", "overview", "timeline", "people", "architecture", "technology",
    "daily_life", "religion", "mythology", "trade", "art", "lesser_known_facts",
    "archaeological_evidence", "scholarly_debates", "sources",
]

def build_prompt(topic):
    return f"""Research this specific topic for a historically responsible documentary.
Topic: {topic.title}
Category: {topic.category}
Region: {topic.region}
Period: {topic.period}
Angle: {topic.description}

Return ONLY one JSON object with keys: {RESEARCH_SCHEMA_KEYS}.
Every claim object must have a classification from: {sorted(VALID_CLASSIFICATIONS)}.
Never invent sources, dates, people, buildings, technologies, artifacts, quotations or archaeological discoveries.
Clearly separate established facts, archaeological evidence, scholarly interpretation, oral tradition, mythology and uncertainty.
Focus on the subject itself, not colonization."""

def normalize(data, topic):
    data = data if isinstance(data, dict) else {}
    out = {}
    for key in RESEARCH_SCHEMA_KEYS:
        out[key] = data.get(key, "" if key in ("topic", "overview") else [])
    out["topic"] = topic.title
    for value in out.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("classification") not in VALID_CLASSIFICATIONS:
                    item["classification"] = "uncertain"
    return out

def run(paths, job_id, topic, qwen):
    result = normalize(qwen.generate_json(build_prompt(topic), max_new_tokens=2400), topic)
    result["_generated_at"] = now_iso()
    write_json_atomic(paths.research(job_id), result)
    write_json_atomic(paths.sources(job_id), result.get("sources", []))
    return result
