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
    "topic",
    "overview",
    "timeline",
    "people",
    "architecture",
    "technology",
    "daily_life",
    "religion",
    "mythology",
    "trade",
    "art",
    "lesser_known_facts",
    "archaeological_evidence",
    "scholarly_debates",
    "sources",
]


def build_prompt(topic):

    return f"""
Research this topic for a historically responsible documentary.

Topic: {topic.title}
Category: {topic.category}
Region: {topic.region}
Period: {topic.period}
Angle: {topic.description}

Return ONLY one valid JSON object.

Required keys:
{RESEARCH_SCHEMA_KEYS}

For factual claims, use one of these classifications:

{sorted(VALID_CLASSIFICATIONS)}

IMPORTANT:

- Never invent sources.
- Never invent dates.
- Never invent people.
- Never invent buildings.
- Never invent archaeological discoveries.
- Never invent technologies.
- Never invent quotations.
- Never present mythology as established history.
- Clearly identify oral traditions.
- Clearly identify scholarly interpretations.
- Clearly identify uncertainty.
- Separate archaeological evidence from speculation.
- Avoid making colonization the central framing.
- Focus on the civilization, people, technology, architecture,
  art, mythology, practices, achievements and historical context.

If reliable evidence for something cannot be established,
say that it is uncertain rather than guessing.
"""


def normalize(data, topic):

    data = data if isinstance(data, dict) else {}

    output = {}

    for key in RESEARCH_SCHEMA_KEYS:

        if key in ("topic", "overview"):
            default = ""
        else:
            default = []

        output[key] = data.get(key, default)

    output["topic"] = topic.title

    for key, value in output.items():

        if not isinstance(value, list):
            continue

        for item in value:

            if not isinstance(item, dict):
                continue

            classification = item.get(
                "classification"
            )

            if classification not in VALID_CLASSIFICATIONS:
                item["classification"] = "uncertain"

    return output


def run(paths, job_id, topic, qwen):

    result = qwen.generate_json(
        build_prompt(topic),
        max_new_tokens=3000,
    )

    output = normalize(
        result,
        topic
    )

    output["_generated_at"] = now_iso()

    write_json_atomic(
        paths.research(job_id),
        output
    )

    write_json_atomic(
        paths.sources(job_id),
        output.get("sources", [])
    )

    return output
