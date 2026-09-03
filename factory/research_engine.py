from __future__ import annotations

import json
import os
import re
from urllib.parse import quote_plus, urljoin

import requests

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
    "topic", "overview", "timeline", "people", "architecture", "technology",
    "daily_life", "religion", "mythology", "trade", "art", "lesser_known_facts",
    "archaeological_evidence", "scholarly_debates", "sources",
]


def _request_json(url, params=None, timeout=15):
    r = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "BlackHistoryFactory/1.0"},
    )
    r.raise_for_status()
    return r.json()


def _search_wikipedia(topic, limit=6):
    """Find useful Wikipedia articles as research leads, not final authority."""
    api = "https://en.wikipedia.org/w/api.php"
    data = _request_json(
        api,
        {
            "action": "query",
            "list": "search",
            "srsearch": topic.title,
            "srlimit": limit,
            "format": "json",
            "utf8": 1,
        },
    )

    results = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title", "")
        if not title:
            continue
        results.append({
            "title": title,
            "url": "https://en.wikipedia.org/wiki/" + quote_plus(title.replace(" ", "_")),
            "snippet": re.sub(r"<[^>]+>", "", item.get("snippet", "")),
        })
    return results


def _fetch_wikipedia_extract(title):
    api = "https://en.wikipedia.org/w/api.php"
    data = _request_json(
        api,
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 0,
            "exchars": 8000,
            "titles": title,
            "format": "json",
            "utf8": 1,
        },
    )
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if page.get("extract"):
            return page["extract"]
    return ""


def _search_duckduckgo(topic, limit=6):
    """Best-effort broad web source discovery. Failure is non-fatal."""
    url = "https://html.duckduckgo.com/html/"
    try:
        r = requests.get(
            url,
            params={"q": topic.title},
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (Black History Factory research)"
            },
        )
        r.raise_for_status()
    except Exception:
        return []

    html = r.text
    results = []

    # DuckDuckGo's result markup is deliberately parsed conservatively.
    for match in re.finditer(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        re.I | re.S,
    ):
        href = match.group(1)
        title = re.sub(r"<[^>]+>", "", match.group(2))
        title = re.sub(r"\s+", " ", title).strip()
        if not href or not title:
            continue
        results.append({"title": title, "url": href})
        if len(results) >= limit:
            break

    return results


def source_search(topic):
    """Collect source leads before Qwen builds the evidence dossier."""
    material = []
    seen = set()

    try:
        wiki = _search_wikipedia(topic)
        for item in wiki:
            url = item["url"]
            if url in seen:
                continue
            seen.add(url)
            extract = _fetch_wikipedia_extract(item["title"])
            material.append({
                "title": item["title"],
                "url": url,
                "source_type": "encyclopedic_reference",
                "search_snippet": item.get("snippet", ""),
                "material": extract[:3500],
            })
    except Exception as e:
        print(f"[RESEARCH] Wikipedia search warning: {e}")

    for item in _search_duckduckgo(topic):
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        material.append({
            "title": item["title"],
            "url": item["url"],
            "source_type": "web_search_result",
            "search_snippet": "",
            "material": "",
        })

    return material[:8]


def build_prompt(topic, source_material):
    source_json = json.dumps(source_material, ensure_ascii=False, indent=2)
    return f"""
You are the evidence-dossier researcher for a historically responsible documentary.

TOPIC:
{topic.title}
Category: {topic.category}
Region: {topic.region}
Period: {topic.period}
Specific angle: {topic.description}

SOURCE SEARCH MATERIAL:
{source_json}

Build an evidence/fact dossier from the supplied source material.

RULES:
- Use the supplied sources as evidence leads.
- Do not invent sources.
- Do not invent facts that are absent from the source material.
- Distinguish established fact from archaeological evidence,
  scholarly interpretation, oral tradition, mythology and uncertainty.
- Where evidence is weak or conflicting, say so.
- Preserve source URLs in the sources field.
- Do not present oral tradition or mythology as established fact.
- Focus on the subject itself rather than colonization.

Return ONLY one JSON object with these keys:
{RESEARCH_SCHEMA_KEYS}

Every claim object that has a classification must use one of:
{sorted(VALID_CLASSIFICATIONS)}
"""


def normalize(data, topic, source_material):
    data = data if isinstance(data, dict) else {}
    out = {}
    for key in RESEARCH_SCHEMA_KEYS:
        out[key] = data.get(key, "" if key in ("topic", "overview") else [])

    out["topic"] = topic.title

    # Preserve the actual source-search results even if Qwen omits some.
    if not isinstance(out["sources"], list) or not out["sources"]:
        out["sources"] = [
            {
                "title": x.get("title", ""),
                "url": x.get("url", ""),
                "source_type": x.get("source_type", ""),
            }
            for x in source_material
        ]

    for key, value in out.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    if item.get("classification") not in VALID_CLASSIFICATIONS:
                        if "classification" in item:
                            item["classification"] = "uncertain"

    return out


def run(paths, job_id, topic, config=None, qwen=None):
    if qwen is None:
        raise ValueError("Qwen client is required for research.")

    print(f"[RESEARCH] {job_id} | SOURCE SEARCH")
    source_material = source_search(topic)
    print(f"[RESEARCH] {job_id} | SOURCES FOUND: {len(source_material)}")

    if not source_material:
        raise RuntimeError("Source search returned no usable sources.")

    print(f"[RESEARCH] {job_id} | QWEN3-4B | BUILDING EVIDENCE DOSSIER")
    result = qwen.generate_json(
        build_prompt(topic, source_material),
        max_new_tokens=3000,
        retries=3,
    )

    result = normalize(result, topic, source_material)
    result["_generated_at"] = now_iso()
    result["_source_search"] = source_material

    write_json_atomic(paths.research(job_id), result)
    write_json_atomic(paths.sources(job_id), result.get("sources", []))
    return result
