"""Topic database: storage, exact + light semantic dedupe, and selection."""

from __future__ import annotations
import random
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from .utils import read_json, write_json_atomic


@dataclass
class Topic:
    id: str
    title: str
    category: str
    region: str
    period: str = ""
    description: str = ""   # the specific angle/hook to research
    aliases: list = field(default_factory=list)
    used: bool = False

    def to_dict(self):
        return asdict(self)


def _normalize(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    stop = {"the", "a", "an", "of", "in", "how", "its", "and", "to", "at", "on"}
    return {w for w in words if w not in stop}


def _overlap_ratio(a: str, b: str) -> float:
    sa, sb = _normalize(a), _normalize(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def load_topics(paths) -> list:
    raw = read_json(paths.topics_json, default=[])
    return [Topic(**t) for t in raw]


def save_topics(paths, topics: list) -> None:
    write_json_atomic(paths.topics_json, [t.to_dict() for t in topics])


def find_possible_duplicates(new_title: str, existing: list, threshold: float = 0.55) -> list:
    """Returns existing topics whose title/aliases overlap enough with
    new_title that a human (or the AI) should confirm they're distinct
    before adding new_title to the database."""
    hits = []
    for t in existing:
        candidates = [t.title] + t.aliases
        if any(_overlap_ratio(new_title, c) >= threshold for c in candidates):
            hits.append(t)
    return hits


def add_topic(paths, topic: Topic, *, force: bool = False):
    """Returns (added, possible_duplicates). If possible_duplicates is
    non-empty and force is False, the topic is NOT added -- surface the
    duplicates to the AI/human for a decision first."""
    topics = load_topics(paths)
    dupes = find_possible_duplicates(topic.title, topics)
    if dupes and not force:
        return False, dupes
    topics.append(topic)
    save_topics(paths, topics)
    return True, dupes


def select_next_topic(paths) -> Optional[Topic]:
    """Random selection among unused topics."""
    topics = load_topics(paths)
    unused = [t for t in topics if not t.used]
    if not unused:
        return None
    return random.choice(unused)


def mark_used(paths, topic_id: str) -> None:
    topics = load_topics(paths)
    used_log = read_json(paths.used_topics_json, default=[])
    for t in topics:
        if t.id == topic_id:
            t.used = True
            used_log.append(t.to_dict())
    save_topics(paths, topics)
    write_json_atomic(paths.used_topics_json, used_log)


def reject_topic(paths, topic_id: str, reason: str = "") -> None:
    topics = load_topics(paths)
    remaining = []
    rejected_log = read_json(paths.rejected_topics_json, default=[])
    for t in topics:
        if t.id == topic_id:
            entry = t.to_dict()
            entry["reason"] = reason
            rejected_log.append(entry)
        else:
            remaining.append(t)
    save_topics(paths, remaining)
    write_json_atomic(paths.rejected_topics_json, rejected_log)
