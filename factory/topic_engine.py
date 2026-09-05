from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import random
import re


# ============================================================
# HELPERS
# ============================================================

def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path, default=None):
    path = Path(path)

    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


def _topic_title(value):
    """
    Safely extract a topic title from:
      - Topic object
      - dict
      - string
    """
    if isinstance(value, Topic):
        return str(value.title).strip()

    if isinstance(value, dict):
        return str(
            value.get("title")
            or value.get("topic")
            or value.get("name")
            or ""
        ).strip()

    if isinstance(value, str):
        return value.strip()

    return ""


def _topic_category(value):
    """
    Safely extract a category.
    """
    if isinstance(value, Topic):
        return str(value.category or "history").strip()

    if isinstance(value, dict):
        return str(
            value.get("category")
            or value.get("type")
            or value.get("field")
            or "history"
        ).strip()

    return "history"


def _topic_description(value):
    """
    Safely extract a description.
    """
    if isinstance(value, Topic):
        return str(value.description or "").strip()

    if isinstance(value, dict):
        return str(
            value.get("description")
            or value.get("summary")
            or value.get("details")
            or ""
        ).strip()

    return ""


# ============================================================
# TOPIC OBJECT
# ============================================================

@dataclass
class Topic:
    """
    Topic passed through the Qwen pipeline.

    The category field is intentionally included because
    research_engine.py uses topic.category.
    """

    title: str
    category: str = "history"
    description: str = ""

    # Preserve any additional metadata from topics.json.
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.title = str(self.title or "").strip()

        self.category = str(
            self.category or "history"
        ).strip()

        self.description = str(
            self.description or ""
        ).strip()

        if not isinstance(self.metadata, dict):
            self.metadata = {}

    def to_dict(self):
        data = dict(self.metadata)

        data.update(
            {
                "title": self.title,
                "category": self.category,
                "description": self.description,
            }
        )

        return data


# ============================================================
# TOPIC CONVERSION
# ============================================================

def _make_topic(value):
    """
    Convert a topics.json entry into a Topic object.

    Supported input formats:

        "Ewe Drumming and Dance Tradition"

    or:

        {
            "title": "Ewe Drumming and Dance Tradition",
            "category": "music",
            "description": "..."
        }
    """

    if isinstance(value, Topic):
        return value

    if isinstance(value, str):
        return Topic(
            title=value,
            category="history",
            description="",
        )

    if isinstance(value, dict):
        title = (
            value.get("title")
            or value.get("topic")
            or value.get("name")
        )

        if not title:
            return None

        category = (
            value.get("category")
            or value.get("type")
            or value.get("field")
            or "history"
        )

        description = (
            value.get("description")
            or value.get("summary")
            or value.get("details")
            or ""
        )

        metadata = dict(value)

        return Topic(
            title=title,
            category=category,
            description=description,
            metadata=metadata,
        )

    return None


# ============================================================
# PATH HELPERS
# ============================================================

def _topics_path(paths):
    return Path(paths.root) / "01_TOPICS" / "topics.json"


def _used_path(paths):
    return Path(paths.root) / "01_TOPICS" / "used.json"


def _claimed_path(paths):
    return Path(paths.root) / "01_TOPICS" / "claimed.json"


def _rejected_path(paths):
    return Path(paths.root) / "01_TOPICS" / "rejected.json"


# ============================================================
# JOB ID
# ============================================================

def _next_job_id(paths):
    """
    Find the next BH###### job ID.

    Existing jobs are scanned so Colab restarts never cause
    the job number to reset.
    """

    jobs_dir = Path(paths.root) / "02_JOBS"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    highest = 0

    for item in jobs_dir.iterdir():
        if not item.is_dir():
            continue

        match = re.fullmatch(
            r"BH(\d{6})",
            item.name,
            flags=re.IGNORECASE,
        )

        if match:
            highest = max(
                highest,
                int(match.group(1)),
            )

    return f"BH{highest + 1:06d}"


# ============================================================
# RESUMABLE JOBS
# ============================================================

RESUMABLE_STATUSES = {
    "QWEN_RESEARCHING",
    "FACT_CHECKING",
    "QWEN_FACT_CHECK",
    "QWEN_NARRATION",
    "QWEN_SCENE_PLANNING",
    "SCENE_PLANNING",
    "QWEN_ERROR",
    "FAILED",
    "ERROR",
}


def _manifest_status(manifest):
    if not isinstance(manifest, dict):
        return ""

    return str(
        manifest.get("status")
        or manifest.get("stage")
        or ""
    ).upper().strip()


def find_resumable_job(paths):
    """
    Find the oldest incomplete Qwen job.

    Returns:
        (Topic, job_id)

    or:

        (None, None)
    """

    jobs_dir = Path(paths.root) / "02_JOBS"

    if not jobs_dir.exists():
        return None, None

    candidates = []

    for job_dir in jobs_dir.iterdir():
        if not job_dir.is_dir():
            continue

        manifest_path = job_dir / "job.json"

        if not manifest_path.exists():
            continue

        manifest = _read_json(
            manifest_path,
            {},
        )

        if not isinstance(manifest, dict):
            continue

        status = _manifest_status(manifest)

        if status not in RESUMABLE_STATUSES:
            continue

        topic_data = manifest.get("topic")

        if topic_data is None:
            topic_data = {
                "title": manifest.get("title", ""),
                "category": manifest.get(
                    "category",
                    "history",
                ),
                "description": manifest.get(
                    "description",
                    "",
                ),
            }

        topic = _make_topic(topic_data)

        if topic is None or not topic.title:
            continue

        created = (
            manifest.get("created_at")
            or manifest.get("claimed_at")
            or ""
        )

        candidates.append(
            (
                created,
                job_dir.name,
                topic,
            )
        )

    if not candidates:
        return None, None

    candidates.sort(
        key=lambda item: (
            item[0] or "",
            item[1],
        )
    )

    _, job_id, topic = candidates[0]

    # Make absolutely sure the complete job tree exists.
    paths.prepare_job(job_id)

    return topic, job_id


# ============================================================
# CLAIM NEXT TOPIC
# ============================================================

def claim_next_topic(paths):
    """
    Randomly claim an unused topic.

    Returns:
        (Topic, job_id)

    or:

        (None, None)
    """

    paths.ensure_tree()

    topics_data = _read_json(
        _topics_path(paths),
        [],
    )

    used_data = _read_json(
        _used_path(paths),
        [],
    )

    claimed_data = _read_json(
        _claimed_path(paths),
        {},
    )

    rejected_data = _read_json(
        _rejected_path(paths),
        [],
    )

    # --------------------------------------------------------
    # Normalize tracking data
    # --------------------------------------------------------

    if not isinstance(topics_data, list):
        topics_data = []

    if not isinstance(used_data, list):
        used_data = []

    if not isinstance(claimed_data, dict):
        claimed_data = {}

    if not isinstance(rejected_data, list):
        rejected_data = []

    used_titles = {
        _topic_title(item).lower()
        for item in used_data
        if _topic_title(item)
    }

    rejected_titles = {
        _topic_title(item).lower()
        for item in rejected_data
        if _topic_title(item)
    }

    claimed_titles = set()

    for item in claimed_data.values():
        if isinstance(item, dict):
            title = _topic_title(
                item.get("topic", item)
            )
        else:
            title = _topic_title(item)

        if title:
            claimed_titles.add(
                title.lower()
            )

    # --------------------------------------------------------
    # Convert all available topics
    # --------------------------------------------------------

    available = []

    for raw_topic in topics_data:
        topic = _make_topic(raw_topic)

        if topic is None:
            continue

        if not topic.title:
            continue

        title_key = topic.title.lower()

        if title_key in used_titles:
            continue

        if title_key in rejected_titles:
            continue

        if title_key in claimed_titles:
            continue

        available.append(topic)

    if not available:
        return None, None

    # --------------------------------------------------------
    # RANDOM TOPIC
    # --------------------------------------------------------

    topic = random.choice(available)

    # --------------------------------------------------------
    # CREATE JOB
    # --------------------------------------------------------

    job_id = _next_job_id(paths)

    # This is deliberately done BEFORE writing the manifest.
    # It prevents errors such as:
    #
    # FileNotFoundError:
    # .../02_script/narration.txt
    #
    paths.prepare_job(job_id)

    # --------------------------------------------------------
    # Job manifest
    # --------------------------------------------------------

    now = _now()

    manifest = {
        "job_id": job_id,
        "title": topic.title,
        "category": topic.category,
        "description": topic.description,
        "topic": topic.to_dict(),
        "status": "QWEN_RESEARCHING",
        "created_at": now,
        "claimed_at": now,
        "updated_at": now,
    }

    _write_json(
        Path(paths.root)
        / "02_JOBS"
        / job_id
        / "job.json",
        manifest,
    )

    # --------------------------------------------------------
    # Claimed topic record
    # --------------------------------------------------------

    claimed_data[job_id] = {
        "job_id": job_id,
        "topic": topic.to_dict(),
        "title": topic.title,
        "category": topic.category,
        "claimed_at": now,
    }

    _write_json(
        _claimed_path(paths),
        claimed_data,
    )

    return topic, job_id


# ============================================================
# MARK USED
# ============================================================

def mark_used(paths, job_id, topic=None):
    """
    Mark a topic as successfully used.

    Removes it from claimed topics and adds it to used.json.
    """

    paths.ensure_tree()

    used_data = _read_json(
        _used_path(paths),
        [],
    )

    claimed_data = _read_json(
        _claimed_path(paths),
        {},
    )

    if not isinstance(used_data, list):
        used_data = []

    if not isinstance(claimed_data, dict):
        claimed_data = {}

    # If topic wasn't supplied, recover it from claimed.json.
    if topic is None:
        record = claimed_data.get(job_id)

        if isinstance(record, dict):
            topic = _make_topic(
                record.get(
                    "topic",
                    record,
                )
            )

    if topic is None:
        manifest = _read_json(
            Path(paths.root)
            / "02_JOBS"
            / str(job_id)
            / "job.json",
            {},
        )

        if isinstance(manifest, dict):
            topic = _make_topic(
                manifest.get(
                    "topic",
                    manifest,
                )
            )

    if topic is None:
        return False

    # Avoid duplicate used entries.
    existing_titles = {
        _topic_title(item).lower()
        for item in used_data
        if _topic_title(item)
    }

    if topic.title.lower() not in existing_titles:
        used_data.append(
            {
                "job_id": job_id,
                "title": topic.title,
                "category": topic.category,
                "description": topic.description,
                "used_at": _now(),
            }
        )

    _write_json(
        _used_path(paths),
        used_data,
    )

    # Remove from claimed.
    claimed_data.pop(
        str(job_id),
        None,
    )

    _write_json(
        _claimed_path(paths),
        claimed_data,
    )

    # Update job manifest.
    manifest_path = (
        Path(paths.root)
        / "02_JOBS"
        / str(job_id)
        / "job.json"
    )

    manifest = _read_json(
        manifest_path,
        {},
    )

    if isinstance(manifest, dict):
        manifest["status"] = "USED"
        manifest["updated_at"] = _now()
        manifest["used_at"] = _now()

        _write_json(
            manifest_path,
            manifest,
        )

    return True


# ============================================================
# MARK REJECTED
# ============================================================

def mark_rejected(paths, job_id, topic=None, reason=""):
    """
    Mark a topic as rejected so it will not be selected again.
    """

    paths.ensure_tree()

    rejected_data = _read_json(
        _rejected_path(paths),
        [],
    )

    claimed_data = _read_json(
        _claimed_path(paths),
        {},
    )

    if not isinstance(rejected_data, list):
        rejected_data = []

    if not isinstance(claimed_data, dict):
        claimed_data = {}

    if topic is None:
        record = claimed_data.get(job_id)

        if isinstance(record, dict):
            topic = _make_topic(
                record.get(
                    "topic",
                    record,
                )
            )

    if topic is None:
        return False

    existing_titles = {
        _topic_title(item).lower()
        for item in rejected_data
        if _topic_title(item)
    }

    if topic.title.lower() not in existing_titles:
        rejected_data.append(
            {
                "job_id": job_id,
                "title": topic.title,
                "category": topic.category,
                "description": topic.description,
                "reason": reason,
                "rejected_at": _now(),
            }
        )

    _write_json(
        _rejected_path(paths),
        rejected_data,
    )

    claimed_data.pop(
        str(job_id),
        None,
    )

    _write_json(
        _claimed_path(paths),
        claimed_data,
    )

    return True
