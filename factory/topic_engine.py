from pathlib import Path
import json
import random
from datetime import datetime, timezone


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path, default):
    path = Path(path)

    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _topic_title(topic):
    """
    Supports both dictionary topics and simple string topics.
    """
    if isinstance(topic, str):
        return topic.strip()

    if isinstance(topic, dict):
        return str(
            topic.get("title")
            or topic.get("topic")
            or topic.get("name")
            or ""
        ).strip()

    return str(topic).strip()


# ----------------------------------------------------------------------
# TOPIC FILES
# ----------------------------------------------------------------------

def _topics_path(paths):
    return Path(paths.root) / "01_TOPICS" / "topics.json"


def _used_path(paths):
    return Path(paths.root) / "01_TOPICS" / "used.json"


def _claimed_path(paths):
    return Path(paths.root) / "01_TOPICS" / "claimed.json"


def _rejected_path(paths):
    return Path(paths.root) / "01_TOPICS" / "rejected.json"


# ----------------------------------------------------------------------
# JOB ID
# ----------------------------------------------------------------------

def _next_job_id(paths):
    """
    Find the next available BH###### job ID.
    """
    jobs_dir = Path(paths.root) / "02_JOBS"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    highest = 0

    for item in jobs_dir.iterdir():
        if not item.is_dir():
            continue

        name = item.name.upper()

        if not name.startswith("BH"):
            continue

        number = name[2:]

        if number.isdigit():
            highest = max(highest, int(number))

    return f"BH{highest + 1:06d}"


# ----------------------------------------------------------------------
# JOB STATUS
# ----------------------------------------------------------------------

RESUMABLE_STATUSES = {
    "QWEN_RESEARCHING",
    "FACT_CHECKING",
    "QWEN_FACT_CHECK",
    "QWEN_NARRATION",
    "QWEN_SCENE_PLANNING",
    "SCENE_PLANNING",
    "QWEN_ERROR",
    "FAILED",
}


# ----------------------------------------------------------------------
# FIND RESUMABLE JOB
# ----------------------------------------------------------------------

def find_resumable_job(paths):
    """
    Find an existing Qwen job that can be resumed.

    Returns:
        (topic, job_id)

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

        manifest = _read_json(manifest_path, {})

        status = str(manifest.get("status", "")).upper()

        if status not in RESUMABLE_STATUSES:
            continue

        topic_data = manifest.get("topic")

        if topic_data is None:
            topic_data = manifest.get("topic_data")

        if topic_data is None:
            continue

        candidates.append(
            (
                manifest.get("created_at", ""),
                topic_data,
                job_dir.name,
            )
        )

    if not candidates:
        return None, None

    # Resume the oldest incomplete job first.
    candidates.sort(key=lambda x: x[0])

    _, topic_data, job_id = candidates[0]

    return _make_topic(topic_data), job_id


# ----------------------------------------------------------------------
# CLAIM NEW TOPIC
# ----------------------------------------------------------------------

def claim_next_topic(paths):
    """
    Randomly select an unused topic and create a new job.

    IMPORTANT:
    The complete job directory structure is created automatically
    through paths.prepare_job(job_id) BEFORE the job is returned.

    Returns:
        (topic, job_id)

    or:

        (None, None)
    """

    paths.ensure_tree()

    topics = _read_json(_topics_path(paths), [])
    used = _read_json(_used_path(paths), [])
    claimed = _read_json(_claimed_path(paths), {})

    if not isinstance(topics, list):
        topics = []

    if not isinstance(used, list):
        used = []

    if not isinstance(claimed, dict):
        claimed = {}

    # Normalize used topic titles.
    used_titles = {
        _topic_title(item).lower()
        for item in used
        if _topic_title(item)
    }

    # Normalize currently claimed topic titles.
    claimed_titles = {
        _topic_title(item.get("topic", item)).lower()
        for item in claimed.values()
        if _topic_title(item.get("topic", item))
        if isinstance(item, (dict, str))
    }

    # Build list of available topics.
    available = []

    for item in topics:
        title = _topic_title(item)

        if not title:
            continue

        title_key = title.lower()

        if title_key in used_titles:
            continue

        if title_key in claimed_titles:
            continue

        available.append(item)

    if not available:
        return None, None

    # --------------------------------------------------------------
    # RANDOM TOPIC SELECTION
    # --------------------------------------------------------------

    topic_data = random.choice(available)
    topic = _make_topic(topic_data)

    # --------------------------------------------------------------
    # CREATE JOB
    # --------------------------------------------------------------

    job_id = _next_job_id(paths)

    # THIS IS THE PERMANENT DIRECTORY FIX.
    #
    # It creates:
    #
    # BH000001/
    # ├── 01_research/
    # ├── 02_script/
    # ├── 03_scenes/
    # ├── 04_images/
    # ├── 05_audio/
    # ├── 06_video/
    # ├── 06_video/clips/
    # ├── 07_thumbnail/
    # └── state/
    #
    # before any processor tries to write an artifact.
    paths.prepare_job(job_id)

    job_dir = Path(paths.job(job_id))

    # --------------------------------------------------------------
    # JOB MANIFEST
    # --------------------------------------------------------------

    manifest = {
        "job_id": job_id,
        "topic": topic_data,
        "topic_title": topic.title,
        "status": "QWEN_RESEARCHING",
        "created_at": _now(),
        "updated_at": _now(),
        "processor": "qwen",
    }

    _write_json(paths.manifest(job_id), manifest)

    # --------------------------------------------------------------
    # CLAIMED TOPIC TRACKING
    # --------------------------------------------------------------

    claimed[job_id] = {
        "job_id": job_id,
        "topic": topic_data,
        "topic_title": topic.title,
        "claimed_at": _now(),
        "status": "QWEN_RESEARCHING",
    }

    _write_json(_claimed_path(paths), claimed)

    return topic, job_id


# ----------------------------------------------------------------------
# MARK TOPIC USED
# ----------------------------------------------------------------------

def mark_used(paths, job_id, topic=None):
    """
    Mark a topic as successfully processed/used.

    Removes it from claimed.json and adds it to used.json.
    """

    used = _read_json(_used_path(paths), [])
    claimed = _read_json(_claimed_path(paths), {})

    if not isinstance(used, list):
        used = []

    if not isinstance(claimed, dict):
        claimed = {}

    if topic is None:
        claim = claimed.get(job_id)

        if isinstance(claim, dict):
            topic = claim.get("topic", claim.get("topic_title"))

    if topic is not None:
        title = _topic_title(topic)

        # Avoid duplicate entries.
        existing = {
            _topic_title(item).lower()
            for item in used
            if _topic_title(item)
        }

        if title and title.lower() not in existing:
            used.append(topic)

    claimed.pop(job_id, None)

    _write_json(_used_path(paths), used)
    _write_json(_claimed_path(paths), claimed)

    # Update manifest if the job exists.
    manifest_path = paths.manifest(job_id)

    if manifest_path.exists():
        manifest = _read_json(manifest_path, {})

        if isinstance(manifest, dict):
            manifest["status"] = "USED"
            manifest["updated_at"] = _now()
            _write_json(manifest_path, manifest)


# ----------------------------------------------------------------------
# TOPIC OBJECT
# ----------------------------------------------------------------------

class Topic:
    """
    Lightweight topic object used by the processors.
    """

    def __init__(self, title, data=None):
        self.title = str(title).strip()
        self.data = data if isinstance(data, dict) else {}

    def __str__(self):
        return self.title

    def __repr__(self):
        return f"Topic(title={self.title!r})"


def _make_topic(data):
    """
    Convert a topic entry into the Topic object expected by
    qwen_pipeline.py.
    """

    if isinstance(data, str):
        return Topic(data, {"title": data})

    if isinstance(data, dict):
        title = (
            data.get("title")
            or data.get("topic")
            or data.get("name")
            or ""
        )

        return Topic(title, data)

    title = str(data).strip()

    return Topic(title, {"title": title})
