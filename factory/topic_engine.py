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


def _slugify(text):
    """
    Create a stable identifier from a topic title.
    """
    text = str(text or "").strip().lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "-",
        text,
    )

    text = re.sub(
        r"-+",
        "-",
        text,
    ).strip("-")

    return text or "topic"


# ============================================================
# TOPIC FIELD EXTRACTION
# ============================================================

def _topic_id(value):
    if isinstance(value, Topic):
        return str(value.id).strip()

    if isinstance(value, dict):
        value_id = (
            value.get("id")
            or value.get("topic_id")
            or value.get("slug")
        )

        if value_id:
            return str(value_id).strip()

        title = (
            value.get("title")
            or value.get("topic")
            or value.get("name")
            or ""
        )

        return _slugify(title)

    if isinstance(value, str):
        return _slugify(value)

    return "topic"


def _topic_title(value):
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
    if isinstance(value, Topic):
        return str(
            value.category or "history"
        ).strip()

    if isinstance(value, dict):
        return str(
            value.get("category")
            or value.get("type")
            or value.get("field")
            or "history"
        ).strip()

    return "history"


def _topic_region(value):
    if isinstance(value, Topic):
        return str(
            value.region or "Not specified"
        ).strip()

    if isinstance(value, dict):
        return str(
            value.get("region")
            or value.get("location")
            or value.get("geographic_region")
            or "Not specified"
        ).strip()

    return "Not specified"


def _topic_period(value):
    if isinstance(value, Topic):
        return str(
            value.period or "Not specified"
        ).strip()

    if isinstance(value, dict):
        return str(
            value.get("period")
            or value.get("era")
            or value.get("time_period")
            or "Not specified"
        ).strip()

    return "Not specified"


def _topic_description(value):
    if isinstance(value, Topic):
        return str(
            value.description or ""
        ).strip()

    if isinstance(value, dict):
        return str(
            value.get("description")
            or value.get("summary")
            or value.get("details")
            or value.get("angle")
            or ""
        ).strip()

    return ""


# ============================================================
# TOPIC OBJECT
# ============================================================

@dataclass
class Topic:
    """
    Complete topic object used by the Qwen pipeline.

    research_engine.py expects:

        topic.id
        topic.title
        topic.category
        topic.region
        topic.period
        topic.description
    """

    id: str
    title: str
    category: str = "history"
    region: str = "Not specified"
    period: str = "Not specified"
    description: str = ""

    # Preserve any additional fields from topics.json.
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.id = str(
            self.id or _slugify(self.title)
        ).strip()

        self.title = str(
            self.title or ""
        ).strip()

        self.category = str(
            self.category or "history"
        ).strip()

        self.region = str(
            self.region or "Not specified"
        ).strip()

        self.period = str(
            self.period or "Not specified"
        ).strip()

        self.description = str(
            self.description or ""
        ).strip()

        if not isinstance(self.metadata, dict):
            self.metadata = {}

    def to_dict(self):
        """
        Convert the Topic back to a JSON-compatible object.
        """

        data = dict(self.metadata)

        data.update(
            {
                "id": self.id,
                "title": self.title,
                "category": self.category,
                "region": self.region,
                "period": self.period,
                "description": self.description,
            }
        )

        return data


# ============================================================
# TOPIC CONVERSION
# ============================================================

def _make_topic(value):
    """
    Convert a topic entry into a complete Topic object.

    Supported:

        "Venda Reed-Pipe Ensemble Music"

    or:

        {
            "id": "venda-reed-pipe-ensemble-music",
            "title": "Venda Reed-Pipe Ensemble Music",
            "category": "music",
            "region": "Southern Africa",
            "period": "20th century",
            "description": "..."
        }

    Missing region/period are NOT guessed.
    They become "Not specified" so Qwen research can establish them.
    """

    if isinstance(value, Topic):
        return value

    # --------------------------------------------------------
    # Simple string topic
    # --------------------------------------------------------

    if isinstance(value, str):
        title = value.strip()

        if not title:
            return None

        return Topic(
            id=_slugify(title),
            title=title,
            category="history",
            region="Not specified",
            period="Not specified",
            description="",
            metadata={},
        )

    # --------------------------------------------------------
    # Dictionary topic
    # --------------------------------------------------------

    if isinstance(value, dict):

        title = (
            value.get("title")
            or value.get("topic")
            or value.get("name")
        )

        if not title:
            return None

        title = str(title).strip()

        topic_id = (
            value.get("id")
            or value.get("topic_id")
            or value.get("slug")
            or _slugify(title)
        )

        category = (
            value.get("category")
            or value.get("type")
            or value.get("field")
            or "history"
        )

        region = (
            value.get("region")
            or value.get("location")
            or value.get("geographic_region")
            or "Not specified"
        )

        period = (
            value.get("period")
            or value.get("era")
            or value.get("time_period")
            or "Not specified"
        )

        description = (
            value.get("description")
            or value.get("summary")
            or value.get("details")
            or value.get("angle")
            or ""
        )

        metadata = dict(value)

        return Topic(
            id=str(topic_id).strip(),
            title=title,
            category=str(category).strip(),
            region=str(region).strip(),
            period=str(period).strip(),
            description=str(description).strip(),
            metadata=metadata,
        )

    return None


# ============================================================
# PATH HELPERS
# ============================================================

def _topics_path(paths):
    return (
        Path(paths.root)
        / "01_TOPICS"
        / "topics.json"
    )


def _used_path(paths):
    return (
        Path(paths.root)
        / "01_TOPICS"
        / "used.json"
    )


def _claimed_path(paths):
    return (
        Path(paths.root)
        / "01_TOPICS"
        / "claimed.json"
    )


def _rejected_path(paths):
    return (
        Path(paths.root)
        / "01_TOPICS"
        / "rejected.json"
    )


# ============================================================
# JOB ID
# ============================================================

def _next_job_id(paths):
    """
    Find the next BH###### job ID.

    Existing jobs are scanned so Colab restarts do not
    reset the numbering.
    """

    jobs_dir = (
        Path(paths.root)
        / "02_JOBS"
    )

    jobs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    jobs_dir = (
        Path(paths.root)
        / "02_JOBS"
    )

    if not jobs_dir.exists():
        return None, None

    candidates = []

    for job_dir in jobs_dir.iterdir():

        if not job_dir.is_dir():
            continue

        manifest_path = (
            job_dir / "job.json"
        )

        if not manifest_path.exists():
            continue

        manifest = _read_json(
            manifest_path,
            {},
        )

        if not isinstance(manifest, dict):
            continue

        status = _manifest_status(
            manifest
        )

        if status not in RESUMABLE_STATUSES:
            continue

        # ----------------------------------------------------
        # Recover topic
        # ----------------------------------------------------

        topic_data = manifest.get(
            "topic"
        )

        if topic_data is None:
            topic_data = {
                "id": manifest.get(
                    "topic_id"
                ) or manifest.get(
                    "id"
                ) or _slugify(
                    manifest.get(
                        "title",
                        "",
                    )
                ),
                "title": manifest.get(
                    "title",
                    "",
                ),
                "category": manifest.get(
                    "category",
                    "history",
                ),
                "region": manifest.get(
                    "region",
                    "Not specified",
                ),
                "period": manifest.get(
                    "period",
                    "Not specified",
                ),
                "description": manifest.get(
                    "description",
                    "",
                ),
            }

        topic = _make_topic(
            topic_data
        )

        if topic is None:
            continue

        if not topic.title:
            continue

        # ----------------------------------------------------
        # Preserve fields that may exist directly in manifest
        # but were absent from older topic records.
        # ----------------------------------------------------

        if (
            topic.region == "Not specified"
            and manifest.get("region")
        ):
            topic.region = str(
                manifest["region"]
            ).strip()

        if (
            topic.period == "Not specified"
            and manifest.get("period")
        ):
            topic.period = str(
                manifest["period"]
            ).strip()

        if (
            topic.category == "history"
            and manifest.get("category")
        ):
            topic.category = str(
                manifest["category"]
            ).strip()

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

    # Ensure every required job directory exists.
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

    if not isinstance(
        topics_data,
        list,
    ):
        topics_data = []

    if not isinstance(
        used_data,
        list,
    ):
        used_data = []

    if not isinstance(
        claimed_data,
        dict,
    ):
        claimed_data = {}

    if not isinstance(
        rejected_data,
        list,
    ):
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
                item.get(
                    "topic",
                    item,
                )
            )
        else:
            title = _topic_title(item)

        if title:
            claimed_titles.add(
                title.lower()
            )

    # --------------------------------------------------------
    # Build available topic list
    # --------------------------------------------------------

    available = []

    for raw_topic in topics_data:

        topic = _make_topic(
            raw_topic
        )

        if topic is None:
            continue

        if not topic.title:
            continue

        title_key = (
            topic.title.lower()
        )

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

    topic = random.choice(
        available
    )

    # --------------------------------------------------------
    # CREATE JOB
    # --------------------------------------------------------

    job_id = _next_job_id(
        paths
    )

    # IMPORTANT:
    # Create all job folders before any processor writes.
    paths.prepare_job(
        job_id
    )

    now = _now()

    # --------------------------------------------------------
    # Job manifest
    # --------------------------------------------------------

    manifest = {
        "job_id": job_id,

        "topic_id": topic.id,

        "title": topic.title,

        "category": topic.category,

        "region": topic.region,

        "period": topic.period,

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

        "topic_id": topic.id,

        "topic": topic.to_dict(),

        "title": topic.title,

        "category": topic.category,

        "region": topic.region,

        "period": topic.period,

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

def mark_used(
    paths,
    job_id,
    topic=None,
):
    """
    Mark a topic as successfully used.
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

    if not isinstance(
        used_data,
        list,
    ):
        used_data = []

    if not isinstance(
        claimed_data,
        dict,
    ):
        claimed_data = {}

    # --------------------------------------------------------
    # Recover topic if necessary
    # --------------------------------------------------------

    if topic is None:

        record = claimed_data.get(
            job_id
        )

        if isinstance(
            record,
            dict,
        ):
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

        if isinstance(
            manifest,
            dict,
        ):
            topic = _make_topic(
                manifest.get(
                    "topic",
                    manifest,
                )
            )

    if topic is None:
        return False

    # --------------------------------------------------------
    # Avoid duplicates
    # --------------------------------------------------------

    existing_titles =
