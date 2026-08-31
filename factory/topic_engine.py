from dataclasses import dataclass, asdict
import os

from .utils import read_json, write_json_atomic, now_iso


@dataclass
class Topic:
    id: str
    title: str
    category: str
    region: str
    period: str = ""
    description: str = ""
    aliases: list = None
    used: bool = False

    def __post_init__(self):
        self.aliases = self.aliases or []

    def to_dict(self):
        return asdict(self)


def load_topics(paths):
    return [
        Topic(**x)
        for x in read_json(paths.topics_json, [])
        if isinstance(x, dict)
    ]


def _next_job_id(paths):
    root = paths("02_JOBS")
    nums = []

    if os.path.isdir(root):
        for n in os.listdir(root):
            if n.startswith("BH"):
                try:
                    nums.append(int(n[2:]))
                except ValueError:
                    pass

    return f"BH{max(nums, default=0) + 1:06d}"


def _job_status(paths, job_id):
    state = read_json(paths.state(job_id, "qwen"), {}) or {}
    return str(state.get("status", "")).upper()


def find_resumable_job(paths, processor="qwen"):
    """
    Find a previously claimed Qwen job that did not finish.

    This allows Colab to be restarted without immediately abandoning
    the partially processed topic.
    """
    root = paths("02_JOBS")

    if not os.path.isdir(root):
        return None, None

    candidates = []

    for job_id in os.listdir(root):
        if not job_id.startswith("BH"):
            continue

        manifest = read_json(paths.manifest(job_id), {}) or {}

        if manifest.get("claimed_by") != processor:
            continue

        status = _job_status(paths, job_id)

        if status not in {"QWEN_READY", "COMPLETED"}:
            candidates.append(
                (job_id, manifest.get("topic_id"))
            )

    candidates.sort()

    for job_id, topic_id in candidates:
        for topic in load_topics(paths):
            if topic.id == topic_id:
                return topic, job_id

    return None, None


def claim_next_topic(paths, processor="qwen"):
    topics = load_topics(paths)
    claimed = set()

    root = paths("02_JOBS")

    if os.path.isdir(root):
        for job_id in os.listdir(root):
            manifest = read_json(
                paths.manifest(job_id), {}
            ) or {}

            topic_id = manifest.get("topic_id")

            if not topic_id:
                continue

            status = _job_status(paths, job_id)

            # Only prevent a topic from being claimed again if its job
            # is actually active or completed.
            if status not in {"FAILED", "ABANDONED"}:
                claimed.add(topic_id)

    for topic in topics:

        if topic.used:
            continue

        if topic.id in claimed:
            continue

        job_id = _next_job_id(paths)

        # IMPORTANT:
        # DrivePaths.job() accepts ONLY job_id.
        job_root = paths.job(job_id)

        os.makedirs(job_root, exist_ok=True)
        os.makedirs(
            os.path.join(job_root, "state"),
            exist_ok=True
        )

        write_json_atomic(
            paths.manifest(job_id),
            {
                "job_id": job_id,
                "topic_id": topic.id,
                "title": topic.title,
                "created_at": now_iso(),
                "claimed_by": processor,
                "status": "QWEN_RESEARCHING"
            }
        )

        write_json_atomic(
            paths.state(job_id, "qwen"),
            {
                "status": "CLAIMED",
                "updated_at": now_iso(),
                "processor": processor
            }
        )

        return topic, job_id

    return None, None


def mark_used(paths, topic):
    """
    Mark a topic as used ONLY after the entire Qwen pipeline succeeds.
    """

    used = read_json(
        paths.used_topics_json,
        []
    ) or []

    if not any(
        x.get("id") == topic.id
        for x in used
    ):
        used.append(topic.to_dict())

    write_json_atomic(
        paths.used_topics_json,
        used
    )

    topics = load_topics(paths)

    for item in topics:
        if item.id == topic.id:
            item.used = True

    write_json_atomic(
        paths.topics_json,
        [t.to_dict() for t in topics]
     )
