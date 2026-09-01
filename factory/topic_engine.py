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


def _jobs_dir(paths):
    return paths("02_JOBS")


def _next_job_id(paths):
    jobs_dir = _jobs_dir(paths)

    if not os.path.isdir(jobs_dir):
        return "BH000001"

    nums = []

    for name in os.listdir(jobs_dir):
        if name.startswith("BH"):
            try:
                nums.append(int(name[2:]))
            except ValueError:
                pass

    return f"BH{max(nums, default=0) + 1:06d}"


def update_status(paths, job_id, status, **extra):
    """
    Update the main job manifest.

    This is used by qwen_pipeline.py after each major stage.
    """
    manifest_path = paths.manifest(job_id)

    data = read_json(manifest_path, {}) or {}

    data["job_id"] = job_id
    data["status"] = status
    data["updated_at"] = now_iso()

    for key, value in extra.items():
        data[key] = value

    write_json_atomic(manifest_path, data)

    return data


def claim_next_topic(paths, processor="qwen"):
    topics = load_topics(paths)

    jobs_dir = _jobs_dir(paths)
    claimed = set()

    if os.path.isdir(jobs_dir):
        for name in os.listdir(jobs_dir):
            manifest = read_json(paths.manifest(name), {})

            if manifest and manifest.get("topic_id"):
                claimed.add(manifest["topic_id"])

    for topic in topics:
        if topic.used or topic.id in claimed:
            continue

        job_id = _next_job_id(paths)

        os.makedirs(paths.job(job_id), exist_ok=True)
        os.makedirs(paths.job(job_id, "state"), exist_ok=True)

        update_status(
            paths,
            job_id,
            "QWEN_RESEARCHING",
            topic_id=topic.id,
            title=topic.title,
            created_at=now_iso(),
            claimed_by=processor,
        )

        write_json_atomic(
            paths.state(job_id, "qwen"),
            {
                "status": "claimed",
                "updated_at": now_iso(),
                "processor": processor,
            },
        )

        return topic, job_id

    return None, None


def find_resumable_job(paths):
    """
    Find a Qwen job that was started but did not finish.

    This allows Colab to resume from Drive after a reset.
    """
    jobs_dir = _jobs_dir(paths)

    if not os.path.isdir(jobs_dir):
        return None, None

    topics = {
        t.id: t
        for t in load_topics(paths)
    }

    resumable_statuses = {
        "QWEN_RESEARCHING",
        "FACT_CHECKING",
        "QWEN_FACT_CHECK",
        "QWEN_VISUAL_BIBLE",
        "QWEN_NARRATION",
        "QWEN_SCENE_PLANNING",
        "SCENE_PLANNING",
        "QWEN_ERROR",
    }

    for name in sorted(os.listdir(jobs_dir)):
        if not name.startswith("BH"):
            continue

        manifest = read_json(paths.manifest(name), {}) or {}

        if manifest.get("status") not in resumable_statuses:
            continue

        topic_id = manifest.get("topic_id")
        topic = topics.get(topic_id)

        if topic:
            return topic, name

    return None, None


def mark_used(paths, topic):
    used = read_json(paths.used_topics_json, []) or []

    if not any(x.get("id") == topic.id for x in used):
        used.append(topic.to_dict())

    write_json_atomic(
        paths.used_topics_json,
        used
    )

    topics = load_topics(paths)

    updated_topics = []

    for t in topics:
        if t.id == topic.id:
            t.used = True

        updated_topics.append(t.to_dict())

    write_json_atomic(
        paths.topics_json,
        updated_topics
)
