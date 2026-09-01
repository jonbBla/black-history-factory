from dataclasses import dataclass, asdict
import os

from .utils import (
    read_json,
    write_json_atomic,
    now_iso,
)


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
    data = read_json(
        paths.topics_json,
        [],
    ) or []

    return [
        Topic(**x)
        for x in data
        if isinstance(x, dict)
    ]


def _jobs_dir(paths):
    return paths("02_JOBS")


def _next_job_id(paths):

    jobs_dir = _jobs_dir(paths)

    if not os.path.isdir(jobs_dir):
        os.makedirs(
            jobs_dir,
            exist_ok=True,
        )

    numbers = []

    for name in os.listdir(jobs_dir):

        if not name.startswith("BH"):
            continue

        try:
            numbers.append(
                int(name[2:])
            )
        except ValueError:
            pass

    return f"BH{max(numbers, default=0) + 1:06d}"


def _job_status(paths, job_id):

    data = read_json(
        paths.manifest(job_id),
        {},
    ) or {}

    return data.get(
        "status",
        "",
    )


def find_resumable_job(paths):

    jobs_dir = _jobs_dir(paths)

    if not os.path.isdir(jobs_dir):
        return None, None

    topics = {
        t.id: t
        for t in load_topics(paths)
    }

    # States that mean the job is unfinished.
    resumable_statuses = {
        "QWEN_RESEARCHING",
        "QWEN_FACT_CHECKING",
        "QWEN_VISUAL_BIBLE",
        "QWEN_NARRATION",
        "QWEN_SCENE_PLANNING",
        "QWEN_ERROR",
    }

    candidates = []

    for name in os.listdir(jobs_dir):

        if not name.startswith("BH"):
            continue

        manifest = read_json(
            paths.manifest(name),
            {},
        ) or {}

        if not manifest:
            continue

        topic_id = manifest.get(
            "topic_id"
        )

        status = manifest.get(
            "status",
            "",
        )

        if (
            topic_id in topics
            and status in resumable_statuses
        ):
            candidates.append(
                (
                    manifest.get(
                        "created_at",
                        "",
                    ),
                    name,
                    topics[topic_id],
                )
            )

    if not candidates:
        return None, None

    # Oldest incomplete job first.
    candidates.sort(
        key=lambda x: x[0]
    )

    _, job_id, topic = candidates[0]

    return topic, job_id


def _claimed_topic_ids(paths):

    claimed = set()

    jobs_dir = _jobs_dir(paths)

    if not os.path.isdir(jobs_dir):
        return claimed

    for name in os.listdir(jobs_dir):

        if not name.startswith("BH"):
            continue

        manifest = read_json(
            paths.manifest(name),
            {},
        ) or {}

        topic_id = manifest.get(
            "topic_id"
        )

        status = manifest.get(
            "status",
            "",
        )

        # Any job that isn't completely finished
        # still owns its topic.
        if topic_id and status not in {
            "QWEN_READY",
            "REJECTED",
        }:
            claimed.add(topic_id)

    return claimed


def claim_next_topic(
    paths,
    processor="qwen",
):

    topics = load_topics(paths)

    claimed = _claimed_topic_ids(paths)

    # First check for an existing resumable job.
    topic, existing_job = find_resumable_job(
        paths
    )

    if topic and existing_job:
        return topic, existing_job

    for topic in topics:

        if topic.used:
            continue

        if topic.id in claimed:
            continue

        job_id = _next_job_id(paths)

        # Create only the job root/state directory.
        # Individual processors create their own files.
        os.makedirs(
            paths.job(job_id),
            exist_ok=True,
        )

        os.makedirs(
            paths.job(
                job_id
            ) + "/state",
            exist_ok=True,
        )

        write_json_atomic(
            paths.manifest(job_id),
            {
                "job_id": job_id,
                "topic_id": topic.id,
                "title": topic.title,
                "created_at": now_iso(),
                "claimed_by": processor,
                "status": "QWEN_RESEARCHING",
            },
        )

        write_json_atomic(
            paths.state(
                job_id,
                "qwen",
            ),
            {
                "status": "claimed",
                "updated_at": now_iso(),
                "processor": processor,
            },
        )

        return topic, job_id

    return None, None


def mark_used(paths, topic):

    used = read_json(
        paths.used_topics_json,
        [],
    ) or []

    if not any(
        isinstance(x, dict)
        and x.get("id") == topic.id
        for x in used
    ):
        used.append(
            topic.to_dict()
        )

    write_json_atomic(
        paths.used_topics_json,
        used,
    )

    topics = load_topics(paths)

    updated = []

    for current in topics:

        if current.id == topic.id:
            current.used = True

        updated.append(
            current.to_dict()
        )

    write_json_atomic(
        paths.topics_json,
        updated,
    )
