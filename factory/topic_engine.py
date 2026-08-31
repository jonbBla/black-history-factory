from dataclasses import dataclass, asdict
import os

from .utils import (
    read_json,
    write_json_atomic,
    now_iso
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

        self.aliases = (
            self.aliases or []
        )

    def to_dict(self):

        return asdict(self)


def load_topics(paths):

    return [
        Topic(**item)
        for item in read_json(
            paths.topics_json,
            []
        )
        if isinstance(
            item,
            dict
        )
    ]


def _next_job_id(paths):

    root = paths(
        "02_JOBS"
    )

    numbers = []

    if os.path.isdir(root):

        for name in os.listdir(root):

            if not name.startswith(
                "BH"
            ):
                continue

            try:

                numbers.append(
                    int(name[2:])
                )

            except ValueError:
                pass


    return (
        f"BH{max(numbers, default=0) + 1:06d}"
    )


def _job_status(
    paths,
    job_id
):

    state = read_json(
        paths.state(
            job_id,
            "qwen"
        ),
        {}
    ) or {}

    return str(
        state.get(
            "status",
            ""
        )
    ).upper()


def claim_next_topic(
    paths,
    processor="qwen"
):

    topics = load_topics(paths)

    root = paths(
        "02_JOBS"
    )

    claimed_topics = set()


    if os.path.isdir(root):

        for job_id in os.listdir(root):

            if not job_id.startswith(
                "BH"
            ):
                continue


            manifest = read_json(
                paths.manifest(job_id),
                {}
            ) or {}


            topic_id = manifest.get(
                "topic_id"
            )


            if not topic_id:
                continue


            status = str(
                manifest.get(
                    "status",
                    ""
                )
            ).upper()


            # These jobs are still associated
            # with their topic.
            if status not in {
                "FAILED",
                "QWEN_ERROR",
                "REJECTED",
                "ABANDONED"
            }:

                claimed_topics.add(
                    topic_id
                )


    for topic in topics:

        if topic.used:
            continue

        if topic.id in claimed_topics:
            continue


        job_id = _next_job_id(
            paths
        )


        # IMPORTANT:
        # DrivePaths.job() accepts ONE argument.
        job_root = paths.job(
            job_id
        )


        os.makedirs(
            job_root,
            exist_ok=True
        )


        os.makedirs(
            os.path.join(
                job_root,
                "state"
            ),
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
            paths.state(
                job_id,
                "qwen"
            ),
            {
                "status": "CLAIMED",
                "processor": processor,
                "job_id": job_id,
                "updated_at": now_iso()
            }
        )


        return topic, job_id


    return None, None


def mark_used(
    paths,
    topic
):

    used = read_json(
        paths.used_topics_json,
        []
    ) or []


    if not any(
        item.get("id") == topic.id
        for item in used
    ):

        used.append(
            topic.to_dict()
        )


    write_json_atomic(
        paths.used_topics_json,
        used
    )


    topics = load_topics(
        paths
    )


    for item in topics:

        if item.id == topic.id:

            item.used = True


    write_json_atomic(
        paths.topics_json,
        [
            item.to_dict()
            for item in topics
        ]
    )
