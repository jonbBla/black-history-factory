"""Checkpoint engine: makes the factory survive Colab disconnects.

Rule used everywhere in main.py's job loop:
    before doing work for a stage -> check if that stage's OUTPUT FILE
    already exists on Drive -> if yes, SKIP; if no, GENERATE.

The checkpoint file itself is just a human-readable summary of where we are
(stage name + scene progress + stage timing history) -- the *real* source
of truth for "is this already done" is always the actual output file, so a
checkpoint that's slightly out of sync can never cause double-billing an
API or corrupt data; worst case it just re-checks a file that turns out to
already exist.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field, fields as dataclasses_fields
from typing import Optional

from .utils import read_json, write_json_atomic, now_iso

STAGES = [
    "topic_selection",
    "research",
    "fact_check",
    "narration",
    "visual_bible",
    "scene_planning",
    "image_generation",
    "audio_generation",
    "rendering",
    "thumbnail",
    "completed",
]


@dataclass
class Checkpoint:
    job_id: str
    topic_id: str = ""
    stage: str = "topic_selection"
    scene: int = 0
    completed: int = 0
    total: int = 0
    status: str = "running"       # running | paused | error | completed
    title: str = ""
    error: Optional[str] = None
    updated_at: str = field(default_factory=now_iso)
    stage_history: list = field(default_factory=list)  # [{"stage": ..., "at": ...}, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, paths) -> None:
        self.updated_at = now_iso()
        write_json_atomic(paths.checkpoint_json(self.job_id), self.to_dict())

    @classmethod
    def load(cls, paths, job_id: str) -> Optional["Checkpoint"]:
        data = read_json(paths.checkpoint_json(job_id))
        if data is None:
            return None
        # Tolerate older checkpoint files saved before a field was added.
        known = {f.name for f in dataclasses_fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        return cls(**data)

    @classmethod
    def load_or_create(cls, paths, job_id: str, title: str = "", topic_id: str = "") -> "Checkpoint":
        existing = cls.load(paths, job_id)
        if existing is not None:
            return existing
        cp = cls(job_id=job_id, title=title, topic_id=topic_id)
        cp.stage_history.append({"stage": cp.stage, "at": now_iso()})
        cp.save(paths)
        return cp

    def advance(self, paths, *, stage: Optional[str] = None,
                scene: Optional[int] = None, completed: Optional[int] = None,
                total: Optional[int] = None, status: Optional[str] = None) -> None:
        if stage is not None and stage != self.stage:
            self.stage = stage
            self.stage_history.append({"stage": stage, "at": now_iso()})
        if scene is not None:
            self.scene = scene
        if completed is not None:
            self.completed = completed
        if total is not None:
            self.total = total
        if status is not None:
            self.status = status
        self.save(paths)

    def fail(self, paths, message: str) -> None:
        self.status = "error"
        self.error = message
        self.save(paths)


def all_job_ids(paths) -> list:
    """Every job id that has ever been started, derived from actual
    checkpoint files on disk (the ground truth) rather than the topic log --
    topic ids and job ids are separate namespaces and must never be
    conflated when computing the next job id."""
    import os
    logs_dir = paths("09_LOGS")
    if not os.path.isdir(logs_dir):
        return []
    ids = []
    for f in os.listdir(logs_dir):
        if f.endswith(".checkpoint.json"):
            ids.append(f[: -len(".checkpoint.json")])
    return ids


def find_in_progress_job(paths) -> Optional[str]:
    """Looks at 08_STATUS/current.json to see if a job was mid-flight when
    Colab last stopped. Returns its job_id, or None if the last job finished
    (or none exists yet)."""
    current = read_json(paths.status_current)
    if not current:
        return None
    if current.get("status") in ("running", "paused", "error") and current.get("job_id"):
        return current["job_id"]
    return None


def stage_output_exists(paths, job_id: str, stage: str, expected_scenes: int = 0) -> bool:
    """The actual skip/generate check -- looks at real output files, not just
    the checkpoint record, so a stale or missing checkpoint never causes
    redone work or lost work."""
    import os
    if stage == "research":
        return os.path.exists(paths.research_raw(job_id))
    if stage == "fact_check":
        return os.path.exists(paths.research_verified(job_id))
    if stage == "narration":
        return os.path.exists(paths.narration_txt(job_id))
    if stage == "visual_bible":
        return os.path.exists(paths.visual_bible_json(job_id))
    if stage == "scene_planning":
        return os.path.exists(paths.scenes_json(job_id))
    if stage == "image_generation":
        d = paths.images_dir(job_id)
        if not os.path.isdir(d):
            return False
        count = len([f for f in os.listdir(d) if f.endswith(".png")])
        return expected_scenes > 0 and count >= expected_scenes
    if stage == "audio_generation":
        d = paths.audio_dir(job_id)
        if not os.path.isdir(d):
            return False
        count = len([f for f in os.listdir(d) if f.endswith(".wav")])
        return expected_scenes > 0 and count >= expected_scenes
    if stage == "rendering":
        return os.path.exists(paths.video_completed(job_id))
    if stage == "thumbnail":
        return os.path.exists(paths.thumbnail(job_id))
    return False
