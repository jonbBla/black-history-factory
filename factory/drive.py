"""Google Drive mount check and folder-tree path helpers.

Every other module gets its paths from a DrivePaths instance instead of
hardcoding strings, so the folder layout only needs to change in one place.
"""

from __future__ import annotations
import os
from dataclasses import dataclass

SUBFOLDERS = [
    "00_CONFIG",
    "01_TOPICS",
    "02_RESEARCH/raw", "02_RESEARCH/verified", "02_RESEARCH/sources",
    "03_SCRIPTS/narration", "03_SCRIPTS/scenes",
    "04_IMAGES/generating", "04_IMAGES/completed",
    "05_AUDIO/narration", "05_AUDIO/music",
    "06_VIDEOS/rendering", "06_VIDEOS/completed",
    "07_THUMBNAILS",
    "08_STATUS",
    "09_LOGS",
]


@dataclass
class DrivePaths:
    root: str  # e.g. /content/drive/MyDrive/BLACK_HISTORY_FACTORY

    def ensure_tree(self) -> None:
        for sub in SUBFOLDERS:
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)

    def __call__(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    # convenience accessors used throughout the codebase
    @property
    def config_json(self): return self("00_CONFIG", "config.json")
    @property
    def topics_json(self): return self("01_TOPICS", "topics.json")
    @property
    def used_topics_json(self): return self("01_TOPICS", "used_topics.json")
    @property
    def rejected_topics_json(self): return self("01_TOPICS", "rejected_topics.json")
    @property
    def status_current(self): return self("08_STATUS", "current.json")
    @property
    def status_history(self): return self("08_STATUS", "history.json")

    def research_raw(self, job_id): return self("02_RESEARCH", "raw", f"{job_id}.json")
    def research_verified(self, job_id): return self("02_RESEARCH", "verified", f"{job_id}.json")
    def visual_bible_json(self, job_id): return self("03_SCRIPTS", "scenes", f"{job_id}.visual_bible.json")
    def narration_txt(self, job_id): return self("03_SCRIPTS", "narration", f"{job_id}.txt")
    def scenes_json(self, job_id): return self("03_SCRIPTS", "scenes", f"{job_id}.json")
    def images_dir(self, job_id): return self("04_IMAGES", "completed", job_id)
    def audio_dir(self, job_id): return self("05_AUDIO", "narration", job_id)
    def video_rendering(self, job_id): return self("06_VIDEOS", "rendering", f"{job_id}.mp4")
    def video_completed(self, job_id): return self("06_VIDEOS", "completed", f"{job_id}.mp4")
    def thumbnail(self, job_id): return self("07_THUMBNAILS", f"{job_id}.png")
    def checkpoint_json(self, job_id): return self("09_LOGS", f"{job_id}.checkpoint.json")
    def job_manifest(self, job_id): return self("09_LOGS", f"{job_id}.manifest.json")
    def log_file(self, job_id): return self("09_LOGS", f"{job_id}.log")


def mount_drive() -> str:
    """Mounts Google Drive when running inside Colab. Returns the mount root.
    No-op (returns a local path) when not running inside Colab, so the rest
    of the code can be exercised/tested outside Colab too.
    """
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
        return "/content/drive/MyDrive"
    except ImportError:
        local = os.path.expanduser("~/black_history_factory_local_drive")
        os.makedirs(local, exist_ok=True)
        return local
