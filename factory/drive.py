from __future__ import annotations
import os
from dataclasses import dataclass

SUBFOLDERS = [
    "00_CONFIG",
    "01_TOPICS",
    "02_JOBS",
    "04_AUDIO_LIBRARY/music",
    "04_AUDIO_LIBRARY/ambience",
    "04_AUDIO_LIBRARY/sfx",
    "04_OUTPUT/completed",
    "04_OUTPUT/failed",
    "05_STATUS",
    "06_LOGS",
]

@dataclass
class DrivePaths:
    root: str

    def __call__(self, *parts):
        return os.path.join(self.root, *parts)

    def ensure_tree(self):
        for p in SUBFOLDERS:
            os.makedirs(self(p), exist_ok=True)

    @property
    def config_json(self): return self("00_CONFIG", "config.json")
    @property
    def topics_json(self): return self("01_TOPICS", "topics.json")
    @property
    def used_topics_json(self): return self("01_TOPICS", "used_topics.json")
    @property
    def rejected_topics_json(self): return self("01_TOPICS", "rejected_topics.json")
    @property
    def status_current(self): return self("05_STATUS", "current.json")
    @property
    def status_history(self): return self("05_STATUS", "history.json")

    def job(self, j, *parts): return self("02_JOBS", j, *parts)
    def state(self, j, processor): return self("02_JOBS", j, "state", processor + ".json")
    def manifest(self, j): return self("02_JOBS", j, "job.json")

    def research(self, j): return self("02_JOBS", j, "01_research", "research.json")
    def verified(self, j): return self("02_JOBS", j, "01_research", "verified.json")
    # Compatibility names used by fact_checker.py
    def research_raw(self, j): return self.research(j)
    def research_verified(self, j): return self.verified(j)

    def sources(self, j): return self("02_JOBS", j, "01_research", "sources.json")
    def narration(self, j): return self("02_JOBS", j, "02_script", "narration.txt")
    def script(self, j): return self("02_JOBS", j, "02_script", "script.json")
    def visual_bible(self, j): return self("02_JOBS", j, "02_script", "visual_bible.json")
    def scenes(self, j): return self("02_JOBS", j, "03_scenes", "scenes.json")
    def images_dir(self, j): return self("02_JOBS", j, "04_images")
    def audio_dir(self, j): return self("02_JOBS", j, "05_audio")
    def video_render(self, j): return self("02_JOBS", j, "06_video", "rendering.mp4")
    def video_final(self, j): return self("02_JOBS", j, "06_video", "final.mp4")
    def thumbnail(self, j): return self("02_JOBS", j, "07_thumbnail.png")
    def output_video(self, j): return self("04_OUTPUT", "completed", j + ".mp4")

def mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        return "/content/drive/MyDrive"
    except ImportError:
        p = os.path.expanduser("~/black_history_factory_local_drive")
        os.makedirs(p, exist_ok=True)
        return p
