"""Loads and validates 00_CONFIG/config.json from Google Drive."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

# Portrait, tuned for short-form vertical video (Shorts/Reels/TikTok-style
# 9:16). Image dims are close to the video's aspect ratio so video_engine.py's
# zoompan step only has to do light cropping, not aggressive reframing.
DEFAULTS = {
    "project_name": "Black History Factory",
    "language": "English",
    "target_video_minutes": 5,
    "image_width": 896,
    "image_height": 1600,
    "scenes_per_minute": 6,
    "enable_music": True,
    "enable_subtitles": True,
    "video_width": 1080,
    "video_height": 1920,
    "video_fps": 25,
    "github_repo": "",          # e.g. "yourname/black-history-factory"
    "github_dashboard_path": "dashboard/data",

    # --- Art style (see prompts/ART_STYLE.md for the full writeup with
    # reference images) ---
    # This is THE single place the project's visual identity lives. Every
    # scene's image_prompt inherits this via visual_bible.py, which locks
    # it in and ignores anything the research model tries to suggest
    # instead -- deliberately, so the series looks consistent across every
    # episode rather than drifting topic to topic. Change it here (and only
    # here) to change the look of every future video.
    "art_style": (
        "historical cinematic oil realism, painterly brushwork, warm "
        "directional late-afternoon or torchlight lighting, strong "
        "chiaroscuro, muted earth-tone palette with selective warm accent "
        "colors, wide cinematic documentary establishing-shot composition, "
        "emphasis on cloth/stone/metal/skin texture, not photoreal, not "
        "glossy 3D render"
    ),
}


@dataclass
class Config:
    values: dict = field(default_factory=dict)

    def __getattr__(self, name):
        try:
            return self.values[name]
        except KeyError as e:
            raise AttributeError(name) from e

    @classmethod
    def load(cls, drive_root: str) -> "Config":
        """drive_root is the path to BLACK_HISTORY_FACTORY/ on the mounted Drive."""
        cfg_path = Path(drive_root) / "00_CONFIG" / "config.json"
        values = dict(DEFAULTS)
        on_disk = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                on_disk = json.load(f)
            values.update(on_disk)

        # If the file doesn't exist yet, OR it exists but is missing keys
        # that DEFAULTS has (e.g. it was created by an older version of
        # this code before a setting existed), write the full merged set
        # back to disk. Otherwise a new default like art_style would only
        # ever live in memory -- invisible to anyone opening config.json to
        # see or edit it.
        if not cfg_path.exists() or set(DEFAULTS) - set(on_disk):
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(values, f, indent=2)

        return cls(values=values)
