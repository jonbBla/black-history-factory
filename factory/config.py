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

    # --- Art style (see prompts/ART_STYLE.md for the full writeup) ---
    # This is THE single place the project's visual identity lives. Every
    # scene's image_prompt inherits this via visual_bible.py, which locks
    # it in and ignores anything the research model tries to suggest
    # instead. Change it here (and only here) to change the look of every
    # future video.
    #
    # Cinematic 3D render / CGI, NOT any painting style, specifically
    # because of SD-Turbo (the default image backend): painterly texture
    # (oil, matte painting, watercolor -- anything with stochastic
    # brushwork) needs several denoising steps to resolve cleanly, and
    # SD-Turbo only runs 1-4 steps by design. 3D/CGI-render prompts are
    # also a hugely well-represented category in general text-to-image
    # training data, emphasizing defined surfaces/materials/lighting
    # rather than painterly noise, so they render cleanly and sharply even
    # at very few steps. It's also not tied to the same "European romantic
    # painting" genre bias that pulled every earlier attempt toward a
    # generic look regardless of the scene's actual intended content.
    "art_style": (
        "cinematic 3D render, octane render, unreal engine style, "
        "volumetric lighting, highly detailed textures, sharp focus, "
        "dramatic atmosphere, not flat cartoon"
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
