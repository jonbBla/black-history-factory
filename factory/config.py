from __future__ import annotations

import json
import os


DEFAULTS = {
    "project_name": "Black History Factory",
    "language": "English",

    "prepared_job_target": 1,

    "target_video_seconds": 90,
    "min_video_seconds": 80,
    "max_video_seconds": 100,

    "narration_words_min": 175,
    "narration_words_max": 220,

    "scene_count_min": 0,
    "scene_count_max": 0,
    "max_scene_words": 0,

    # ------------------------------------------------------------
    # IMAGE GENERATION
    # ------------------------------------------------------------

    "image_model": "ByteDance/SDXL-Lightning",

    "image_base_model": (
        "stabilityai/stable-diffusion-xl-base-1.0"
    ),

    "image_checkpoint": (
        "sdxl_lightning_4step_unet.safetensors"
    ),

    "image_width": 768,
    "image_height": 1344,

    "image_steps": 4,
    "image_guidance_scale": 0.0,

    "image_mode": "generate_missing_only",

    # ------------------------------------------------------------
    # VIDEO
    # ------------------------------------------------------------

    "video_width": 1080,
    "video_height": 1920,
    "video_fps": 30,

    # ------------------------------------------------------------
    # AUDIO
    # ------------------------------------------------------------

    "enable_subtitles": True,

    "enable_music": True,
    "music_volume": 0.1,

    "enable_ambience": True,
    "ambience_volume": 0.08,

    "enable_sfx": True,
    "sfx_volume": 0.15,

    # ------------------------------------------------------------
    # SOURCE CARD
    # ------------------------------------------------------------

    "source_card_enabled": True,
    "source_card_seconds": 4.5,

    # ------------------------------------------------------------
    # ART STYLE
    # ------------------------------------------------------------

    "art_style": {
        "primary": "cinematic 3D historical reconstruction",

        "description": (
            "epic cinematic historical reconstruction, "
            "physically plausible materials, "
            "period-authentic details, "
            "dramatic natural lighting, "
            "volumetric atmosphere, "
            "strong depth, "
            "detailed surfaces, "
            "cinematic composition, "
            "realistic proportions, "
            "highly detailed environments, "
            "realistic textures, "
            "dramatic scale, "
            "not flat cartoon"
        ),

        "default_renderer_feel": (
            "high-end game cinematic, Unreal Engine style"
        ),
    },

    # ------------------------------------------------------------
    # GITHUB
    # ------------------------------------------------------------

    "github_repo": "",
    "github_dashboard_path": "dashboard/data",
}


class Config:

    def __init__(self, values):
        self.values = values

    def __getattr__(self, name):
        if name in self.values:
            return self.values[name]

        raise AttributeError(name)

    @property
    def art_style_text(self):
        value = self.values.get("art_style", "")

        if isinstance(value, dict):
            return (
                value.get("description")
                or value.get("primary")
                or ""
            )

        return str(value)

    @classmethod
    def load(cls, root):

        path = os.path.join(
            root,
            "00_CONFIG",
            "config.json",
        )

        values = dict(DEFAULTS)

        if os.path.exists(path):

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:
                saved = json.load(f)

            if isinstance(saved, dict):
                values.update(saved)

        else:

            os.makedirs(
                os.path.dirname(path),
                exist_ok=True,
            )

            with open(
                path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    values,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        return cls(values)
