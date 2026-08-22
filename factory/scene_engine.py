"""Phase D -- real implementation.

Contract (unchanged):
  input:  narration text + visual bible + config (scenes_per_minute,
          target_video_minutes) + a QwenClient
  output: paths.scenes_json(job_id) -- list of scene dicts:
    { scene_id, duration, narration, location, period, characters,
      objects, camera, transition, image_prompt }

Design decision: the Qwen call asks ONLY for narration segmentation and
blocking (location/period/characters/objects/camera/transition) -- it does
NOT ask the model to write image_prompt. image_prompt is instead composed
programmatically here from the shared visual_bible + this scene's own
fields.

image_prompt budget, corrected against REAL observed data: SDXL-Lightning
(the default image backend, see image_engine.py) uses CLIP text encoding
with a hard 77-token limit -- and unlike a generic ~1.3 tokens/word
estimate, actual logged output on this project's own content showed a
~2.07 tokens/word ratio (specialized/foreign vocabulary like "djellabas"
and "chiaroscuro" splits into more subword tokens than plain English).
That means the real safe word budget is roughly 35 words, not ~57. Content
past the budget is silently and completely dropped by CLIP, not gracefully
degraded -- so this composes the prompt with scene-differentiating content
(location, characters, objects) prioritized ahead of shared style
boilerplate, since losing a little style consistency on one image is a
minor hit, while losing scene differentiation on EVERY image (which is
what happened before this fix) makes the whole video look repetitive.
lighting and materials are dropped entirely from the composed prompt --
they're the lowest-information-per-word fields and largely redundant with
style's own lighting description already.

Camera movement is deliberately NOT included in image_prompt: it's
consumed directly from scene["camera"] by video_engine.py's zoompan filter
AFTER the still image is generated. It was never doing anything useful as
literal text in a still-image generation prompt.

If you switch to FLUX (load_flux() in image_engine.py instead of
load_sdxl_lightning()), this prompt is more conservative than FLUX's
256-token T5 budget actually allows -- FLUX could handle significantly
more detail. That's an accepted tradeoff for now since SDXL-Lightning is
the recommended default; a backend-aware prompt length (computed at
generation time in image_engine.py rather than baked in here) would be a
cleaner fix if FLUX becomes the primary path again.
"""

from __future__ import annotations
import os
from .utils import write_json_atomic

CAMERA_MOVES = {
    "zoom_in", "zoom_out", "pan_left", "pan_right",
    "pan_up", "pan_down", "slow_push", "slow_pull",
}
DEFAULT_CAMERA = "slow_push"
DEFAULT_TRANSITION = "crossfade"

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")


def _load_template(name: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(narration_text: str, config) -> str:
    template = _load_template("scene_planning.txt")
    return template.format(
        scenes_per_minute=config.scenes_per_minute,
        narration_text=narration_text,
    )


def _trim(text: str, max_words: int) -> str:
    """Cuts to a maximum word count rather than letting the tokenizer cut
    wherever it happens to land -- deliberate, budgeted truncation instead
    of unpredictable mid-sentence loss."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text.rstrip(".")
    return " ".join(words[:max_words]).rstrip(".,;:")


def _compose_image_prompt(scene: dict, visual_bible: dict) -> str:
    parts = [_trim(visual_bible.get("style", ""), 7)]

    location = scene.get("location", "") or visual_bible.get("environment", "")
    if location:
        parts.append(_trim(location, 5))

    period_region = ", ".join(
        p for p in [visual_bible.get("period", ""), visual_bible.get("region", "")] if p
    )
    if period_region:
        parts.append(period_region)

    if scene.get("characters"):
        parts.append(", ".join(scene["characters"][:2]))
    if scene.get("objects"):
        parts.append(", ".join(scene["objects"][:2]))

    parts.append(_trim(visual_bible.get("clothing", ""), 4))
    parts.append(_trim(visual_bible.get("architecture", ""), 4))

    joined = ", ".join(p for p in parts if p)
    return _trim(joined, 35)  # absolute ceiling regardless of how verbose any field is


def _normalize_scene(raw: dict, scene_id: int, visual_bible: dict) -> dict:
    duration = raw.get("duration", 8)
    if not isinstance(duration, (int, float)) or duration <= 0:
        duration = 8

    scene = {
        "scene_id": scene_id,
        "duration": int(duration),
        "narration": raw.get("narration", "") if isinstance(raw.get("narration"), str) else "",
        "location": raw.get("location", "") if isinstance(raw.get("location"), str) else visual_bible.get("region", ""),
        "period": raw.get("period", "") if isinstance(raw.get("period"), str) else visual_bible.get("period", ""),
        "characters": raw.get("characters") if isinstance(raw.get("characters"), list) else [],
        "objects": raw.get("objects") if isinstance(raw.get("objects"), list) else [],
        "camera": raw.get("camera") if raw.get("camera") in CAMERA_MOVES else DEFAULT_CAMERA,
        "transition": raw.get("transition") if isinstance(raw.get("transition"), str) and raw.get("transition") else DEFAULT_TRANSITION,
    }
    scene["image_prompt"] = _compose_image_prompt(scene, visual_bible)
    return scene


def run(paths, job_id: str, narration_text: str, visual_bible: dict, config, qwen=None) -> list:
    if qwen is None:
        total_scenes = max(1, int(config.target_video_minutes * config.scenes_per_minute))
        scenes = [
            _normalize_scene({"narration": "[NO MODEL LOADED]"}, i, visual_bible)
            for i in range(1, total_scenes + 1)
        ]
        write_json_atomic(paths.scenes_json(job_id), scenes)
        return scenes

    prompt = build_prompt(narration_text, config)
    try:
        raw_scenes = qwen.generate_json(prompt, max_new_tokens=3500)
    except ValueError as e:
        raise RuntimeError(f"Scene planning failed for job {job_id}: {e}") from e

    if not isinstance(raw_scenes, list) or len(raw_scenes) == 0:
        raise RuntimeError(f"Scene planning for job {job_id} returned no scenes")

    scenes = [
        _normalize_scene(s if isinstance(s, dict) else {}, i, visual_bible)
        for i, s in enumerate(raw_scenes, start=1)
    ]

    write_json_atomic(paths.scenes_json(job_id), scenes)
    return scenes
  
