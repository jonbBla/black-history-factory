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
fields, so every single scene is guaranteed to carry the locked art style,
lighting, architecture, and clothing consistently, rather than depending on
the model to re-include all of that correctly across N separate scenes.
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
    """Visual bible fields (architecture/clothing/materials/etc.) often
    come back from Qwen as full sentences, not short tags -- concatenating
    several of them plus the locked style string easily produces a prompt
    exceeding both CLIP's 77-token limit and even FLUX's 256-token T5
    budget, silently truncating mid-sentence at generation time. Trimming
    each contributing field to a short phrase here keeps the composed
    prompt within a realistic budget on purpose, rather than truncating
    unpredictably wherever the tokenizer happens to cut it off.
    """
    words = text.split()
    if len(words) <= max_words:
        return text.rstrip(".")
    return " ".join(words[:max_words]).rstrip(".,;:")


def _compose_image_prompt(scene: dict, visual_bible: dict) -> str:
    # Targets FLUX's T5 encoder budget (256 tokens, ~190-200 words with
    # margin for punctuation/subword overhead) as the real constraint --
    # FLUX also feeds the prompt to a CLIP encoder with a 77-token limit,
    # but that's expected and harmless by design (CLIP only contributes a
    # pooled summary embedding; T5 handles the actual detailed semantics).
    # SDXL-Lightning has no T5 branch, so it will still truncate harder at
    # CLIP's 77 tokens regardless -- a documented tradeoff in
    # load_sdxl_lightning()'s docstring, not something one shared prompt
    # can fully avoid for both backends at once.
    parts = [
        _trim(visual_bible.get("style", ""), 35),
        _trim(visual_bible.get("lighting", ""), 20),
        ", ".join(p for p in [visual_bible.get("period", ""), visual_bible.get("region", "")] if p),
        _trim(visual_bible.get("architecture", ""), 15),
        _trim(visual_bible.get("clothing", ""), 15),
        _trim(visual_bible.get("materials", ""), 12),
        _trim(scene.get("location", "") or visual_bible.get("environment", ""), 12),
    ]
    if scene.get("characters"):
        parts.append("featuring: " + ", ".join(scene["characters"][:3]))
    if scene.get("objects"):
        parts.append("with: " + ", ".join(scene["objects"][:3]))
    parts.append(f"camera movement: {scene.get('camera', DEFAULT_CAMERA)}")
    return ", ".join(p for p in parts if p)


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
