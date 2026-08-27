"""Phase D -- real implementation.

Contract:
  input:  narration text + visual bible + config (scenes_per_minute,
          target_video_minutes) + a QwenClient
  output: paths.scenes_json(job_id) -- list of scene dicts:
    { scene_id, duration, narration, location, period, characters,
      objects, camera, transition, visual_focus, image_prompt }

image_prompt composition, redesigned around SD-Turbo's actual needs (the
default image backend, see image_engine.py): SD-Turbo's prompt-following
is explicitly weaker than SDXL/FLUX, and this project runs it at
guidance_scale=0 for 1-4 step speed, which also means negative-style
phrasing ("not photoreal") gets essentially no benefit from classifier-free
guidance. A short, single-subject prompt reliably outperforms a long,
multi-clause one on a weak-alignment model -- cramming many disconnected
fragments into one prompt (the previous approach: style + lighting +
region + architecture + clothing + materials + location, each separately
trimmed) gives the model too much to juggle, and it tends to fall back on
whichever tokens dominate (the shared style text), which is why every
image was coming out looking nearly identical regardless of scene content.

The fix follows a small, fixed slot structure instead:
    [subject/situation] + [environment] + [style] + [lighting]
Each slot is short and clean, not word-count-truncated mid-sentence.

`visual_focus` (a Qwen-generated field, see prompts/scene_planning.txt) is
the subject/situation slot -- Qwen writes a short, concrete, single-focus
description of what's actually visible in the scene, which is a much
better source for this than trying to algorithmically synthesize one from
scattered location/characters/objects fields. Qwen understands the
narrative; string concatenation never will. A fallback synthesizes
something reasonable from those scattered fields only if visual_focus is
missing (e.g. an older scenes.json from before this field existed, or the
no-model placeholder path).

Camera movement is deliberately NOT included in image_prompt: it's
consumed directly from scene["camera"] by video_engine.py's zoompan filter
AFTER the still image is generated -- it was never doing anything useful
as literal text in a still-image generation prompt.
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


def build_prompt(narration_text: str, visual_bible: dict, config) -> str:
    template = _load_template("scene_planning.txt")
    vb = visual_bible or {}
    visual_bible_summary = ", ".join(
        f"{k}: {v}" for k, v in (
            ("region", vb.get("region", "")),
            ("period", vb.get("period", "")),
            ("architecture", vb.get("architecture", "")),
            ("clothing", vb.get("clothing", "")),
            ("materials", vb.get("materials", "")),
            ("people", vb.get("people", "")),
        ) if v
    )
    return template.format(
        scenes_per_minute=config.scenes_per_minute,
        narration_text=narration_text,
        visual_bible_summary=visual_bible_summary or "(none established yet)",
    )


def _trim(text: str, max_words: int) -> str:
    """Cuts to a maximum word count -- used only as a final safety cap on
    already-short content, not as the primary way of shortening a long
    field (see _trim_by_clauses for that)."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text.rstrip(".")
    return " ".join(words[:max_words]).rstrip(".,;:")


def _trim_by_clauses(text: str, max_clauses: int) -> str:
    """Splits on commas and keeps the first N clauses. config.art_style is
    already written as comma-separated tags (see config.py), so cutting at
    a clause boundary produces a clean, grammatically complete short
    phrase -- unlike _trim()'s word-count cut, which can end awkwardly
    mid-clause (e.g. "...traditional attire such")."""
    if not text:
        return ""
    clauses = [c.strip() for c in text.split(",") if c.strip()]
    return ", ".join(clauses[:max_clauses])


def _fallback_subject(scene: dict, visual_bible: dict) -> str:
    """Only used if Qwen didn't return visual_focus. Synthesizes a single
    short phrase from whatever scattered fields ARE available, rather than
    the old approach of including all of them separately. If a character
    is present, the established skin tone/complexion (visual_bible
    "people") is appended so even this degraded fallback path doesn't lose
    the same representation accuracy the primary Qwen-written path is
    instructed to include."""
    chars = scene.get("characters") or []
    objs = scene.get("objects") or []
    bits = (chars[:1] + objs[:1]) or [scene.get("location", "") or visual_bible.get("environment", "")]
    subject = ", ".join(b for b in bits if b) or "a historical scene"

    if chars:
        people = visual_bible.get("people", "")
        if people and people != "(not specified)":
            subject = f"{subject}, {people}"

    return subject


def _compose_image_prompt(scene: dict, visual_bible: dict) -> str:
    # 1. Subject/situation -- the single most important slot. Qwen writes
    # this as an already-complete visual description INCLUDING setting
    # (see prompts/scene_planning.txt's example: "...crossing dunes at
    # dusk..."), so when it's present, a separate location/environment
    # slot would just repeat what's already in it -- wasted tokens on
    # redundancy, which actively hurts a weak-alignment model rather than
    # helping. Only the fallback path (visual_focus missing) needs an
    # explicit environment slot, since the fallback subject doesn't
    # naturally include one.
    subject = (scene.get("visual_focus") or "").strip()
    has_visual_focus = bool(subject)
    if not has_visual_focus:
        subject = _fallback_subject(scene, visual_bible)

    parts = [subject]

    if not has_visual_focus:
        environment = scene.get("location", "") or visual_bible.get("environment", "")
        if environment:
            parts.append(environment)

    # 2. Period/region -- brief historical grounding, distinct from
    # physical setting, so this is kept even when visual_focus already
    # covers the setting.
    period_region = ", ".join(
        p for p in [visual_bible.get("period", ""), visual_bible.get("region", "")] if p
    )
    if period_region:
        parts.append(period_region)

    # 3. Style -- the locked, series-wide look (see config.art_style),
    # cut at a clause boundary rather than mid-sentence.
    style_tag = _trim_by_clauses(visual_bible.get("style", ""), 2)
    if style_tag:
        parts.append(style_tag)

    # 4. Lighting -- short by construction already (see
    # prompts/visual_bible.txt's "5-10 word phrase" instruction to Qwen);
    # word-trim here is just a safety cap, not the primary shortening.
    lighting_tag = _trim(visual_bible.get("lighting", ""), 6)
    if lighting_tag:
        parts.append(lighting_tag)

    joined = ", ".join(p for p in parts if p)
    # SD-Turbo benefits from a short, focused prompt as a matter of
    # quality, not just fitting a token budget -- this cap is intentionally
    # tighter than the ~37-word hard CLIP limit computed from this
    # project's own observed tokenization ratio.
    return _trim(joined, 30)


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
        "visual_focus": raw.get("visual_focus", "") if isinstance(raw.get("visual_focus"), str) else "",
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

    prompt = build_prompt(narration_text, visual_bible, config)
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
