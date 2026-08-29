import json
import re
from .utils import write_json_atomic

CAMERAS = ["zoom_in", "zoom_out", "pan_left", "pan_right", "slow_push", "slow_pull"]
TRANSITIONS = ["hard_cut", "crossfade"]

def _norm(text):
    """Normalize text for duplicate detection."""
    return re.sub(r"\W+", " ", str(text).lower()).strip()

def _fingerprint(scene):
    return (
        _norm(scene.get("narration", "")),
        _norm(scene.get("visual_focus", ""))
    )

def _duplicate_report(scenes):
    seen = {}
    duplicates = []
    for i, s in enumerate(scenes, 1):
        fp = _fingerprint(s)
        if not fp[0] and not fp[1]:
            duplicates.append(i)
            continue
        if fp in seen:
            duplicates.append(i)
        else:
            seen[fp] = i

    # Also catch exact repeated narration even when the visual changes.
    narration_seen = {}
    repeated_narration = []
    for i, s in enumerate(scenes, 1):
        n = _norm(s.get("narration", ""))
        if not n:
            continue
        if n in narration_seen:
            repeated_narration.append(i)
        else:
            narration_seen[n] = i
    return duplicates, repeated_narration

def _clean_scenes(scenes, min_count, max_count):
    """Remove source-card scenes and obvious duplicates before validation."""
    cleaned = []
    seen_narration = set()
    seen_focus = set()

    for s in scenes:
        if not isinstance(s, dict):
            continue

        narration = str(s.get("narration", "")).strip()
        focus = str(s.get("visual_focus", "historical scene")).strip()

        # Source cards are created by the video processor, not the scene planner.
        low = (narration + " " + focus).lower()
        if "source card" in low or "primary source:" in low or "sources:" in low:
            continue

        nkey = _norm(narration)
        fkey = _norm(focus)

        # Exact repeated narration is never allowed.
        if nkey and nkey in seen_narration:
            continue

        # Exact repeated visual focus is also rejected.
        if fkey and fkey in seen_focus:
            continue

        seen_narration.add(nkey)
        seen_focus.add(fkey)

        cleaned.append({
            "scene_id": len(cleaned) + 1,
            "narration": narration,
            "visual_focus": focus,
            "camera": s.get("camera", "slow_push") if s.get("camera") in CAMERAS else "slow_push",
            "transition": s.get("transition", "hard_cut") if s.get("transition") in TRANSITIONS else "hard_cut",
            "historical_role": str(s.get("historical_role", "")).strip()
        })

    return cleaned

def _planner_prompt(narration, vb, config):
    return f"""
You are the scene planner for a fast-paced 90-second historical documentary.

Break the narration below into {config.scene_count_min}-{config.scene_count_max} UNIQUE visual scenes.

CRITICAL RULES:
1. The scenes must cover the narration ONCE from beginning to end, in order.
2. NEVER repeat, copy, paraphrase, or restate a previous scene's narration.
3. Every narration field must represent a DIFFERENT consecutive segment of the supplied narration.
4. Do NOT create an opening hook twice.
5. Do NOT repeat the conclusion.
6. Do NOT create a "recap", "reiterate", or "reinforce" scene.
7. Do NOT create a source-card scene. The video processor adds the source card separately.
8. Do NOT invent facts, people, locations, objects, architecture, dates, clothing, or technology not supported by the narration/research/visual bible.
9. Vary visual compositions. Do not reuse the same visual focus for multiple scenes unless the narration explicitly requires the same subject and the visual action is substantially different.
10. Prefer 2-4 second visual beats and hard cuts. Use crossfade only for meaningful changes of time, place, or mood.
11. A scene's narration should be the actual portion of the supplied narration that is spoken over that scene, not newly written narration.
12. The FINAL scene must finish the supplied narration naturally. Do not restart the story.

Return ONLY:
{{"scenes":[
  {{
    "scene_id": 1,
    "narration": "...",
    "visual_focus": "...",
    "camera": "zoom_in|zoom_out|pan_left|pan_right|slow_push|slow_pull",
    "transition": "hard_cut|crossfade",
    "historical_role": "..."
  }}
]}}

Visual bible:
{json.dumps(vb, ensure_ascii=False)}

Full narration:
{narration}
"""

def run(paths, job_id, narration, vb, config, qwen):
    prompt = _planner_prompt(narration, vb, config)

    # Try twice if the model repeats scenes. The second prompt explicitly
    # reports the violation so Qwen can repair the complete scene plan.
    scenes = None
    last_report = None

    for attempt in range(3):
        extra = ""
        if last_report:
            extra = f"""
The previous scene plan FAILED validation.
Repeated narration scene IDs: {last_report['repeated_narration']}.
Duplicate scene IDs: {last_report['duplicates']}.
Regenerate the ENTIRE scene list. Do not patch only those scenes.
"""
        raw = qwen.generate_json(
            prompt + extra,
            max_new_tokens=max(5000, config.scene_count_max * 180)
        )
        candidate = raw.get("scenes", raw) if isinstance(raw, dict) else raw
        if not isinstance(candidate, list):
            last_report = {"duplicates": [], "repeated_narration": ["INVALID_OUTPUT"]}
            continue

        candidate = _clean_scenes(
            candidate,
            config.scene_count_min,
            config.scene_count_max
        )
        dup, repeated = _duplicate_report(candidate)

        if not dup and not repeated and len(candidate) >= config.scene_count_min:
            scenes = candidate
            break

        last_report = {
            "duplicates": dup,
            "repeated_narration": repeated,
            "count": len(candidate)
        }

    if scenes is None:
        raise ValueError(
            "Scene planner produced repeated/invalid scenes after 3 attempts: "
            + json.dumps(last_report)
        )

    # Build image prompts only after the scene list passes validation.
    out = []
    for i, s in enumerate(scenes, 1):
        focus = s["visual_focus"]
        image_prompt = (
            f"{focus}, {vb.get('period','')}, {vb.get('region','')}, "
            f"{vb.get('style','')}, {vb.get('lighting','dramatic natural light')}"
        )
        out.append({
            "scene_id": i,
            "narration": s["narration"],
            "visual_focus": focus,
            "camera": s["camera"],
            "transition": s["transition"],
            "historical_role": s["historical_role"],
            "image_prompt": image_prompt
        })

    write_json_atomic(paths.scenes(job_id), out)
    return out
