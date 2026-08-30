import json
from .utils import write_json_atomic

CAMERAS = ["zoom_in","zoom_out","pan_left","pan_right","slow_push","slow_pull"]

def normalize_text(text):
    return " ".join(str(text).lower().split())

def duplicate_narration(scenes):
    seen = set()
    for scene in scenes:
        narration = normalize_text(scene.get("narration", ""))
        if not narration or narration in seen:
            return True
        seen.add(narration)
    return False

def duplicate_visuals(scenes):
    seen = set()
    for scene in scenes:
        focus = normalize_text(scene.get("visual_focus", ""))
        if not focus or focus in seen:
            return True
        seen.add(focus)
    return False

def validate_scenes(scenes, config):
    if not isinstance(scenes, list):
        return False, "Scenes is not a list"
    if not (config.scene_count_min <= len(scenes) <= config.scene_count_max):
        return False, "Invalid scene count"
    if duplicate_narration(scenes):
        return False, "Repeated narration detected"
    if duplicate_visuals(scenes):
        return False, "Repeated visual focus detected"
    for scene in scenes:
        for key in ["narration","visual_focus","camera","transition","historical_role"]:
            if not str(scene.get(key, "")).strip():
                return False, f"Missing {key}"
    return True, "OK"

def build_prompt(narration, vb, config):
    return f"""You are the visual scene planner for a historical documentary.

Create {config.scene_count_min}-{config.scene_count_max} UNIQUE visual beats.

The narration below is ONE continuous timeline. EVERY PART must be represented EXACTLY ONCE.

DO NOT:
- repeat the opening hook later
- repeat an earlier narration segment
- repeat the same visual
- repeat the same historical event
- restart the story
- create a second summary or conclusion
- create a source card, citations, or text cards

Each scene must advance the story. Normally use approximately 2-5 seconds per visual beat.
Use varied visual types: environment, architecture, people, artifacts, tools, maps, landscape, reconstruction, close-up, activity, evidence, daily life.
Do not invent unsupported historical details.

Return ONLY:
{{"scenes":[{{"scene_id":1,"narration":"...","visual_focus":"...","camera":"...","transition":"...","historical_role":"..."}}]}}

Allowed cameras: {CAMERAS}

NARRATION:
{narration}

VISUAL BIBLE:
{json.dumps(vb, ensure_ascii=False)}"""

def run(paths, job_id, narration, vb, config, qwen):
    prompt = build_prompt(narration, vb, config)
    last_error = ""
    for attempt in range(3):
        current_prompt = prompt if attempt == 0 else prompt + f"""
The previous scene plan FAILED validation because:
{last_error}
Regenerate the ENTIRE scene plan. Do not patch it. Every narration segment must appear exactly once.
"""
        raw = qwen.generate_json(current_prompt, max_new_tokens=4500)
        scenes = raw.get("scenes", raw) if isinstance(raw, dict) else raw
        valid, error = validate_scenes(scenes, config)
        if valid:
            out = []
            for i, scene in enumerate(scenes, 1):
                focus = str(scene.get("visual_focus","historical scene")).strip()
                out.append({
                    "scene_id": i,
                    "narration": str(scene.get("narration","")).strip(),
                    "visual_focus": focus,
                    "camera": scene.get("camera") if scene.get("camera") in CAMERAS else "slow_push",
                    "transition": scene.get("transition","hard_cut"),
                    "historical_role": scene.get("historical_role",""),
                    "image_prompt": f"{focus}, {vb.get('period','')}, {vb.get('region','')}, {vb.get('style','')}, {vb.get('lighting','dramatic natural light')}"
                })
            write_json_atomic(paths.scenes(job_id), out)
            return out
        last_error = error
    raise ValueError(f"Scene generation failed validation: {last_error}")
