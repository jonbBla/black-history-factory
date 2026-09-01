from __future__ import annotations
import json, re
from .utils import write_json_atomic

CAMERAS = ["zoom_in", "zoom_out", "pan_left", "pan_right", "slow_push", "slow_pull"]

def norm(value):
    return re.sub(r"[^a-z0-9 ]", "", str(value).lower()).strip()

def similarity(a, b):
    A, B = set(norm(a).split()), set(norm(b).split())
    return len(A & B) / max(1, len(A | B))

def validate(scenes, config):
    if not isinstance(scenes, list) or not int(config.scene_count_min) <= len(scenes) <= int(config.scene_count_max):
        return False, "invalid scene count"
    narr, visuals = [], []
    for scene in scenes:
        if any(not str(scene.get(k, "")).strip() for k in ("narration", "visual_focus", "historical_role")):
            return False, "missing required scene field"
        n, v = str(scene["narration"]).strip(), str(scene["visual_focus"]).strip()
        if len(n.split()) > int(config.max_scene_words):
            return False, f"scene narration too long: {len(n.split())} words"
        if norm(n) in narr:
            return False, "duplicate narration"
        if any(similarity(v, old) >= 0.72 for old in visuals):
            return False, "near-duplicate visual focus"
        narr.append(norm(n)); visuals.append(v)
    return True, "OK"

def build_prompt(narration, visual_bible, config, correction=""):
    return f"""Break this narration into {config.scene_count_min}-{config.scene_count_max} UNIQUE visual beats.
Every spoken part must appear once, in order. Each scene narration must be {config.max_scene_words} words or fewer.
Do not repeat the hook, facts, events, sentences or conclusion. Every scene must advance the story.
Use hard_cut normally; crossfade only for meaningful changes of time, place or mood. No source card. No citations.
Return ONLY JSON: {{\"scenes\":[{{\"scene_id\":1,\"narration\":\"...\",\"visual_focus\":\"specific visual\",\"camera\":\"zoom_in\",\"transition\":\"hard_cut\",\"historical_role\":\"...\"}}]}}.
Cameras: {CAMERAS}. Never invent unsupported historical details.
{correction}
NARRATION:
{narration}
VISUAL BIBLE:
{json.dumps(visual_bible, ensure_ascii=False)}"""

def run(paths, job_id, narration, vb, config, qwen):
    error = ""
    for _ in range(4):
        raw = qwen.generate_json(build_prompt(narration, vb, config, error), max_new_tokens=3000)
        scenes = raw.get("scenes", raw) if isinstance(raw, dict) else raw
        ok, error = validate(scenes, config)
        if ok:
            break
    else:
        raise ValueError(f"Scene planning failed after 4 attempts: {error}")
    out = []
    for i, scene in enumerate(scenes, 1):
        focus = str(scene["visual_focus"]).strip()
        camera = scene.get("camera") if scene.get("camera") in CAMERAS else CAMERAS[(i - 1) % len(CAMERAS)]
        image_prompt = (
            f"{focus}, {vb.get('period','')}, {vb.get('region','')}, "
            f"architecture: {vb.get('architecture','')}, clothing: {vb.get('clothing','')}, "
            f"materials: {vb.get('materials','')}, environment: {vb.get('environment','')}, "
            f"people: {vb.get('people','')}, {vb.get('style','')}, {vb.get('lighting','dramatic natural lighting')}"
        )
        out.append({"scene_id": i, "narration": str(scene["narration"]).strip(), "visual_focus": focus,
                    "camera": camera, "transition": str(scene.get("transition", "hard_cut")),
                    "historical_role": str(scene.get("historical_role", "")), "image_prompt": image_prompt})
    write_json_atomic(paths.scenes(job_id), out)
    return out
