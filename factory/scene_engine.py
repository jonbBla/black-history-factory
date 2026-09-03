from __future__ import annotations

import json
import os
import re

from .utils import read_json, write_json_atomic

MIN_SCENES = 18
MAX_SCENES = 22
MIN_SCENE_WORDS = 4
MAX_SCENE_WORDS = 14


def word_tokens(text):
    return re.findall(r"\b[\w’'-]+\b", text)


def word_count(text):
    return len(word_tokens(text))


def normalize_word_sequence(text):
    return [x.lower() for x in word_tokens(text)]


def validate_plan(data, narration):
    if not isinstance(data, dict):
        raise ValueError("Scene planner output must be a JSON object.")

    scenes = data.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("Scene planner output is missing 'scenes'.")

    if not MIN_SCENES <= len(scenes) <= MAX_SCENES:
        raise ValueError(
            f"Invalid scene count: {len(scenes)}; required {MIN_SCENES}-{MAX_SCENES}."
        )

    previous = []
    for i, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict):
            raise ValueError(f"Scene {i} is not an object.")
        if "narration" not in scene or "visual_beat" not in scene:
            raise ValueError(f"Scene {i} must contain narration and visual_beat.")

        n = word_count(str(scene["narration"]))
        if n < MIN_SCENE_WORDS or n > MAX_SCENE_WORDS:
            raise ValueError(
                f"Scene {i} has {n} words; required {MIN_SCENE_WORDS}-{MAX_SCENE_WORDS}."
            )
        if not str(scene["visual_beat"]).strip():
            raise ValueError(f"Scene {i} has an empty visual beat.")

        previous.extend(normalize_word_sequence(scene["narration"]))

    original = normalize_word_sequence(narration)
    if previous != original:
        raise ValueError(
            "Scene narration does not preserve the original narration word-for-word "
            "and in the original order."
        )

    return scenes


def scene_count_for(words):
    # Keep the pacing around 8–11 spoken words per shot while respecting the cap.
    n = round(words / 9.5)
    return max(MIN_SCENES, min(MAX_SCENES, n))


def plan_prompt(narration, visual_bible, target_count):
    return f"""
You are the cinematic scene director for a fast-paced historical documentary.

EXACT NARRATION:
{narration}

LOCKED VISUAL CONTEXT:
{visual_bible}

Create an intelligent scene plan using the EXACT narration above.

TARGET SCENES: {target_count}
ALLOWED SCENES: {MIN_SCENES}-{MAX_SCENES}
ALLOWED WORDS PER SCENE: {MIN_SCENE_WORDS}-{MAX_SCENE_WORDS}

CRITICAL NARRATION RULES:
- Use every narration word exactly once.
- Keep the original word order.
- Do not add words.
- Do not remove words.
- Do not rewrite words.
- Do not summarize.
- Do not invent narration.
- You may choose where a scene begins and ends.
- Do not make scenes mathematically equal.

Choose boundaries intelligently based on:
- sentence meaning
- natural spoken pauses
- punctuation
- changes of subject
- changes of action
- changes of location
- changes of time
- visual reveals
- important historical details

Each scene should represent a distinct visual beat.

For each scene return:
- scene_id: sequential integer starting at 1
- narration: exact consecutive words from the narration
- visual_beat: concise description of what the viewer should see

Do NOT write the final image-generation prompt yet.

Return ONLY valid JSON:
{{
  "scenes": [
    {{
      "scene_id": 1,
      "narration": "exact consecutive narration words",
      "visual_beat": "specific cinematic visual beat"
    }}
  ]
}}
"""


def image_prompt_prompt(scene, visual_bible, config):
    style = getattr(config, "art_style_text", "cinematic 3D historical reconstruction")
    return f"""
Create one detailed SDXL image-generation prompt for this historical documentary scene.

SCENE NARRATION:
{scene['narration']}

VISUAL BEAT:
{scene['visual_beat']}

LOCKED VISUAL BIBLE:
{visual_bible}

ART DIRECTION:
{style}

Build the prompt in this order:
1. Main visible subject/action.
2. Period and regional grounding.
3. Historically supported clothing, architecture, tools, materials and environment.
4. Composition, camera angle, depth and scale.
5. Natural dramatic lighting and volumetric atmosphere.
6. Cinematic 3D historical reconstruction / high-end game cinematic quality.

Do not invent historical details merely to make the image spectacular.
Do not add modern objects, modern clothing, text, logos or watermarks.
Do not mention the narration or explain your choices.
Return ONLY the image prompt.
"""


def generate_plan(qwen, narration, visual_bible, target_count):
    last_error = None
    for attempt in range(1, 5):
        print(f"[SCENES] PLAN ATTEMPT {attempt}/4")
        try:
            data = qwen.generate_json(
                plan_prompt(narration, visual_bible, target_count),
                max_new_tokens=2600,
                retries=2,
            )
            validate_plan(data, narration)
            return data["scenes"]
        except Exception as e:
            last_error = e
            print(f"[SCENES] PLAN ATTEMPT {attempt} FAILED | {e}")
    raise ValueError(f"Scene planning failed after 4 attempts: {last_error}")


def run(paths, job_id, narration, visual_bible, config, qwen):
    narration = narration.strip()
    total_words = word_count(narration)
    target_count = scene_count_for(total_words)

    print(
        f"[QWEN] SCENES {job_id} | INTELLIGENT PLANNING | "
        f"{total_words} words | target {target_count} scenes"
    )

    scenes_dir = os.path.dirname(paths.scenes(job_id))
    partial_path = os.path.join(scenes_dir, "scenes_partial.json")
    os.makedirs(scenes_dir, exist_ok=True)

    # First create the scene plan.
    raw_scenes = generate_plan(qwen, narration, visual_bible, target_count)

    # Generate image descriptions scene-by-scene. Save partial progress so a
    # Colab reset does not throw away completed descriptions.
    partial = read_json(partial_path, {}) or {}
    completed = partial.get("scenes", []) if isinstance(partial, dict) else []
    completed_by_id = {
        int(x["scene_id"]): x
        for x in completed
        if isinstance(x, dict) and str(x.get("scene_id", "")).isdigit()
    }

    final_scenes = []

    for i, scene in enumerate(raw_scenes, 1):
        sid = int(scene["scene_id"])

        if sid in completed_by_id and completed_by_id[sid].get("image_prompt"):
            image_prompt = completed_by_id[sid]["image_prompt"]
            print(f"[QWEN] IMAGE DESCRIPTION {i}/{len(raw_scenes)} | SCENE {sid} | exists, skip")
        else:
            print(f"[QWEN] IMAGE DESCRIPTION {i}/{len(raw_scenes)} | SCENE {sid} | generating")
            image_prompt = qwen.generate(
                image_prompt_prompt(scene, visual_bible, config),
                max_new_tokens=800,
                temperature=0.35,
            ).strip()
            if not image_prompt:
                raise ValueError(f"Scene {sid} returned an empty image description.")

        item = {
            "scene_id": sid,
            "scene_number": sid,
            "narration": str(scene["narration"]).strip(),
            "word_count": word_count(scene["narration"]),
            "visual_beat": str(scene["visual_beat"]).strip(),
            "image_prompt": image_prompt,
            "image_description": image_prompt,
        }
        final_scenes.append(item)

        write_json_atomic(
            partial_path,
            {"scene_count": len(raw_scenes), "scenes": final_scenes},
        )

    output = {
        "scene_count": len(final_scenes),
        "narration_word_count": total_words,
        "scenes": final_scenes,
    }

    validate_plan(output, narration)
    write_json_atomic(paths.scenes(job_id), output)

    try:
        os.remove(partial_path)
    except FileNotFoundError:
        pass

    print(
        f"[QWEN] SCENES {job_id} | COMPLETE | "
        f"{len(final_scenes)} scenes"
    )
    return output
