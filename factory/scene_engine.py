import json
import re
from pathlib import Path

from factory.utils import write_json_atomic


TARGET_SCENES = 20
MIN_SCENES = 18
MAX_SCENES = 22

MIN_SCENE_WORDS = 4
MAX_SCENE_WORDS = 14

MAX_ATTEMPTS = 4


def count_words(text):
    """Use one consistent word-counting method."""
    return len(re.findall(r"\b[\w’'-]+\b", str(text)))


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def similar(a, b):
    """
    Simple similarity check to catch repeated scene narration.
    """
    a_words = set(normalize_text(a).split())
    b_words = set(normalize_text(b).split())

    if not a_words or not b_words:
        return False

    overlap = len(a_words & b_words)
    smaller = min(len(a_words), len(b_words))

    return (overlap / smaller) >= 0.75


def validate_scenes(scenes, narration):
    """
    Validate the complete scene plan.
    """

    if not isinstance(scenes, list):
        return False, "scenes is not a list"

    count = len(scenes)

    if count < MIN_SCENES or count > MAX_SCENES:
        return False, f"invalid scene count: {count}"

    seen = []

    for i, scene in enumerate(scenes, start=1):

        if not isinstance(scene, dict):
            return False, f"scene {i} is not an object"

        scene_narration = str(scene.get("narration", "")).strip()

        if not scene_narration:
            return False, f"scene {i} has no narration"

        words = count_words(scene_narration)

        if words < MIN_SCENE_WORDS:
            return False, f"scene narration too short: {words} words"

        if words > MAX_SCENE_WORDS:
            return False, f"scene narration too long: {words} words"

        # Catch exact duplicates.
        normalized = normalize_text(scene_narration)

        if normalized in seen:
            return False, f"duplicate scene narration at scene {i}"

        # Catch near duplicates.
        for previous in seen:
            if similar(scene_narration, previous):
                return False, f"near-duplicate scene narration at scene {i}"

        seen.append(normalized)

    # Make sure the scene plan is actually connected to the narration.
    narration_words = set(normalize_text(narration).split())

    if not narration_words:
        return False, "original narration is empty"

    scene_words = set()

    for scene in scenes:
        scene_words.update(
            normalize_text(scene.get("narration", "")).split()
        )

    overlap = len(narration_words & scene_words) / max(len(narration_words), 1)

    if overlap < 0.55:
        return False, f"scene narration does not sufficiently cover original narration: {overlap:.0%}"

    return True, "valid"


def extract_json(text):
    """
    Extract JSON even if Qwen wraps it in markdown.
    """

    text = str(text).strip()

    # Remove markdown fences.
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    start = text.find("[")

    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):

        char = text[i]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "[":
            depth += 1

        elif char == "]":
            depth -= 1

            if depth == 0:
                candidate = text[start:i + 1]

                try:
                    return json.loads(candidate)
                except Exception:
                    return None

    return None


def build_prompt(narration, visual_bible, correction=None):

    correction_text = ""

    if correction:
        correction_text = f"""
YOUR PREVIOUS OUTPUT FAILED VALIDATION.

Problem:
{correction}

Fix ONLY the problem.

Do not reduce the number of scenes.
Do not rewrite the narration.
Do not repeat scenes.
Do not add a source-card scene.
"""

    return f"""
You are the scene planner for a short historical documentary.

Create a visual scene plan from the EXISTING narration below.

IMPORTANT:
You are NOT writing a new narration.

You are dividing the existing narration into consecutive short narration
segments and assigning one visual beat to each segment.

TARGET:
20 scenes.

ALLOWED:
18 to 22 scenes.

Every scene narration MUST contain:
4 to 14 words.

The scene narrations must:

1. Follow the original narration in EXACT chronological order.
2. Cover the narration from beginning to end.
3. Never repeat the same information.
4. Never invent additional narration.
5. Never combine unrelated parts of the story.
6. Never repeat a previous scene.
7. Never create a "Source card" scene.
8. Never create an introduction that isn't present in the narration.
9. Never create an ending that isn't present in the narration.

Think of the narration as a paragraph that must be cut into
20 consecutive pieces.

Example:

Original:
"King X became ruler after his father's death. He expanded the kingdom
and built new trade routes."

Good segmentation:

Scene 1:
"King X became ruler after his father's death."

Scene 2:
"He expanded the kingdom."

Scene 3:
"And built new trade routes."

Do NOT rewrite these sentences into different wording.

For each scene provide:

- scene_number
- narration
- visual_description

The visual_description should describe what should appear on screen.

The visual description should be cinematic, historically appropriate,
specific and visually useful.

VISUAL STYLE:
{visual_bible}

OUTPUT ONLY VALID JSON.

Required format:

[
  {{
    "scene_number": 1,
    "narration": "...",
    "visual_description": "..."
  }}
]

Continue until the entire narration has been covered.

ORIGINAL NARRATION:
{narration}

{correction_text}
"""


def run(paths, job_id, narration, visual_bible, config, qwen):

    print(
        f"[SCENES] {job_id} | "
        f"TARGET: ~{TARGET_SCENES} scenes | "
        f"acceptable: {MIN_SCENES}-{MAX_SCENES}"
    )

    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):

        if attempt == 1:
            print(
                f"[SCENES] {job_id} | "
                f"ATTEMPT {attempt}/{MAX_ATTEMPTS} | "
                f"Planning {TARGET_SCENES} visual beats"
            )
        else:
            print(
                f"[SCENES] {job_id} | "
                f"ATTEMPT {attempt}/{MAX_ATTEMPTS} | "
                f"Correcting scene plan"
            )

        prompt = build_prompt(
            narration=narration,
            visual_bible=visual_bible,
            correction=last_error
        )

        try:

            raw = qwen.generate(
                prompt,
                max_new_tokens=1400,
                temperature=0.35
            )

            scenes = extract_json(raw)

            if scenes is None:
                last_error = "invalid JSON output"
                print(
                    f"[SCENES] {job_id} | "
                    f"Invalid: {last_error}"
                )
                continue

            valid, error = validate_scenes(
                scenes,
                narration
            )

            if not valid:

                last_error = error

                print(
                    f"[SCENES] {job_id} | "
                    f"Invalid: {error}"
                )

                continue

            # Normalize scene numbers.
            for i, scene in enumerate(scenes, start=1):
                scene["scene_number"] = i

            output = {
                "job_id": job_id,
                "scene_count": len(scenes),
                "scenes": scenes
            }

            output_path = paths.scenes(job_id)

            write_json_atomic(
                output_path,
                output
            )

            print(
                f"[SCENES] {job_id} | "
                f"ACCEPTED | {len(scenes)} scenes"
            )

            print(
                f"[SCENES] {job_id} | "
                f"COMPLETE | {output_path}"
            )

            return output

        except Exception as e:

            last_error = str(e)

            print(
                f"[SCENES] {job_id} | "
                f"ERROR: {last_error}"
            )

    raise RuntimeError(
        f"Scene planning failed after "
        f"{MAX_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )
