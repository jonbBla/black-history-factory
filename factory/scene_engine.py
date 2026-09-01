import json
import re

from .utils import write_json_atomic


CAMERAS = [
    "zoom_in",
    "zoom_out",
    "pan_left",
    "pan_right",
    "slow_push",
    "slow_pull",
]

TRANSITIONS = [
    "hard_cut",
    "crossfade",
]


TARGET_SCENES = 20
MIN_SCENES = 18
MAX_SCENES = 22

MAX_SCENE_WORDS = 14
MIN_SCENE_WORDS = 4


def word_count(text):
    return len(re.findall(r"\b[\w’'-]+\b", str(text)))


def normalize_text(text):
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def similarity(a, b):
    """
    Simple word-overlap similarity.
    Used only to catch obvious duplicate scenes.
    """
    a = set(normalize_text(a).split())
    b = set(normalize_text(b).split())

    if not a or not b:
        return 0.0

    return len(a & b) / max(len(a | b), 1)


def build_prompt(narration, vb):
    return f"""
Break the following documentary narration into approximately {TARGET_SCENES}
short visual beats.

IMPORTANT SCENE REQUIREMENTS:

TARGET:
{TARGET_SCENES} scenes.

VALID RANGE:
{MIN_SCENES}-{MAX_SCENES} scenes.

Each scene must contain:
- scene_id
- narration
- visual_focus
- camera
- transition
- historical_role

NARRATION REQUIREMENTS:

1. Every word of the original narration must be represented.
2. Do NOT repeat a sentence.
3. Do NOT repeat a fact.
4. Do NOT repeat the hook later.
5. Do NOT repeat the conclusion.
6. Each scene should normally contain 4-{MAX_SCENE_WORDS} words.
7. Keep each visual beat short and easy to display for 2–5 seconds.
8. Every scene must move the story forward.
9. Do not invent historical information.
10. Do not add narration that is not present in the supplied narration.
11. Preserve the original meaning and wording as much as possible.
12. Create visually different scenes whenever the narration changes subject.

CAMERA OPTIONS:
{CAMERAS}

TRANSITION OPTIONS:
{TRANSITIONS}

VISUAL BIBLE:
{json.dumps(vb, ensure_ascii=False)}

Return ONLY this JSON:

{{
  "scenes": [
    {{
      "scene_id": 1,
      "narration": "...",
      "visual_focus": "...",
      "camera": "zoom_in",
      "transition": "hard_cut",
      "historical_role": "..."
    }}
  ]
}}

DOCUMENTARY NARRATION:

{narration}
"""


def repair_prompt(scenes, narration):
    return f"""
Repair this documentary scene breakdown.

The original narration is:

{narration}

Current scene breakdown:

{json.dumps(scenes, ensure_ascii=False)}

Create a corrected scene list.

Requirements:

- Return between {MIN_SCENES} and {MAX_SCENES} scenes.
- Target approximately {TARGET_SCENES}.
- Every scene must have narration.
- Every scene narration must contain between
  {MIN_SCENE_WORDS} and {MAX_SCENE_WORDS} words where possible.
- Do not repeat narration.
- Do not repeat facts.
- Do not invent facts.
- Do not add new information.
- Preserve the original narration.
- Divide the narration into logical visual beats.
- Each scene should have a different visual focus.
- Return ONLY valid JSON.

Format:

{{
  "scenes": [
    {{
      "scene_id": 1,
      "narration": "...",
      "visual_focus": "...",
      "camera": "zoom_in",
      "transition": "hard_cut",
      "historical_role": "..."
    }}
  ]
}}
"""


def validate_scenes(scenes, original_narration):
    if not isinstance(scenes, list):
        return False, "scene output is not a list"

    if not (MIN_SCENES <= len(scenes) <= MAX_SCENES):
        return False, f"invalid scene count: {len(scenes)}"

    narration_parts = []

    for scene in scenes:
        if not isinstance(scene, dict):
            return False, "scene is not an object"

        text = str(scene.get("narration", "")).strip()

        if not text:
            return False, "scene has empty narration"

        wc = word_count(text)

        if wc > MAX_SCENE_WORDS:
            return False, (
                f"scene narration too long: {wc} words"
            )

        narration_parts.append(normalize_text(text))

    # Exact duplicate detection.
    if len(narration_parts) != len(set(narration_parts)):
        return False, "duplicate scene narration"

    # Obvious near-duplicate detection.
    for i in range(len(narration_parts)):
        for j in range(i + 1, len(narration_parts)):
            if similarity(
                narration_parts[i],
                narration_parts[j],
            ) >= 0.82:
                return False, (
                    f"near-duplicate narration scenes "
                    f"{i + 1} and {j + 1}"
                )

    return True, "valid"


def make_image_prompt(focus, vb):
    period = vb.get("period", "")
    region = vb.get("region", "")
    style = vb.get("style", "")
    lighting = vb.get(
        "lighting",
        "dramatic natural lighting",
    )

    parts = [
        focus,
        period,
        region,
        style,
        lighting,
    ]

    return ", ".join(
        str(x).strip()
        for x in parts
        if str(x).strip()
    )


def normalize_scene(scene, scene_id, vb):
    focus = str(
        scene.get(
            "visual_focus",
            "historical scene",
        )
    ).strip()

    narration = str(
        scene.get("narration", "")
    ).strip()

    camera = scene.get("camera", "slow_push")

    if camera not in CAMERAS:
        camera = CAMERAS[
            (scene_id - 1) % len(CAMERAS)
        ]

    transition = scene.get(
        "transition",
        "hard_cut",
    )

    if transition not in TRANSITIONS:
        transition = "hard_cut"

    return {
        "scene_id": scene_id,
        "narration": narration,
        "visual_focus": focus,
        "camera": camera,
        "transition": transition,
        "historical_role": str(
            scene.get(
                "historical_role",
                "",
            )
        ).strip(),
        "image_prompt": make_image_prompt(
            focus,
            vb,
        ),
    }


def run(paths, job_id, narration, vb, config, qwen):

    print(
        f"[SCENES] {job_id} | "
        f"STAGE 1/3 | Planning {TARGET_SCENES} visual beats"
    )

    prompt = build_prompt(
        narration,
        vb,
    )

    last_error = ""

    for attempt in range(1, 5):

        print(
            f"[SCENES] {job_id} | "
            f"Attempt {attempt}/4"
        )

        try:
            raw = qwen.generate_json(
                prompt,
                max_new_tokens=1800,
            )

            if isinstance(raw, dict):
                scenes = raw.get(
                    "scenes",
                    [],
                )
            else:
                scenes = raw

            valid, reason = validate_scenes(
                scenes,
                narration,
            )

            if valid:

                print(
                    f"[SCENES] {job_id} | "
                    f"Valid scene count: {len(scenes)}"
                )

                break

            last_error = reason

            print(
                f"[SCENES] {job_id} | "
                f"Invalid: {reason}"
            )

            # Ask Qwen specifically to repair
            # the existing breakdown.
            if attempt < 4:

                repair = qwen.generate_json(
                    repair_prompt(
                        scenes,
                        narration,
                    ),
                    max_new_tokens=1600,
                )

                if isinstance(repair, dict):
                    repaired = repair.get(
                        "scenes",
                        [],
                    )
                else:
                    repaired = repair

                valid, reason = validate_scenes(
                    repaired,
                    narration,
                )

                if valid:
                    scenes = repaired

                    print(
                        f"[SCENES] {job_id} | "
                        f"Repair successful: "
                        f"{len(scenes)} scenes"
                    )

                    break

                last_error = reason

        except Exception as e:
            last_error = str(e)

            print(
                f"[SCENES] {job_id} | "
                f"Attempt error: {e}"
            )

    else:
        raise ValueError(
            f"Scene planning failed after 4 attempts: "
            f"{last_error}"
        )

    # ---------------------------------------------------------
    # FINAL NORMALIZATION
    # ---------------------------------------------------------

    output = []

    for i, scene in enumerate(
        scenes,
        start=1,
    ):
        output.append(
            normalize_scene(
                scene,
                i,
                vb,
            )
        )

    # Final duplicate check.
    narrations = [
        normalize_text(x["narration"])
        for x in output
    ]

    if len(narrations) != len(set(narrations)):
        raise ValueError(
            "Scene planner produced duplicate narration "
            "after normalization."
        )

    print(
        f"[SCENES] {job_id} | "
        f"STAGE 2/3 | Creating image prompts"
    )

    write_json_atomic(
        paths.scenes(job_id),
        output,
    )

    print(
        f"[SCENES] {job_id} | "
        f"STAGE 3/3 | COMPLETE | "
        f"{len(output)} scenes"
    )

    return output
