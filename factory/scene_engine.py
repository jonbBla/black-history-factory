import json
import re


MIN_SCENE_WORDS = 4
MAX_SCENE_WORDS = 14
MIN_SCENES = 18
MAX_SCENES = 22


def word_count(text):
    return len(re.findall(r"\b[\w’'-]+\b", text))


def clean_json_text(text):
    """
    Remove accidental markdown fences if Qwen returns them.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def validate_scene_plan(data, narration):
    """
    Validate Qwen's scene plan.

    Python validates structure.
    Python does NOT create or rewrite scenes.
    """

    if not isinstance(data, dict):
        raise ValueError("Scene planner output must be an object.")

    scenes = data.get("scenes")

    if not isinstance(scenes, list):
        raise ValueError("Scene planner output has no 'scenes' list.")

    if not (MIN_SCENES <= len(scenes) <= MAX_SCENES):
        raise ValueError(
            f"Invalid scene count: {len(scenes)}. "
            f"Required {MIN_SCENES}-{MAX_SCENES}."
        )

    for i, scene in enumerate(scenes, 1):

        if not isinstance(scene, dict):
            raise ValueError(
                f"Scene {i} is not an object."
            )

        if "narration" not in scene:
            raise ValueError(
                f"Scene {i} is missing narration."
            )

        if "visual_beat" not in scene:
            raise ValueError(
                f"Scene {i} is missing visual_beat."
            )

        scene_narration = str(scene["narration"]).strip()

        words = word_count(scene_narration)

        if words < MIN_SCENE_WORDS:
            raise ValueError(
                f"Scene {i} too short: {words} words."
            )

        if words > MAX_SCENE_WORDS:
            raise ValueError(
                f"Scene {i} too long: {words} words."
            )

        if not scene["visual_beat"].strip():
            raise ValueError(
                f"Scene {i} has an empty visual beat."
            )

    # --------------------------------------------------------
    # Verify that the scene narration represents the original
    # narration rather than invented/re-written narration.
    #
    # We normalize whitespace only.
    # --------------------------------------------------------

    original = re.sub(
        r"\s+",
        " ",
        narration.strip()
    )

    planned = " ".join(
        str(scene["narration"]).strip()
        for scene in scenes
    )

    planned = re.sub(r"\s+", " ", planned)

    if original != planned:
        raise ValueError(
            "Scene narration does not exactly preserve "
            "the accepted narration."
        )

    return True


def generate_scene_plan(
    qwen,
    narration,
    visual_bible,
    config,
    attempt=1,
):
    """
    Ask Qwen to intelligently divide the narration into scenes.

    Qwen decides:
      - scene boundaries
      - scene count
      - visual beat for each scene

    Python only validates the result.
    """

    total_words = word_count(narration)

    prompt = f"""
You are the scene director for a fast-paced historical documentary.

You have already been given a completed narration.

Your job is to intelligently divide the EXACT narration into
visual scenes.

NARRATION:
{narration}

VISUAL BIBLE:
{visual_bible}

TOTAL NARRATION WORDS:
{total_words}

SCENE REQUIREMENTS:

1. Create between 18 and 22 scenes.
2. Each scene must contain between 4 and 14 narration words.
3. Preserve the narration EXACTLY.
4. Do NOT rewrite the narration.
5. Do NOT summarize the narration.
6. Do NOT add words.
7. Do NOT remove words.
8. Do NOT change word order.
9. Do NOT merge words.
10. Do NOT invent narration.
11. Every word from the narration must appear exactly once.
12. Scene narration must remain in the original order.

INTELLIGENT SCENE BOUNDARIES:

Prefer boundaries that make cinematic sense.

Strongly prefer:
- sentence endings
- natural pauses
- commas
- semicolons
- clauses
- changes in subject
- changes in action
- changes in location
- changes in time
- important visual reveals

Do NOT mechanically create scenes of equal length.

A scene can contain 5 words.
Another can contain 9.
Another can contain 12.
Another can contain 14.

The goal is NATURAL CINEMATIC PACING, not mathematical uniformity.

VISUAL BEATS:

For every scene, create a concise visual beat describing
what the audience should see during that narration.

The visual beat should be:
- visually specific
- historically appropriate
- connected directly to the narration
- consistent with the visual bible
- suitable for a cinematic 3D historical reconstruction

Do not write image-generation prompts yet.

Return ONLY valid JSON.

Required format:

{{
  "scenes": [
    {{
      "scene_number": 1,
      "narration": "exact words from narration",
      "visual_beat": "what should visually happen"
    }}
  ]
}}
"""

    result = qwen.generate(
        prompt,
        max_new_tokens=1800,
        temperature=0.25,
    )

    result = clean_json_text(result)

    try:
        data = json.loads(result)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Scene planner returned invalid JSON: {e}"
        )

    validate_scene_plan(data, narration)

    return data


def generate_image_description(
    qwen,
    scene,
    visual_bible,
    config,
):
    """
    Convert the scene's visual beat into a detailed
    SDXL-ready image description.

    Qwen does NOT control scene structure here.
    """

    prompt = f"""
You are creating a cinematic image-generation description
for a historical documentary.

SCENE NARRATION:
{scene["narration"]}

VISUAL BEAT:
{scene["visual_beat"]}

VISUAL BIBLE:
{visual_bible}

Create one detailed image description for SDXL.

The image must visually communicate the narration and visual beat.

Include when appropriate:
- people
- clothing
- physical appearance
- architecture
- landscape
- objects
- materials
- period-appropriate technology
- activities
- body positions
- facial expressions
- environmental details
- lighting
- atmosphere
- camera composition
- depth
- scale

STYLE:

cinematic 3D historical reconstruction,
epic cinematic historical reconstruction,
physically plausible materials,
period-authentic details,
dramatic natural lighting,
volumetric atmosphere,
strong depth,
detailed surfaces,
cinematic composition,
realistic proportions,
highly detailed environments,
realistic textures,
dramatic scale,
high-end game cinematic,
Unreal Engine style.

Avoid:
- modern objects
- modern clothing
- anachronisms
- flat cartoon appearance
- generic fantasy elements
- text
- logos
- watermarks

Return ONLY the image description.
"""

    result = qwen.generate(
        prompt,
        max_new_tokens=700,
        temperature=0.35,
    )

    return result.strip()


def run(paths, job_id, narration, visual_bible, config, qwen):

    print(
        f"[QWEN] SCENES {job_id} | "
        "INTELLIGENT SCENE PLANNING"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT modify narration here.
    # --------------------------------------------------------

    narration = narration.strip()

    print(
        f"[SCENES] EXACT NARRATION: "
        f"{word_count(narration)} words"
    )

    # --------------------------------------------------------
    # Scene planning retries.
    # Qwen gets multiple chances to correct its own plan.
    # --------------------------------------------------------

    scene_plan = None

    for attempt in range(1, 5):

        print(
            f"[SCENES] PLAN ATTEMPT "
            f"{attempt}/4"
        )

        try:

            scene_plan = generate_scene_plan(
                qwen=qwen,
                narration=narration,
                visual_bible=visual_bible,
                config=config,
                attempt=attempt,
            )

            print(
                f"[SCENES] PLAN ACCEPTED | "
                f"{len(scene_plan['scenes'])} scenes"
            )

            break

        except Exception as e:

            print(
                f"[SCENES] PLAN ATTEMPT {attempt} "
                f"FAILED | {e}"
            )

            if attempt == 4:
                raise ValueError(
                    "Scene planning failed after 4 attempts."
                )


    # --------------------------------------------------------
    # Generate detailed image descriptions.
    # --------------------------------------------------------

    final_scenes = []

    total = len(scene_plan["scenes"])

    for index, scene in enumerate(
        scene_plan["scenes"],
        1
    ):

        scene_number = scene["scene_number"]

        print(
            f"[QWEN] IMAGE DESCRIPTION "
            f"{index}/{total} | SCENE {scene_number}"
        )

        image_description = generate_image_description(
            qwen=qwen,
            scene=scene,
            visual_bible=visual_bible,
            config=config,
        )

        if not image_description:
            raise ValueError(
                f"Scene {scene_number} returned "
                "an empty image description."
            )

        final_scenes.append({
            "scene_number": scene_number,
            "narration": scene["narration"],
            "word_count": word_count(
                scene["narration"]
            ),
            "visual_beat": scene["visual_beat"],
            "image_description": image_description,
        })


    # --------------------------------------------------------
    # Final validation.
    # --------------------------------------------------------

    if not (18 <= len(final_scenes) <= 22):
        raise ValueError(
            f"Invalid final scene count: "
            f"{len(final_scenes)}"
        )

    combined = " ".join(
        scene["narration"]
        for scene in final_scenes
    )

    combined = re.sub(
        r"\s+",
        " ",
        combined.strip()
    )

    original = re.sub(
        r"\s+",
        " ",
        narration.strip()
    )

    if combined != original:
        raise ValueError(
            "Final scene narration does not exactly "
            "match the accepted narration."
        )

    for scene in final_scenes:

        words = word_count(scene["narration"])

        if not (
            MIN_SCENE_WORDS
            <= words
            <= MAX_SCENE_WORDS
        ):
            raise ValueError(
                f"Scene {scene['scene_number']} "
                f"has {words} words."
            )

    print(
        f"[QWEN] SCENES {job_id} | "
        f"COMPLETE | {len(final_scenes)} scenes"
    )

    return {
        "scene_count": len(final_scenes),
        "scenes": final_scenes,
    }
