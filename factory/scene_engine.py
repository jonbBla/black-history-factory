import json

from .utils import write_json_atomic


CAMERAS = [
    "zoom_in",
    "zoom_out",
    "pan_left",
    "pan_right",
    "slow_push",
    "slow_pull"
]


def normalize(value):
    return " ".join(
        str(value or "").lower().split()
    ).strip()


def validate_scenes(scenes, config):

    if not isinstance(scenes, list):
        return False, "Output is not a scene list."

    if not (
        config.scene_count_min
        <= len(scenes)
        <= config.scene_count_max
    ):
        return False, (
            f"Scene count {len(scenes)} is outside "
            f"{config.scene_count_min}-"
            f"{config.scene_count_max}."
        )

    narration_seen = set()
    visual_seen = set()

    for scene in scenes:

        required = [
            "narration",
            "visual_focus",
            "camera",
            "transition",
            "historical_role"
        ]

        for key in required:
            if not str(
                scene.get(key, "")
            ).strip():
                return False, f"Missing {key}."

        narration = normalize(
            scene["narration"]
        )

        visual = normalize(
            scene["visual_focus"]
        )

        # Exact duplicate detection.
        if narration in narration_seen:
            return False, (
                "Repeated scene narration detected."
            )

        if visual in visual_seen:
            return False, (
                "Repeated visual focus detected."
            )

        narration_seen.add(narration)
        visual_seen.add(visual)

        # Individual scene narration should stay short.
        words = len(
            str(scene["narration"]).split()
        )

        if words > 16:
            return False, (
                f"Scene narration is too long "
                f"({words} words)."
            )

    return True, "OK"


def build_prompt(
    narration,
    visual_bible,
    config,
    correction=""
):

    return f"""
You are the visual scene planner for a
fast-paced historical documentary.

Break the narration into
{config.scene_count_min}-{config.scene_count_max}
UNIQUE visual scenes.

The narration is ONE continuous story.

CRITICAL RULES:

1. Preserve the original narration order.
2. Every part of the narration must be represented.
3. Every narration segment must appear ONCE.
4. NEVER repeat the hook later.
5. NEVER repeat a sentence.
6. NEVER repeat a fact unnecessarily.
7. NEVER repeat the same event.
8. NEVER restart the story.
9. NEVER create a second conclusion.
10. NEVER create a source card.
11. NEVER create citations.
12. NEVER create text cards.
13. Do not invent unsupported historical details.
14. Each scene should advance the story.
15. Keep each scene narration to 16 words or fewer.
16. Prefer approximately 4-12 spoken words per scene.
17. Use different visual focuses whenever possible.
18. Use varied camera movements.
19. The final scene should conclude the story naturally.

The final source card is handled separately by the
video processor.

PREFERRED VISUAL TYPES:

- people
- architecture
- landscapes
- artifacts
- tools
- maps
- environments
- historical reconstruction
- close-ups
- daily life
- evidence
- ceremonies
- technology
- craftsmanship
- trade
- interiors

Do NOT force all categories into the video.
Choose visuals that actually correspond to the narration.

ALLOWED CAMERAS:

{CAMERAS}

Return ONLY valid JSON in this exact structure:

{{
  "scenes": [
    {{
      "scene_id": 1,
      "narration": "short narration beat",
      "visual_focus": "specific visual",
      "camera": "zoom_in",
      "transition": "hard_cut",
      "historical_role": "what this scene communicates"
    }}
  ]
}}

{correction}

FULL NARRATION:

{narration}

VISUAL BIBLE:

{json.dumps(
    visual_bible,
    ensure_ascii=False
)}
"""


def run(
    paths, 
    job_id, 
    topic, 
    verified, 
    config, 
    qwen
):

    last_error = ""

    # Up to three attempts.
    for attempt in range(3):

        correction = ""

        if attempt:

            correction = f"""
The previous scene plan FAILED validation.

Reason:
{last_error}

Regenerate the ENTIRE scene list.

Do not patch the previous result.

You MUST remove all repeated narration
and repeated visual focuses.

Make every scene advance the story.
"""

        prompt = build_prompt(
            narration,
            vb,
            config,
            correction=correction
        )

        raw = qwen.generate_json(
            prompt,
            max_new_tokens=4500
        )

        if isinstance(raw, dict):
            scenes = raw.get(
                "scenes",
                []
            )
        else:
            scenes = raw

        valid, error = validate_scenes(
            scenes,
            config
        )

        if not valid:
            last_error = error
            continue

        output = []

        for index, scene in enumerate(
            scenes,
            start=1
        ):

            focus = str(
                scene.get(
                    "visual_focus",
                    "historical scene"
                )
            ).strip()

            camera = scene.get(
                "camera"
            )

            if camera not in CAMERAS:
                camera = CAMERAS[
                    (index - 1) % len(CAMERAS)
                ]

            period = vb.get(
                "period",
                ""
            )

            region = vb.get(
                "region",
                ""
            )

            style = vb.get(
                "style",
                ""
            )

            lighting = vb.get(
                "lighting",
                "dramatic natural light"
            )

            image_prompt = (
                f"{focus}, "
                f"{period}, "
                f"{region}, "
                f"{style}, "
                f"{lighting}"
            )

            output.append(
                {
                    "scene_id": index,
                    "narration": str(
                        scene["narration"]
                    ).strip(),
                    "visual_focus": focus,
                    "camera": camera,
                    "transition": str(
                        scene.get(
                            "transition",
                            "hard_cut"
                        )
                    ).strip(),
                    "historical_role": str(
                        scene.get(
                            "historical_role",
                            ""
                        )
                    ).strip(),
                    "image_prompt": image_prompt
                }
            )

        write_json_atomic(
            paths.scenes(job_id),
            output
        )

        return output

    raise ValueError(
        "Scene planning failed after 3 attempts: "
        + last_error
    )
