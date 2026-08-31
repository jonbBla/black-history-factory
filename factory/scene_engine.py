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


TRANSITIONS = [
    "hard_cut",
    "hard_cut",
    "hard_cut",
    "crossfade"
]


def normalize(text):

    return " ".join(
        str(text or "")
        .lower()
        .strip()
        .split()
    )


def similarity(a, b):

    a_words = set(
        normalize(a).split()
    )

    b_words = set(
        normalize(b).split()
    )

    if not a_words or not b_words:
        return 0.0

    return len(
        a_words & b_words
    ) / len(
        a_words | b_words
    )


def validate_scenes(
    scenes,
    config
):

    if not isinstance(
        scenes,
        list
    ):

        return False, (
            "Scene output is not a list."
        )


    if not (
        config.scene_count_min
        <= len(scenes)
        <= config.scene_count_max
    ):

        return False, (
            f"Scene count {len(scenes)} "
            f"is outside "
            f"{config.scene_count_min}-"
            f"{config.scene_count_max}."
        )


    narrations = []
    visuals = []


    for index, scene in enumerate(
        scenes,
        start=1
    ):

        if not isinstance(
            scene,
            dict
        ):

            return False, (
                f"Scene {index} is not an object."
            )


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

                return False, (
                    f"Scene {index} "
                    f"is missing {key}."
                )


        narration = normalize(
            scene["narration"]
        )

        visual = normalize(
            scene["visual_focus"]
        )


        if narration in narrations:

            return False, (
                f"Scene {index} repeats "
                "previous narration."
            )


        if visual in visuals:

            return False, (
                f"Scene {index} repeats "
                "previous visual."
            )


        # Prevent near-identical narration.
        for previous in narrations:

            if similarity(
                narration,
                previous
            ) >= 0.82:

                return False, (
                    f"Scene {index} is too "
                    "similar to another scene."
                )


        words = len(
            str(
                scene["narration"]
            ).split()
        )


        if words > 16:

            return False, (
                f"Scene {index} contains "
                f"{words} words. "
                "Maximum is 16."
            )


        narrations.append(
            narration
        )

        visuals.append(
            visual
        )


    return True, "OK"


def build_prompt(
    narration,
    visual_bible,
    config,
    correction=""
):

    return f"""
You are the scene planner for a fast-paced
historical documentary.

Turn the following COMPLETE narration into
{config.scene_count_min}-{config.scene_count_max}
UNIQUE visual scenes.

The narration is approximately
{config.narration_words_min}-{config.narration_words_max}
words.

IMPORTANT:

Each scene is a SHORT VISUAL BEAT.

Each scene narration must contain
4-16 spoken words.

The entire narration must be covered from
beginning to end.

DO NOT duplicate narration.

DO NOT duplicate the hook.

DO NOT repeat facts.

DO NOT restart the story.

DO NOT create a second conclusion.

DO NOT simply copy the same sentences twice.

Every scene must advance the story.

A scene may combine a few adjacent words
from the narration, but it must preserve the
original meaning and chronological order.

VISUAL RULES:

Every scene should have a meaningfully
different visual focus.

Prefer:

- people
- architecture
- interiors
- landscapes
- artifacts
- tools
- craftsmanship
- maps
- settlements
- ceremonies
- daily life
- trade
- technology
- archaeological evidence
- environmental details
- close-ups
- wide establishing shots

Do not invent unsupported historical details.

Do not add an artifact merely because it
would look interesting.

Use the research-supported visual bible.

CAMERA OPTIONS:

{CAMERAS}

TRANSITIONS:

{TRANSITIONS}

Use hard cuts frequently.
Use crossfade mainly for changes of
time, location or mood.

The final scene should finish the documentary.

DO NOT create the source card.
The video processor will create the source card.

Return ONLY:

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
    narration,
    vb,
    config,
    qwen
):

    last_error = ""

    for attempt in range(3):

        correction = ""

        if attempt:

            correction = f"""
The previous scene plan FAILED.

Reason:
{last_error}

Generate the ENTIRE scene plan again.

Do not modify only one scene.

The new version must:

- contain no repeated narration
- contain no repeated visuals
- contain no repeated hook
- contain no second conclusion
- keep every scene under 16 words
- cover the complete narration
- move forward chronologically
"""


        prompt = build_prompt(
            narration,
            vb,
            config,
            correction
        )


        raw = qwen.generate_json(
            prompt,
            max_new_tokens=4200
        )


        if isinstance(
            raw,
            dict
        ):

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
                    ""
                )
            ).strip()


            camera = scene.get(
                "camera"
            )

            if camera not in CAMERAS:

                camera = CAMERAS[
                    (index - 1)
                    % len(CAMERAS)
                ]


            transition = scene.get(
                "transition",
                "hard_cut"
            )

            if transition not in (
                "hard_cut",
                "crossfade"
            ):

                transition = "hard_cut"


            period = str(
                vb.get(
                    "period",
                    ""
                )
            ).strip()


            region = str(
                vb.get(
                    "region",
                    ""
                )
            ).strip()


            style = str(
                vb.get(
                    "style",
                    ""
                )
            ).strip()


            lighting = str(
                vb.get(
                    "lighting",
                    ""
                )
            ).strip()


            architecture = str(
                vb.get(
                    "architecture",
                    ""
                )
            ).strip()


            clothing = str(
                vb.get(
                    "clothing",
                    ""
                )
            ).strip()


            materials = str(
                vb.get(
                    "materials",
                    ""
                )
            ).strip()


            environment = str(
                vb.get(
                    "environment",
                    ""
                )
            ).strip()


            people = str(
                vb.get(
                    "people",
                    ""
                )
            ).strip()


            image_prompt = (
                f"{focus}, "
                f"{period}, "
                f"{region}, "
                f"{architecture}, "
                f"{clothing}, "
                f"{materials}, "
                f"{environment}, "
                f"{people}, "
                f"{style}, "
                f"{lighting}, "
                "highly detailed environments, "
                "realistic textures, "
                "dramatic scale, "
                "realistic proportions"
            )


            output.append(
                {
                    "scene_id": index,

                    "narration": str(
                        scene["narration"]
                    ).strip(),

                    "visual_focus": focus,

                    "camera": camera,

                    "transition": transition,

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
        "Scene planning failed after "
        f"3 attempts: {last_error}"
    )
