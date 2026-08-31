"""
Visual bible generation.

The visual bible establishes consistent visual rules for every scene
within a documentary:

- period
- region
- architecture
- clothing
- materials
- environment
- people
- lighting
- style

The series-wide art style always comes from config.art_style.
Qwen may describe topic-specific visual details, but it cannot
override the configured art style.
"""

from __future__ import annotations

import json
import os

from .utils import write_json_atomic


PROMPTS_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    "prompts"
)


DEFAULT_STYLE = (
    "epic cinematic historical reconstruction, "
    "physically plausible materials, "
    "period-authentic details, "
    "dramatic natural lighting, "
    "volumetric atmosphere, "
    "strong depth, "
    "detailed surfaces, "
    "cinematic composition, "
    "realistic proportions"
)


DEFAULT_LIGHTING = (
    "dramatic natural lighting, "
    "volumetric atmosphere, "
    "strong depth and realistic shadows"
)


def _load_template(name: str) -> str:

    path = os.path.join(
        PROMPTS_DIR,
        name
    )

    if not os.path.exists(path):
        return ""

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()


def build_prompt(
    topic,
    research: dict,
    art_style: str
) -> str:

    template = _load_template(
        "visual_bible.txt"
    )

    if template:

        overview = ""

        if isinstance(research, dict):
            overview = research.get(
                "overview",
                ""
            )

        if not isinstance(
            overview,
            str
        ):
            overview = str(overview)

        try:
            return template.format(
                topic_title=topic.title,
                region=topic.region,
                period=topic.period,
                research_summary=overview[:2500],
                art_style=art_style
            )
        except Exception:
            pass

    # Safe fallback prompt.
    return f"""
Create a visual bible for this historical documentary topic.

Topic:
{topic.title}

Category:
{topic.category}

Region:
{topic.region}

Period:
{topic.period}

Description:
{topic.description}

Research:
{json.dumps(research, ensure_ascii=False)[:5000]}

Locked art style:
{art_style}

Return ONLY valid JSON with these fields:

{{
  "architecture": "...",
  "clothing": "...",
  "materials": "...",
  "environment": "...",
  "people": "...",
  "lighting": "..."
}}

Rules:

- Use historically appropriate details.
- Do not invent unsupported architecture.
- Do not invent clothing.
- Do not invent technologies.
- Do not invent ethnic identities.
- Do not invent physical characteristics.
- If something is unknown, say so.
- Keep mythology separate from established history.
- Keep oral traditions separate from established history.
"""


def run(
    paths,
    job_id,
    topic,
    research,
    config,
    qwen
):

    # ---------------------------------------------------------
    # LOCKED SERIES ART STYLE
    # ---------------------------------------------------------

    art_style = getattr(
        config,
        "art_style",
        None
    )

    if isinstance(
        art_style,
        dict
    ):
        art_style = (
            art_style.get("description")
            or art_style.get("primary")
        )

    art_style = str(
        art_style or DEFAULT_STYLE
    ).strip()


    # ---------------------------------------------------------
    # BASE VISUAL BIBLE
    # ---------------------------------------------------------

    bible = {

        "period": (
            getattr(topic, "period", "")
            or "unspecified"
        ),

        "region": (
            getattr(topic, "region", "")
            or "unspecified"
        ),

        "architecture": "",

        "clothing": "",

        "materials": "",

        "environment": "",

        "people": "",

        "lighting": DEFAULT_LIGHTING,

        # NEVER allow Qwen to change this.
        "style": art_style
    }


    # ---------------------------------------------------------
    # ASK QWEN FOR TOPIC-SPECIFIC DETAILS
    # ---------------------------------------------------------

    if qwen is not None:

        prompt = build_prompt(
            topic,
            research,
            art_style
        )

        try:

            result = qwen.generate_json(
                prompt,
                max_new_tokens=900
            )

        except Exception as exc:

            print(
                "Visual bible generation warning:",
                exc
            )

            result = {}


        if isinstance(
            result,
            dict
        ):

            allowed = [
                "architecture",
                "clothing",
                "materials",
                "environment",
                "people"
            ]

            for key in allowed:

                value = result.get(key)

                if isinstance(
                    value,
                    str
                ) and value.strip():

                    bible[key] = value.strip()


            lighting = result.get(
                "lighting"
            )

            if isinstance(
                lighting,
                str
            ) and lighting.strip():

                bible["lighting"] = (
                    lighting.strip()
                )


    # ---------------------------------------------------------
    # FALLBACKS
    # ---------------------------------------------------------

    for key in [
        "architecture",
        "clothing",
        "materials",
        "environment",
        "people"
    ]:

        if not bible[key]:

            bible[key] = (
                "Use only historically supported "
                "details from the research."
            )


    # ---------------------------------------------------------
    # SAVE TO DRIVE
    # ---------------------------------------------------------

    write_json_atomic(
        paths.visual_bible(job_id),
        bible
    )

    return bible
