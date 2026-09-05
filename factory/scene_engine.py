from __future__ import annotations

import json
import os
import re

from .utils import write_json_atomic


CAMERAS = {
    "slow_push",
    "zoom_in",
    "zoom_out",
    "pan_left",
    "pan_right",
    "slow_pull",
}


def count_words(text):
    return len(re.findall(r"\b[\w’'-]+\b", text or ""))


def split_sentences(text):
    """
    Split narration into numbered sentences while preserving
    the original narration text.
    """
    text = re.sub(r"\s+", " ", (text or "").strip())

    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)

    return [
        {
            "sentence_id": i,
            "text": part.strip(),
        }
        for i, part in enumerate(parts, 1)
        if part.strip()
    ]


def _prompt(narration, research, fact_check, config):
    style = getattr(
        config,
        "art_style_text",
        "cinematic 3D historical reconstruction",
    )

    sentences = split_sentences(narration)

    numbered_narration = "\n".join(
        f"[SENTENCE {s['sentence_id']}] {s['text']}"
        for s in sentences
    )

    return f"""You are the cinematic scene director for a short
historical documentary.

Your job is to transform the narration into an intelligent sequence
of visual scenes.

IMPORTANT:
There is NO fixed scene count.

Determine the appropriate number of scenes yourself based on the
content and visual changes in the narration.

Do NOT create extra scenes simply to reach a target number.

NARRATION SENTENCES
{numbered_narration}

RESEARCH
{json.dumps(research, ensure_ascii=False, indent=2)}

FACT CHECK
{json.dumps(fact_check, ensure_ascii=False, indent=2)}

SCENE RULES

1. Every narration sentence must belong to exactly one scene.

2. Scenes must follow the original narration order.

3. Use consecutive sentence ranges only.

4. Create a new scene when the visual subject meaningfully changes.

5. Keep related narration together when one visual can effectively
   represent it.

6. Do not split narration mechanically.

7. Do not combine unrelated subjects into one scene.

8. Create as many scenes as necessary, but no more than needed.

9. Short narration may naturally produce only a few scenes.

10. Long narration may naturally produce many scenes.

11. Do NOT rewrite, paraphrase, summarize, or reproduce the narration.

12. Python will attach the original narration to each scene.

13. Every scene must have a detailed visual description suitable
    for direct use by an image-generation model.

14. Visual descriptions should intelligently incorporate relevant
    details from the research, including where appropriate:

    - historically appropriate people
    - clothing and textiles
    - architecture
    - buildings
    - tools and technology
    - materials
    - environment
    - geography
    - objects
    - food and daily life
    - religion and belief
    - mythology and oral tradition
    - trade and economy
    - art
    - archaeology

15. Do not invent unsupported historical details.

16. Do not use generic African stereotypes.

17. Do not use modern objects in historical scenes unless appropriate
    to the historical period.

18. Do not add text, captions, logos, watermarks, UI elements,
    or modern graphic overlays to the image.

19. Make each visual description specific enough that an image model
    can produce a useful historical reconstruction.

20. Think like a documentary cinematographer, not a slideshow designer.

ART STYLE
{style}

Return ONLY valid JSON.

Required format:

{{
  "scenes": [
    {{
      "scene_id": 1,
      "sentence_start": 1,
      "sentence_end": 2,
      "visual_description": "Detailed cinematic visual description...",
      "camera": "slow_push"
    }}
  ]
}}

The scene_id must start at 1 and increase sequentially.

camera must be exactly one of:

slow_push
zoom_in
zoom_out
pan_left
pan_right
slow_pull
"""


def _extract_json(text):
    text = (text or "").strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")

    if start < 0:
        raise ValueError("No JSON object found.")

    depth = 0
    inside_string = False
    escaped = False

    for i in range(start, len(text)):
        c = text[i]

        if inside_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                inside_string = False

        else:
            if c == '"':
                inside_string = True

            elif c == "{":
                depth += 1

            elif c == "}":
                depth -= 1

                if depth == 0:
                    return json.loads(text[start:i + 1])

    raise ValueError("Incomplete JSON object.")


def _validate_ranges(data, sentences):
    if not isinstance(data, dict):
        raise ValueError("Scene response must be an object.")

    scenes = data.get("scenes")

    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Missing scenes list.")

    total_sentences = len(sentences)

    expected_start = 1
    validated = []

    for i, scene in enumerate(scenes, 1):

        if not isinstance(scene, dict):
            raise ValueError(f"Scene {i} is invalid.")

        scene_id = int(scene.get("scene_id", -1))

        if scene_id != i:
            raise ValueError(
                f"Scene {i} has wrong scene_id."
            )

        start = int(scene.get("sentence_start", -1))
        end = int(scene.get("sentence_end", -1))

        if start != expected_start:
            raise ValueError(
                f"Scene {i} starts at sentence {start}; "
                f"expected {expected_start}."
            )

        if end < start:
            raise ValueError(
                f"Scene {i} has invalid sentence range."
            )

        if end > total_sentences:
            raise ValueError(
                f"Scene {i} ends beyond narration."
            )

        visual = str(
            scene.get("visual_description", "")
        ).strip()

        if not visual:
            raise ValueError(
                f"Scene {i} missing visual_description."
            )

        camera = str(
            scene.get("camera", "")
        ).strip()

        if camera not in CAMERAS:
            raise ValueError(
                f"Scene {i} has invalid camera: {camera}"
            )

        validated.append({
            "scene_id": i,
            "sentence_start": start,
            "sentence_end": end,
            "visual_description": visual,
            "camera": camera,
        })

        expected_start = end + 1

    if expected_start != total_sentences + 1:
        raise ValueError(
            "Narration coverage incomplete. "
            f"Last covered sentence: {expected_start - 1}; "
            f"total: {total_sentences}."
        )

    return validated


def run(
    paths,
    job_id,
    narration,
    research,
    fact_check,
    config,
    qwen,
):
    os.makedirs(
        os.path.dirname(paths.scenes(job_id)),
        exist_ok=True,
    )

    sentences = split_sentences(narration)

    if not sentences:
        raise ValueError("Narration is empty.")

    prompt = _prompt(
        narration,
        research,
        fact_check,
        config,
    )

    print(
        f"[SCENES] Qwen scene director | "
        f"{len(sentences)} narration sentences | "
        f"scene count decided by Qwen"
    )

    # Small output because Qwen no longer needs to reproduce narration.
    data = qwen.generate_json(
        prompt,
        max_new_tokens=2600,
        retries=1,
    )

    scenes = _validate_ranges(
        data,
        sentences,
    )

    final = []

    for scene in scenes:

        start = scene["sentence_start"]
        end = scene["sentence_end"]

        # IMPORTANT:
        # Use the original narration rather than anything generated
        # by Qwen. This guarantees exact narration preservation.
        original_narration = " ".join(
            s["text"]
            for s in sentences[start - 1:end]
        ).strip()

        visual = scene["visual_description"]

        final.append({
            "scene_id": scene["scene_id"],
            "scene_number": scene["scene_id"],
            "narration": original_narration,
            "word_count": count_words(original_narration),
            "visual_description": visual,

            # Compatibility aliases for the Image Processor.
            "image_prompt": visual,
            "image_description": visual,

            "camera": scene["camera"],
        })

    write_json_atomic(
        paths.scenes(job_id),
        final,
    )

    print(
        f"[SCENES] COMPLETE | {len(final)} scenes"
    )

    return final
