import json
import re

from factory.utils import write_json_atomic


# ============================================================
# SCENE CONFIGURATION
# ============================================================

TARGET_SCENES = 20
MIN_SCENES = 18
MAX_SCENES = 22

MIN_SCENE_WORDS = 4
MAX_SCENE_WORDS = 14

SEGMENT_ATTEMPTS = 4
VISUAL_ATTEMPTS = 3


# ============================================================
# TEXT UTILITIES
# ============================================================

def count_words(text):
    """
    Consistent word counter used throughout the scene engine.
    """
    return len(re.findall(r"\b[\w’'-]+\b", str(text)))


def clean_text(text):
    """
    Clean model output without changing its meaning.
    """
    text = str(text).strip()

    # Remove markdown/code fences.
    text = re.sub(r"```(?:text|json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")

    # Remove accidental scene labels.
    text = re.sub(
        r"^\s*(?:scene\s*)?\d+\s*[:.)-]\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def similar(a, b):
    """
    Detect obvious repeated/near-repeated scene narration.
    """
    a_words = set(normalize_text(a).split())
    b_words = set(normalize_text(b).split())

    if not a_words or not b_words:
        return False

    overlap = len(a_words & b_words)
    smaller = min(len(a_words), len(b_words))

    return (overlap / smaller) >= 0.75


# ============================================================
# SEGMENTATION
# ============================================================

def segmentation_prompt(narration):
    """
    Ask Qwen ONLY to divide the existing narration.

    No JSON.
    No visual descriptions.
    No rewriting.
    """

    return f"""
You are dividing an existing documentary narration into short
consecutive narration segments.

DO NOT write a new narration.

DO NOT summarize.

DO NOT add information.

DO NOT remove information.

DO NOT change the wording unnecessarily.

Simply divide the original narration into approximately {TARGET_SCENES}
consecutive pieces.

Requirements:

- Target: {TARGET_SCENES} segments
- Acceptable: {MIN_SCENES}-{MAX_SCENES} segments
- Every segment must contain {MIN_SCENE_WORDS}-{MAX_SCENE_WORDS} words.
- Keep the original order exactly.
- Every segment must come directly from the original narration.
- Do not repeat information.
- Do not create a title.
- Do not create a source card.
- Do not create an introduction that isn't in the narration.
- Do not create an ending that isn't in the narration.

OUTPUT FORMAT:

Put ONE segment on each line.

Do not number the lines.

Do not use bullets.

Do not use JSON.

ORIGINAL NARRATION:

{narration}
"""


def correction_segmentation_prompt(narration, previous_output, error):
    """
    Targeted correction rather than asking the model to reinvent
    the entire task.
    """

    return f"""
Fix the segmentation below.

ORIGINAL NARRATION:
{narration}

PREVIOUS SEGMENTATION:
{previous_output}

PROBLEM:
{error}

Create a corrected segmentation.

Rules:

- Target exactly {TARGET_SCENES} segments.
- Acceptable range: {MIN_SCENES}-{MAX_SCENES}.
- Each segment must contain {MIN_SCENE_WORDS}-{MAX_SCENE_WORDS} words.
- Preserve the original narration and its order.
- Do not rewrite the story.
- Do not add information.
- Do not remove information.
- Do not repeat information.
- Do not number the lines.
- Do not use bullets.
- Do not use JSON.

ONE SEGMENT PER LINE.
"""


def parse_segments(raw):
    """
    Convert plain-text Qwen output into a Python list.

    Handles common model formatting mistakes.
    """

    raw = str(raw).strip()

    # Remove markdown fences.
    raw = re.sub(
        r"```(?:text|txt|json)?",
        "",
        raw,
        flags=re.IGNORECASE
    )
    raw = raw.replace("```", "")

    lines = []

    for line in raw.splitlines():

        line = line.strip()

        if not line:
            continue

        # Remove bullets.
        line = re.sub(r"^[-*•]\s*", "", line)

        # Remove numbered labels.
        line = re.sub(
            r"^\s*(?:scene\s*)?\d+\s*[:.)-]\s*",
            "",
            line,
            flags=re.IGNORECASE
        )

        line = line.strip()

        if line:
            lines.append(line)

    return lines


def validate_segments(segments):
    """
    Validate segmentation independently from visual generation.
    """

    if not isinstance(segments, list):
        return False, "segments is not a list"

    count = len(segments)

    if count < MIN_SCENES or count > MAX_SCENES:
        return False, f"invalid segment count: {count}"

    normalized_seen = []

    for index, segment in enumerate(segments, start=1):

        words = count_words(segment)

        if words < MIN_SCENE_WORDS:
            return False, (
                f"segment {index} too short: "
                f"{words} words"
            )

        if words > MAX_SCENE_WORDS:
            return False, (
                f"segment {index} too long: "
                f"{words} words"
            )

        normalized = normalize_text(segment)

        if normalized in normalized_seen:
            return False, (
                f"duplicate segment at position {index}"
            )

        for previous in normalized_seen:
            if similar(segment, previous):
                return False, (
                    f"near-duplicate segment at position {index}"
                )

        normalized_seen.append(normalized)

    return True, "valid"


# ============================================================
# VISUAL DESCRIPTION GENERATION
# ============================================================

def visual_prompt(segment, visual_bible, scene_number):

    return f"""
Create a cinematic visual description for scene {scene_number} of a
historical documentary.

IMPORTANT:

The narration below is already final.

Do NOT rewrite it.

Do NOT explain the narration.

Do NOT add dialogue.

Do NOT mention the narrator.

Describe ONLY what should be visible on screen.

The visual must directly represent the narration.

NARRATION:

{segment}

VISUAL BIBLE:

{visual_bible}

Create ONE concise but detailed visual description.

Include useful visual information such as:

- people
- clothing
- architecture
- environment
- objects
- actions
- historical setting
- lighting
- atmosphere
- camera/composition

Keep the scene historically plausible.

Avoid modern objects unless the narration specifically requires them.

Do not write a list.

Do not use JSON.

Return ONLY the visual description.
"""


def clean_visual_description(text):
    """
    Clean visual output.
    """

    text = str(text).strip()

    text = re.sub(
        r"```(?:text)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace("```", "")

    # Remove accidental prefixes.
    text = re.sub(
        r"^\s*(?:visual\s*description|description)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


# ============================================================
# BUILD FINAL SCENE
# ============================================================

def build_scene(scene_number, narration, visual_description):

    return {
        "scene_number": scene_number,
        "narration": narration,
        "visual_description": visual_description
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def run(paths, job_id, narration, visual_bible, config, qwen):

    print(
        f"[SCENES] {job_id} | "
        f"TARGET: ~{TARGET_SCENES} scenes | "
        f"acceptable: {MIN_SCENES}-{MAX_SCENES}"
    )

    # ========================================================
    # STAGE 1 — SEGMENT NARRATION
    # ========================================================

    print(
        f"[SCENES] {job_id} | "
        f"STAGE 1/2 | segmenting narration"
    )

    segments = None
    previous_output = None
    last_error = None

    for attempt in range(1, SEGMENT_ATTEMPTS + 1):

        if attempt == 1:

            print(
                f"[SCENES] {job_id} | "
                f"SEGMENT ATTEMPT {attempt}/{SEGMENT_ATTEMPTS}"
            )

            prompt = segmentation_prompt(narration)

        else:

            print(
                f"[SCENES] {job_id} | "
                f"SEGMENT ATTEMPT {attempt}/{SEGMENT_ATTEMPTS} | "
                f"correcting: {last_error}"
            )

            prompt = correction_segmentation_prompt(
                narration,
                previous_output,
                last_error
            )

        try:

            raw = qwen.generate(
                prompt,
                max_new_tokens=700,
                temperature=0.25
            )

            previous_output = raw

            parsed = parse_segments(raw)

            valid, error = validate_segments(parsed)

            if not valid:

                last_error = error

                print(
                    f"[SCENES] {job_id} | "
                    f"Invalid segmentation: {error}"
                )

                continue

            segments = parsed

            print(
                f"[SCENES] {job_id} | "
                f"SEGMENTATION ACCEPTED | "
                f"{len(segments)} segments"
            )

            break

        except Exception as e:

            last_error = str(e)

            print(
                f"[SCENES] {job_id} | "
                f"Segmentation error: {last_error}"
            )

    if segments is None:

        raise RuntimeError(
            f"Scene segmentation failed after "
            f"{SEGMENT_ATTEMPTS} attempts. "
            f"Last error: {last_error}"
        )

    # ========================================================
    # STAGE 2 — VISUAL DESCRIPTIONS
    # ========================================================

    print(
        f"[SCENES] {job_id} | "
        f"STAGE 2/2 | generating visual descriptions"
    )

    scenes = []

    for index, segment in enumerate(segments, start=1):

        print(
            f"[SCENES] {job_id} | "
            f"SCENE {index}/{len(segments)} | "
            f"generating visual description"
        )

        visual_description = None
        last_visual_error = None

        for attempt in range(1, VISUAL_ATTEMPTS + 1):

            try:

                prompt = visual_prompt(
                    segment,
                    visual_bible,
                    index
                )

                raw_visual = qwen.generate(
                    prompt,
                    max_new_tokens=450,
                    temperature=0.35
                )

                visual_description = clean_visual_description(
                    raw_visual
                )

                if not visual_description:

                    last_visual_error = (
                        "empty visual description"
                    )

                    continue

                # Reject obvious model failures.
                if len(visual_description.split()) < 8:

                    last_visual_error = (
                        "visual description too short"
                    )

                    visual_description = None
                    continue

                break

            except Exception as e:

                last_visual_error = str(e)

        if visual_description is None:

            raise RuntimeError(
                f"Visual generation failed for "
                f"scene {index}: "
                f"{last_visual_error}"
            )

        scenes.append(
            build_scene(
                scene_number=index,
                narration=segment,
                visual_description=visual_description
            )
        )

        print(
            f"[SCENES] {job_id} | "
            f"SCENE {index}/{len(segments)} | "
            f"visual description ready"
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if len(scenes) < MIN_SCENES:
        raise RuntimeError(
            f"Too few scenes generated: {len(scenes)}"
        )

    if len(scenes) > MAX_SCENES:
        raise RuntimeError(
            f"Too many scenes generated: {len(scenes)}"
        )

    for scene in scenes:

        words = count_words(scene["narration"])

        if words < MIN_SCENE_WORDS:
            raise RuntimeError(
                f"Scene {scene['scene_number']} "
                f"has only {words} words"
            )

        if words > MAX_SCENE_WORDS:
            raise RuntimeError(
                f"Scene {scene['scene_number']} "
                f"has {words} words"
            )

    # ========================================================
    # SAVE
    # ========================================================

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
