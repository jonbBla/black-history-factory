import re
from pathlib import Path


# ============================================================
# NARRATION → SCENE SEGMENTATION
# ============================================================

MIN_SCENE_WORDS = 4
MAX_SCENE_WORDS = 14


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text))


def normalize_text(text: str) -> str:
    """Clean whitespace without changing the actual wording."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize_words(text: str):
    """
    Return words while preserving punctuation attached to them.

    Example:
        'The kingdom rose, powerful and wealthy.'
    becomes:
        ['The', 'kingdom', 'rose,', 'powerful', 'and', 'wealthy.']
    """
    return re.findall(r"\S+", normalize_text(text))


def punctuation_score(token: str) -> int:
    """
    Higher score = better/natural place to end a scene.
    """

    token = token.rstrip('"”’)]}')

    if token.endswith((".", "!", "?")):
        return 100

    if token.endswith(";"):
        return 85

    if token.endswith(":"):
        return 70

    if token.endswith(","):
        return 45

    return 0


def connector_score(token: str) -> int:
    """
    Natural linguistic transition points.
    """

    word = re.sub(r"[^\w’'-]", "", token).lower()

    connectors = {
        "and",
        "but",
        "while",
        "because",
        "although",
        "where",
        "when",
        "as",
        "yet",
        "so",
        "then",
        "which",
        "who",
        "whose",
    }

    return 20 if word in connectors else 0


def boundary_score(tokens, position):
    """
    Score a possible boundary after tokens[position - 1].
    """

    if position <= 0 or position > len(tokens):
        return -9999

    token = tokens[position - 1]

    score = punctuation_score(token)

    # Avoid ending immediately before a connector.
    if position < len(tokens):
        next_word = re.sub(
            r"[^\w’'-]",
            "",
            tokens[position]
        ).lower()

        if next_word in {
            "and",
            "but",
            "because",
            "while",
            "although",
            "when",
            "where",
            "which",
            "who",
        }:
            score -= 15

    # Connector after the boundary is usually less natural.
    score += connector_score(token)

    return score


def choose_scene_count(total_words: int) -> int:
    """
    Choose between 18–22 scenes.

    Approximately 8–12 words per scene is preferred,
    while never exceeding 14 words.
    """

    if total_words <= 150:
        return 18

    if total_words <= 175:
        return 19

    if total_words <= 195:
        return 20

    if total_words <= 215:
        return 21

    return 22


def segment_narration(narration: str):
    """
    Deterministically divide narration into natural scenes.

    Rules:
      - 18–22 scenes
      - minimum 4 words
      - maximum 14 words
      - strongly prefers sentence endings
      - then semicolons/colons
      - then commas
      - then natural connectors
      - arbitrary word splitting only as a last resort

    IMPORTANT:
      The narration wording itself is NOT rewritten.
    """

    narration = normalize_text(narration)
    tokens = tokenize_words(narration)

    total = len(tokens)

    if total < MIN_SCENE_WORDS:
        raise ValueError(
            f"Narration too short: {total} words."
        )

    target_scenes = choose_scene_count(total)

    # Make sure the requested number of scenes is mathematically possible.
    minimum_possible = (total + MAX_SCENE_WORDS - 1) // MAX_SCENE_WORDS
    maximum_possible = total // MIN_SCENE_WORDS

    target_scenes = max(
        target_scenes,
        minimum_possible
    )

    target_scenes = min(
        target_scenes,
        maximum_possible
    )

    target_words = total / target_scenes

    print(
        f"[SCENES] NARRATION: {total} words | "
        f"TARGET: {target_scenes} scenes | "
        f"AVG: {target_words:.1f} words"
    )

    # --------------------------------------------------------
    # Dynamic programming.
    #
    # dp[i][k] = best cost for splitting first i words
    # into k scenes.
    # --------------------------------------------------------

    INF = float("inf")

    dp = [
        [INF] * (target_scenes + 1)
        for _ in range(total + 1)
    ]

    previous = [
        [None] * (target_scenes + 1)
        for _ in range(total + 1)
    ]

    dp[0][0] = 0

    for i in range(1, total + 1):

        for k in range(1, target_scenes + 1):

            # Last scene length.
            for start in range(
                max(0, i - MAX_SCENE_WORDS),
                i - MIN_SCENE_WORDS + 1
            ):

                length = i - start

                if length < MIN_SCENE_WORDS:
                    continue

                if length > MAX_SCENE_WORDS:
                    continue

                if dp[start][k - 1] == INF:
                    continue

                # ------------------------------------------------
                # Length cost.
                #
                # Prefer roughly 7–12 words.
                # ------------------------------------------------

                if length < 7:
                    length_cost = (7 - length) ** 2 * 2
                elif length > 12:
                    length_cost = (length - 12) ** 2 * 2
                else:
                    length_cost = 0

                # ------------------------------------------------
                # Boundary quality.
                # ------------------------------------------------

                boundary = boundary_score(tokens, i)

                # Strongly reward punctuation.
                boundary_cost = -boundary * 0.8

                # ------------------------------------------------
                # Penalize awkward tiny fragments.
                # ------------------------------------------------

                fragment_cost = 0

                if length <= 5:
                    fragment_cost += 8

                # ------------------------------------------------
                # Slightly discourage splitting at arbitrary words.
                # ------------------------------------------------

                if boundary == 0:
                    boundary_cost += 12

                cost = (
                    dp[start][k - 1]
                    + length_cost
                    + boundary_cost
                    + fragment_cost
                )

                if cost < dp[i][k]:
                    dp[i][k] = cost
                    previous[i][k] = start

    # --------------------------------------------------------
    # Reconstruct scenes.
    # --------------------------------------------------------

    if previous[total][target_scenes] is None:
        raise ValueError(
            "Could not find a valid narration segmentation "
            f"for {total} words into {target_scenes} scenes."
        )

    boundaries = []

    i = total
    k = target_scenes

    while k > 0:

        start = previous[i][k]

        if start is None:
            raise ValueError(
                "Scene reconstruction failed."
            )

        boundaries.append((start, i))

        i = start
        k -= 1

    boundaries.reverse()

    scenes = []

    for index, (start, end) in enumerate(boundaries, 1):

        text = " ".join(tokens[start:end]).strip()
        words = word_count(text)

        if words < MIN_SCENE_WORDS:
            raise ValueError(
                f"Scene {index} too short: {words} words"
            )

        if words > MAX_SCENE_WORDS:
            raise ValueError(
                f"Scene {index} too long: {words} words"
            )

        scenes.append({
            "scene_number": index,
            "narration": text,
            "word_count": words,
        })

    print(
        f"[SCENES] Created {len(scenes)} scenes successfully."
    )

    print(
        "[SCENES] Lengths:",
        [s["word_count"] for s in scenes]
    )

    return scenes


# ============================================================
# VISUAL DESCRIPTION GENERATION
# ============================================================

VISUAL_PROMPT = """
You are creating a visual description for a historical documentary scene.

The narration for this scene is:

{narration}

Historical visual bible:

{visual_bible}

Describe exactly what should appear on screen.

Requirements:
- One cinematic visual description.
- Do NOT write narration.
- Do NOT explain your reasoning.
- Do NOT return JSON.
- Do NOT use bullet points.
- Focus on visible subjects, actions, environment, architecture,
  clothing, objects, lighting, camera composition and atmosphere.
- Keep the visual historically appropriate to the topic.
- Make the scene visually specific rather than generic.
- Maintain continuity with the historical visual bible.
- Use cinematic 3D historical reconstruction.
- Physically plausible materials.
- Realistic proportions.
- Detailed environments and textures.
- Dramatic natural lighting.
- Volumetric atmosphere.
- Strong depth and cinematic composition.
- High-end game cinematic / Unreal Engine style.
- Not flat cartoon.

Return ONLY the visual description.
"""


def generate_visual_description(
    qwen,
    narration,
    visual_bible,
    max_new_tokens=450
):
    """
    Qwen handles ONLY visual description generation.

    It does NOT control:
      - scene count
      - scene boundaries
      - narration
      - JSON structure
    """

    prompt = VISUAL_PROMPT.format(
        narration=narration,
        visual_bible=str(visual_bible)
    )

    result = qwen.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.35,
    )

    result = normalize_text(result)

    # Remove accidental formatting.
    result = re.sub(
        r"^(visual description|description)\s*:\s*",
        "",
        result,
        flags=re.IGNORECASE
    )

    return result.strip()


# ============================================================
# MAIN RUNNER
# ============================================================

def run(paths, job_id, narration, visual_bible, config, qwen):

    print(
        f"[QWEN] SCENES {job_id} | "
        "Punctuation-aware deterministic segmentation"
    )

    # --------------------------------------------------------
    # 1. Segment narration using Python.
    # --------------------------------------------------------

    scene_parts = segment_narration(narration)

    # --------------------------------------------------------
    # 2. Generate visual description for each scene.
    # --------------------------------------------------------

    final_scenes = []

    for scene in scene_parts:

        number = scene["scene_number"]
        scene_narration = scene["narration"]

        print(
            f"[QWEN] SCENE {number}/{len(scene_parts)} | "
            f"{scene['word_count']} words | "
            "generating visual description"
        )

        visual_description = generate_visual_description(
            qwen=qwen,
            narration=scene_narration,
            visual_bible=visual_bible,
            max_new_tokens=450,
        )

        if not visual_description:
            raise ValueError(
                f"Scene {number} returned an empty "
                "visual description."
            )

        final_scenes.append({
            "scene_number": number,
            "narration": scene_narration,
            "word_count": scene["word_count"],
            "visual_description": visual_description,
        })

    # --------------------------------------------------------
    # 3. Final validation.
    # --------------------------------------------------------

    if not (18 <= len(final_scenes) <= 22):
        raise ValueError(
            f"Invalid scene count: {len(final_scenes)}"
        )

    for scene in final_scenes:

        words = word_count(scene["narration"])

        if words < MIN_SCENE_WORDS:
            raise ValueError(
                f"Scene {scene['scene_number']} has "
                f"only {words} words."
            )

        if words > MAX_SCENE_WORDS:
            raise ValueError(
                f"Scene {scene['scene_number']} has "
                f"{words} words."
            )

    print(
        f"[QWEN] SCENES {job_id} | "
        f"SUCCESS | {len(final_scenes)} scenes"
    )

    return {
        "scene_count": len(final_scenes),
        "scenes": final_scenes,
    }
