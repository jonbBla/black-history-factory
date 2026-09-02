import re

from factory.utils import write_json_atomic


# ============================================================
# CONFIGURATION
# ============================================================

MIN_SCENES = 18
TARGET_SCENES = 20
MAX_SCENES = 22

MIN_SCENE_WORDS = 4
MAX_SCENE_WORDS = 14

VISUAL_ATTEMPTS = 3


# ============================================================
# WORD / TEXT UTILITIES
# ============================================================

def words(text):
    return re.findall(r"\b[\w’'-]+\b", str(text))


def count_words(text):
    return len(words(text))


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):
    """
    Split narration into sentences while preserving the original
    wording as much as possible.
    """

    text = re.sub(r"\s+", " ", str(text)).strip()

    # Split after normal sentence punctuation.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        s.strip()
        for s in sentences
        if s.strip()
    ]


# ============================================================
# DETERMINISTIC SEGMENTATION
# ============================================================

def segment_narration(narration):
    """
    Deterministically divide the narration into approximately
    18–22 short visual beats.

    Qwen is NOT involved here.
    """

    sentences = split_sentences(narration)

    if not sentences:
        raise ValueError("Narration is empty.")

    total_words = count_words(narration)

    # Choose scene count based on narration length.
    if total_words <= 174:
        target = 18
    elif total_words <= 184:
        target = 19
    elif total_words <= 202:
        target = 20
    elif total_words <= 212:
        target = 21
    else:
        target = 22

    target = max(MIN_SCENES, min(MAX_SCENES, target))

    print(
        f"[SCENES] NARRATION: {total_words} words | "
        f"TARGET: {target} scenes"
    )

    # --------------------------------------------------------
    # First attempt: split naturally by sentences.
    # --------------------------------------------------------

    chunks = []

    for sentence in sentences:

        sentence_words = count_words(sentence)

        if sentence_words <= MAX_SCENE_WORDS:
            chunks.append(sentence)
            continue

        # Long sentence: split into word groups at punctuation /
        # conjunction boundaries where possible.
        chunks.extend(
            split_long_sentence(sentence)
        )

    # --------------------------------------------------------
    # If we have too few chunks, split the longest chunks.
    # --------------------------------------------------------

    while len(chunks) < target:

        index = find_longest_splittable_chunk(chunks)

        if index is None:
            break

        original = chunks.pop(index)

        pieces = split_chunk(original)

        if len(pieces) <= 1:
            chunks.insert(index, original)
            break

        for piece in reversed(pieces):
            chunks.insert(index, piece)

    # --------------------------------------------------------
    # If we have too many chunks, merge neighboring chunks.
    # --------------------------------------------------------

    while len(chunks) > target:

        index = find_best_merge(chunks)

        if index is None:
            break

        merged = (
            chunks[index].rstrip()
            + " "
            + chunks[index + 1].lstrip()
        )

        chunks[index:index + 2] = [merged]

    # --------------------------------------------------------
    # Final cleanup.
    # --------------------------------------------------------

    chunks = [c.strip() for c in chunks if c.strip()]

    # If a chunk is still too long, split it.
    final_chunks = []

    for chunk in chunks:

        if count_words(chunk) <= MAX_SCENE_WORDS:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(
                split_chunk(chunk)
            )

    chunks = final_chunks

    # Final merge if splitting pushed us above target.
    while len(chunks) > MAX_SCENES:

        index = find_best_merge(chunks)

        if index is None:
            break

        merged = (
            chunks[index].rstrip()
            + " "
            + chunks[index + 1].lstrip()
        )

        chunks[index:index + 2] = [merged]

    # --------------------------------------------------------
    # Validate.
    # --------------------------------------------------------

    valid, error = validate_segments(chunks)

    if not valid:
        raise ValueError(
            f"Deterministic segmentation failed: {error}"
        )

    return chunks


def split_long_sentence(sentence):
    """
    Split a long sentence using natural punctuation/conjunctions.
    """

    # First try commas, semicolons and conjunctions.
    pieces = re.split(
        r"(?<=[,;:])\s+|"
        r"\s+(?=(?:and|but|while|because|after|before|"
        r"when|where|which|who|as|then)\s+)",
        sentence,
        flags=re.IGNORECASE
    )

    pieces = [
        p.strip(" ,;:")
        for p in pieces
        if p.strip(" ,;:")
    ]

    # If natural splitting worked, recursively split anything
    # still too long.
    result = []

    for piece in pieces:

        if count_words(piece) <= MAX_SCENE_WORDS:
            result.append(piece)
        else:
            result.extend(
                split_chunk(piece)
            )

    return result


def split_chunk(text):
    """
    Hard fallback: split a chunk into <=14-word pieces.

    Attempts to keep pieces around 8–10 words.
    """

    tokens = text.split()

    if len(tokens) <= MAX_SCENE_WORDS:
        return [text.strip()]

    pieces = []

    # Aim for approximately 8–10 words per scene.
    target_size = 9

    current = []

    for token in tokens:

        current.append(token)

        if len(current) >= target_size:

            pieces.append(
                " ".join(current).strip()
            )

            current = []

    if current:
        pieces.append(
            " ".join(current).strip()
        )

    # If the final piece is too short, merge it backward.
    if len(pieces) >= 2:

        if count_words(pieces[-1]) < MIN_SCENE_WORDS:

            pieces[-2] = (
                pieces[-2]
                + " "
                + pieces[-1]
            )

            pieces.pop()

    return pieces


def find_longest_splittable_chunk(chunks):

    candidates = []

    for i, chunk in enumerate(chunks):

        wc = count_words(chunk)

        if wc >= 2 * MIN_SCENE_WORDS:
            candidates.append((wc, i))

    if not candidates:
        return None

    candidates.sort(reverse=True)

    return candidates[0][1]


def find_best_merge(chunks):
    """
    Find neighboring chunks whose combined length is still
    within the maximum scene size.
    """

    candidates = []

    for i in range(len(chunks) - 1):

        combined_words = (
            count_words(chunks[i])
            + count_words(chunks[i + 1])
        )

        if combined_words <= MAX_SCENE_WORDS:

            # Prefer balanced/short combinations.
            candidates.append(
                (
                    abs(combined_words - 10),
                    i
                )
            )

    if not candidates:
        return None

    candidates.sort()

    return candidates[0][1]


# ============================================================
# VALIDATION
# ============================================================

def validate_segments(segments):

    if not segments:
        return False, "no segments"

    if len(segments) < MIN_SCENES:
        return False, (
            f"only {len(segments)} scenes; "
            f"minimum is {MIN_SCENES}"
        )

    if len(segments) > MAX_SCENES:
        return False, (
            f"{len(segments)} scenes; "
            f"maximum is {MAX_SCENES}"
        )

    seen = set()

    for i, segment in enumerate(segments, start=1):

        wc = count_words(segment)

        if wc < MIN_SCENE_WORDS:
            return False, (
                f"scene {i} too short: {wc} words"
            )

        if wc > MAX_SCENE_WORDS:
            return False, (
                f"scene {i} too long: {wc} words"
            )

        key = normalize(segment)

        if key in seen:
            return False, (
                f"duplicate narration in scene {i}"
            )

        seen.add(key)

    return True, "valid"


# ============================================================
# VISUAL DESCRIPTION PROMPT
# ============================================================

def visual_prompt(
    scene_number,
    scene_narration,
    visual_bible
):

    return f"""
You are creating ONE visual description for a historical
documentary scene.

SCENE NUMBER:
{scene_number}

NARRATION:
{scene_narration}

VISUAL BIBLE:
{visual_bible}

Describe exactly what should be visible on screen for this
narration.

The image should directly communicate the narration.

Include useful visual details such as:

- people
- clothing
- architecture
- environment
- objects
- actions
- historical setting
- lighting
- atmosphere
- camera composition
- scale and depth

Maintain historical plausibility.

Use the established visual bible consistently.

STYLE:
Cinematic 3D historical reconstruction,
high-end game cinematic,
Unreal Engine style,
physically plausible materials,
period-authentic details,
dramatic natural lighting,
volumetric atmosphere,
strong depth,
detailed environments,
realistic textures,
realistic proportions,
cinematic composition.

Do not:

- rewrite the narration
- add narration
- create dialogue
- mention the narrator
- create a source card
- mention modern objects unless historically appropriate
- use bullet points
- use JSON

Return ONLY the visual description.
"""


# ============================================================
# VISUAL DESCRIPTION GENERATION
# ============================================================

def generate_visual(
    qwen,
    job_id,
    scene_number,
    narration,
    visual_bible
):

    for attempt in range(1, VISUAL_ATTEMPTS + 1):

        print(
            f"[SCENES] {job_id} | "
            f"SCENE {scene_number} | "
            f"VISUAL ATTEMPT {attempt}/{VISUAL_ATTEMPTS}"
        )

        try:

            raw = qwen.generate(
                visual_prompt(
                    scene_number,
                    narration,
                    visual_bible
                ),
                max_new_tokens=450,
                temperature=0.35
            )

            visual = str(raw).strip()

            # Remove markdown fences.
            visual = re.sub(
                r"```(?:text)?",
                "",
                visual,
                flags=re.IGNORECASE
            )

            visual = visual.replace("```", "").strip()

            # Remove accidental label.
            visual = re.sub(
                r"^(?:visual description|description)\s*:\s*",
                "",
                visual,
                flags=re.IGNORECASE
            )

            if count_words(visual) < 8:

                print(
                    f"[SCENES] {job_id} | "
                    f"SCENE {scene_number} | "
                    f"visual too short"
                )

                continue

            return visual

        except Exception as e:

            print(
                f"[SCENES] {job_id} | "
                f"SCENE {scene_number} | "
                f"visual error: {e}"
            )

    raise RuntimeError(
        f"Could not generate visual description "
        f"for scene {scene_number}"
    )


# ============================================================
# MAIN RUNNER
# ============================================================

def run(
    paths,
    job_id,
    narration,
    visual_bible,
    config,
    qwen
):

    print(
        f"[SCENES] {job_id} | "
        f"STAGE: SCENE_PLANNING"
    )

    # ========================================================
    # STEP 1
    # Python segments narration.
    # ========================================================

    print(
        f"[SCENES] {job_id} | "
        f"STEP 1/2 | deterministic narration segmentation"
    )

    segments = segment_narration(narration)

    print(
        f"[SCENES] {job_id} | "
        f"SEGMENTATION COMPLETE | "
        f"{len(segments)} scenes"
    )

    for i, segment in enumerate(segments, start=1):

        print(
            f"[SCENES] {job_id} | "
            f"SCENE {i:02d} | "
            f"{count_words(segment)} words | "
            f"{segment}"
        )

    # ========================================================
    # STEP 2
    # Qwen creates ONLY visual descriptions.
    # ========================================================

    print(
        f"[SCENES] {job_id} | "
        f"STEP 2/2 | generating visual descriptions"
    )

    scenes = []

    for scene_number, segment in enumerate(
        segments,
        start=1
    ):

        visual_description = generate_visual(
            qwen=qwen,
            job_id=job_id,
            scene_number=scene_number,
            narration=segment,
            visual_bible=visual_bible
        )

        scenes.append(
            {
                "scene_number": scene_number,
                "narration": segment,
                "visual_description": visual_description
            }
        )

        print(
            f"[SCENES] {job_id} | "
            f"SCENE {scene_number}/{len(segments)} | "
            f"READY"
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    final_narration_words = sum(
        count_words(scene["narration"])
        for scene in scenes
    )

    original_narration_words = count_words(narration)

    print(
        f"[SCENES] {job_id} | "
        f"ORIGINAL WORDS: {original_narration_words}"
    )

    print(
        f"[SCENES] {job_id} | "
        f"SCENE WORDS: {final_narration_words}"
    )

    if len(scenes) < MIN_SCENES:
        raise RuntimeError(
            f"Too few scenes: {len(scenes)}"
        )

    if len(scenes) > MAX_SCENES:
        raise RuntimeError(
            f"Too many scenes: {len(scenes)}"
        )

    for scene in scenes:

        wc = count_words(scene["narration"])

        if wc < MIN_SCENE_WORDS:
            raise RuntimeError(
                f"Scene {scene['scene_number']} "
                f"is too short: {wc} words"
            )

        if wc > MAX_SCENE_WORDS:
            raise RuntimeError(
                f"Scene {scene['scene_number']} "
                f"is too long: {wc} words"
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
