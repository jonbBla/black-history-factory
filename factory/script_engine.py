import re


MIN_NARRATION_WORDS = 170
MAX_NARRATION_WORDS = 220
TARGET_NARRATION_WORDS = 195
MAX_NARRATION_ATTEMPTS = 6


def count_words(text):
    return len(re.findall(r"\b[\w’'-]+\b", text))


def clean_narration(text):
    """
    Clean formatting without rewriting the narration.
    """
    text = text.strip()

    # Remove accidental markdown/code fences.
    text = re.sub(r"^```(?:text)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Remove common labels.
    text = re.sub(
        r"^(narration|script)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_valid_length(text):
    words = count_words(text)

    return (
        MIN_NARRATION_WORDS
        <= words
        <= MAX_NARRATION_WORDS
    )


def build_narration_prompt(
    topic,
    research,
    fact_check,
    visual_bible,
    target_words=TARGET_NARRATION_WORDS,
):
    return f"""
Write the final narration for a fast-paced historical documentary.

TOPIC:
{topic}

RESEARCH / EVIDENCE:
{research}

FACT-CHECK / VALIDATION:
{fact_check}

VISUAL BIBLE:
{visual_bible}

NARRATION LENGTH:

Target: approximately {target_words} words.

HARD REQUIREMENT:
- Minimum: {MIN_NARRATION_WORDS} words
- Maximum: {MAX_NARRATION_WORDS} words

Aim for approximately 190–200 words.

CONTENT REQUIREMENTS:
- Tell a compelling historical story.
- Use only information supported by the supplied research
  and fact-check material.
- Do not invent facts.
- Do not add unsupported dates, names, places, quotations,
  traditions or claims.
- Make the narration understandable to a general audience.
- Keep the pacing suitable for a roughly 90-second documentary.
- Use natural punctuation.
- Use short-to-medium sentences where appropriate.
- Avoid unnecessary introductions and conclusions.
- Do not include scene numbers.
- Do not include image descriptions.
- Do not include production instructions.
- Do not include headings.
- Return ONLY the narration.

IMPORTANT:
Stay within the {MIN_NARRATION_WORDS}–{MAX_NARRATION_WORDS}
word range. Do NOT exceed {MAX_NARRATION_WORDS} words.
"""


def generate_narration(
    qwen,
    topic,
    research,
    fact_check,
    visual_bible,
):
    """
    Generate narration with up to 6 attempts.

    Qwen is responsible for writing.
    Python is responsible for enforcing the hard word limit.
    """

    best_script = None
    best_distance = float("inf")

    for attempt in range(
        1,
        MAX_NARRATION_ATTEMPTS + 1
    ):

        print(
            f"[SCRIPT] ATTEMPT "
            f"{attempt}/{MAX_NARRATION_ATTEMPTS} | "
            f"Generating narration"
        )

        # ----------------------------------------------------
        # First attempt gets the normal prompt.
        # Later attempts explicitly tell Qwen what was wrong.
        # ----------------------------------------------------

        prompt = build_narration_prompt(
            topic=topic,
            research=research,
            fact_check=fact_check,
            visual_bible=visual_bible,
        )

        if best_script is not None:

            previous_words = count_words(best_script)

            if previous_words < MIN_NARRATION_WORDS:

                prompt += f"""

CORRECTION REQUIRED:

Your previous narration was only {previous_words} words.

Rewrite it as a COMPLETE narration.

It must contain at least {MIN_NARRATION_WORDS} words
and no more than {MAX_NARRATION_WORDS} words.

Do not simply add filler.
Expand the useful historical information and context.
"""

            elif previous_words > MAX_NARRATION_WORDS:

                prompt += f"""

CORRECTION REQUIRED:

Your previous narration was {previous_words} words.

Rewrite it more concisely.

You MUST reduce it to no more than
{MAX_NARRATION_WORDS} words.

Do not remove important historical facts.
Remove repetition, unnecessary descriptions,
and filler instead.
"""

            else:

                prompt += f"""

The previous attempt was {previous_words} words.

Produce another polished version close to
{TARGET_NARRATION_WORDS} words while remaining
between {MIN_NARRATION_WORDS} and {MAX_NARRATION_WORDS}.
"""

        result = qwen.generate(
            prompt,
            max_new_tokens=700,
            temperature=0.35,
        )

        script = clean_narration(result)

        words = count_words(script)

        print(
            f"[SCRIPT] Attempt {attempt}: "
            f"{words} words"
        )

        # ----------------------------------------------------
        # Keep the closest attempt as a diagnostic fallback.
        # We NEVER automatically accept an invalid script.
        # ----------------------------------------------------

        distance = abs(
            words - TARGET_NARRATION_WORDS
        )

        if distance < best_distance:

            best_script = script
            best_distance = distance

        # ----------------------------------------------------
        # Valid narration.
        # ----------------------------------------------------

        if is_valid_length(script):

            print(
                f"[SCRIPT] ACCEPTED | "
                f"{words} words"
            )

            print(
                f"[SCRIPT] COMPLETE | "
                f"{words} words"
            )

            return script

    # --------------------------------------------------------
    # All attempts failed.
    #
    # IMPORTANT:
    # Do NOT accept the closest script.
    # Let the job fail and remain resumable.
    # --------------------------------------------------------

    best_words = (
        count_words(best_script)
        if best_script
        else 0
    )

    raise ValueError(
        f"Narration generation failed after "
        f"{MAX_NARRATION_ATTEMPTS} attempts. "
        f"Best attempt: {best_words} words. "
        f"Required: {MIN_NARRATION_WORDS}-"
        f"{MAX_NARRATION_WORDS} words."
    )


def run(
    paths,
    job_id,
    topic,
    research,
    fact_check,
    visual_bible,
    config,
    qwen,
):
    print(
        f"[QWEN] {job_id} | "
        "STAGE: NARRATION | writing and validating narration"
    )

    print(
        f"[SCRIPT] {job_id} | "
        f"TARGET: ~{TARGET_NARRATION_WORDS} words | "
        f"acceptable: {MIN_NARRATION_WORDS}-"
        f"{MAX_NARRATION_WORDS}"
    )

    narration = generate_narration(
        qwen=qwen,
        topic=topic,
        research=research,
        fact_check=fact_check,
        visual_bible=visual_bible,
    )

    return narration
