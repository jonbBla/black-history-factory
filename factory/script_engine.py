import re
from .utils import write_text_atomic, write_json_atomic


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

TARGET_WORDS = 195

MIN_ACCEPTABLE_WORDS = 170
MAX_ACCEPTABLE_WORDS = 220

MAX_ATTEMPTS = 3

MAX_NEW_TOKENS = 700


# ---------------------------------------------------------
# WORD COUNT
# ---------------------------------------------------------

def count_words(text):
    """
    Consistent word counting used by both generation
    and validation.
    """

    if not text:
        return 0

    # Remove common formatting.
    text = re.sub(r"```(?:text)?", "", text)
    text = text.replace("```", "")

    # Count actual word-like tokens.
    words = re.findall(
        r"\b[\w’'-]+\b",
        text,
        flags=re.UNICODE
    )

    return len(words)


# ---------------------------------------------------------
# CLEAN OUTPUT
# ---------------------------------------------------------

def clean_narration(text):

    if not text:
        return ""

    text = text.strip()

    # Remove markdown/code fences.
    text = re.sub(
        r"^```(?:text)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    # Remove accidental labels.
    text = re.sub(
        r"^(Narration|Script|Voiceover)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


# ---------------------------------------------------------
# PROMPT
# ---------------------------------------------------------

def build_prompt(topic, research, target_words):

    research_text = str(research)

    # Keep the prompt reasonably small for the 1.5B model.
    research_text = research_text[:12000]

    return f"""
Write a fast-paced vertical documentary narration about:

{topic.title}

TARGET LENGTH:
Approximately {target_words} words.

IMPORTANT:
- Aim for {target_words} words.
- Acceptable final range is {MIN_ACCEPTABLE_WORDS}-{MAX_ACCEPTABLE_WORDS} words.
- Do NOT intentionally write far shorter or longer.
- Do not include a title.
- Do not include scene numbers.
- Do not include stage directions.
- Output ONLY the narration.

STRUCTURE:

1. Start with a powerful hook.
   Use a surprising fact, question, mystery, contradiction,
   or something that makes the viewer immediately curious.

2. Quickly establish what happened and where.

3. Explain the important historical, technological,
   architectural, artistic, cultural, or mythological details.

4. Include lesser-known information when supported by the research.

5. Clearly distinguish established facts from:
   - archaeological evidence
   - scholarly interpretation
   - oral tradition
   - mythology
   - uncertainty

6. Do not invent facts, people, dates, buildings,
   technologies, artifacts, quotations, or sources.

7. Do not make colonization the central theme.

8. Keep the pacing energetic and suitable for a
   roughly 90-second short video.

9. End with a memorable conclusion.

RESEARCH:
{research_text}
"""


# ---------------------------------------------------------
# CORRECTION PROMPT
# ---------------------------------------------------------

def build_correction_prompt(text, word_count):

    if word_count < MIN_ACCEPTABLE_WORDS:

        needed = TARGET_WORDS - word_count

        instruction = f"""
The narration is too short.

Current length: {word_count} words.
Target: approximately {TARGET_WORDS} words.

Expand it naturally by roughly {max(20, needed)} words.

Add useful historical information, context,
a lesser-known verified detail, or significance.

Do NOT repeat existing sentences.
Do NOT add unsupported claims.

Return ONLY the complete revised narration.
"""

    elif word_count > MAX_ACCEPTABLE_WORDS:

        excess = word_count - TARGET_WORDS

        instruction = f"""
The narration is too long.

Current length: {word_count} words.
Target: approximately {TARGET_WORDS} words.

Shorten it naturally by roughly {max(20, excess)} words.

Remove repetition, unnecessary adjectives,
and redundant explanations.

Preserve the hook, important facts,
interesting details, and conclusion.

Do NOT remove important historical qualifications.

Return ONLY the complete revised narration.
"""

    else:

        instruction = """
The narration is already within the acceptable range.

Improve flow and pacing slightly if necessary,
but do not significantly change its length.

Return ONLY the complete narration.
"""

    return f"""
{instruction}

CURRENT NARRATION:

{text}
"""


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def run(paths, job_id, topic, verified, config, qwen):

    research = verified.get(
        "research",
        verified
    )

    print(
        f"[SCRIPT] {job_id} | "
        f"TARGET: ~{TARGET_WORDS} words | "
        f"acceptable: {MIN_ACCEPTABLE_WORDS}-{MAX_ACCEPTABLE_WORDS}"
    )

    # -----------------------------------------------------
    # ATTEMPT 1 — DIRECT GENERATION
    # -----------------------------------------------------

    print(
        f"[SCRIPT] {job_id} | "
        f"ATTEMPT 1/{MAX_ATTEMPTS} | "
        f"Generating narration"
    )

    prompt = build_prompt(
        topic,
        research,
        TARGET_WORDS
    )

    text = qwen.generate(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7
    )

    text = clean_narration(text)

    words = count_words(text)

    print(
        f"[SCRIPT] {job_id} | "
        f"Attempt 1: {words} words"
    )

    # -----------------------------------------------------
    # CHECK
    # -----------------------------------------------------

    if MIN_ACCEPTABLE_WORDS <= words <= MAX_ACCEPTABLE_WORDS:

        final_text = text

        print(
            f"[SCRIPT] {job_id} | "
            f"ACCEPTED | {words} words"
        )

    else:

        final_text = None

        # -------------------------------------------------
        # TARGETED RETRIES
        # -------------------------------------------------

        for attempt in range(2, MAX_ATTEMPTS + 1):

            print(
                f"[SCRIPT] {job_id} | "
                f"ATTEMPT {attempt}/{MAX_ATTEMPTS} | "
                f"Correcting {words}-word narration"
            )

            correction_prompt = build_correction_prompt(
                text,
                words
            )

            text = qwen.generate(
                correction_prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.55
            )

            text = clean_narration(text)

            words = count_words(text)

            print(
                f"[SCRIPT] {job_id} | "
                f"Attempt {attempt}: {words} words"
            )

            if MIN_ACCEPTABLE_WORDS <= words <= MAX_ACCEPTABLE_WORDS:

                final_text = text

                print(
                    f"[SCRIPT] {job_id} | "
                    f"ACCEPTED | {words} words"
                )

                break

        # -------------------------------------------------
        # FALLBACK
        # -------------------------------------------------

        if final_text is None:

            # Do not destroy a reasonably sized narration
            # merely because it missed the preferred range.
            if 150 <= words <= 240:

                final_text = text

                print(
                    f"[SCRIPT] {job_id} | "
                    f"WARNING | accepting {words} words "
                    f"after {MAX_ATTEMPTS} attempts"
                )

            else:

                raise ValueError(
                    f"Narration failed validation after "
                    f"{MAX_ATTEMPTS} attempts. "
                    f"Final count: {words} words."
                )

    # -----------------------------------------------------
    # FINAL VALIDATION
    # -----------------------------------------------------

    final_text = clean_narration(final_text)

    final_count = count_words(final_text)

    if not final_text:
        raise ValueError(
            "Qwen returned an empty narration."
        )

    if final_count < 150:
        raise ValueError(
            f"Narration is unusably short: "
            f"{final_count} words."
        )

    if final_count > 240:
        raise ValueError(
            f"Narration is excessively long: "
            f"{final_count} words."
        )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    write_text_atomic(
        paths.narration(job_id),
        final_text
    )

    write_json_atomic(
        paths.script(job_id),
        {
            "word_count": final_count,
            "target_words": TARGET_WORDS,
            "preferred_min_words": MIN_ACCEPTABLE_WORDS,
            "preferred_max_words": MAX_ACCEPTABLE_WORDS,
            "target_seconds": config.target_video_seconds,
            "format": "fast-paced vertical documentary",
            "hook": (
                final_text.splitlines()[0]
                if final_text.splitlines()
                else ""
            )
        }
    )

    print(
        f"[SCRIPT] {job_id} | "
        f"COMPLETE | {final_count} words"
    )

    return final_text
