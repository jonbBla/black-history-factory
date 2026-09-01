import re
from .utils import write_text_atomic, write_json_atomic


IDEAL_MIN = 185
IDEAL_MAX = 195

ACCEPT_MIN = 175
ACCEPT_MAX = 205

GENERATION_TOKENS = 320
REVISION_TOKENS = 300


def word_count(text):
    return len(re.findall(r"\b[\w’'-]+\b", text))


def clean_narration(text):
    if not isinstance(text, str):
        return ""

    text = text.strip()

    # Remove accidental markdown/code fences.
    text = re.sub(r"^```(?:text|txt)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    # Remove common model prefixes.
    text = re.sub(
        r"^(narration|documentary narration|script)\s*:\s*",
        "",
        text,
        flags=re.I,
    )

    return text.strip()


def generate_initial(topic, research, config, qwen):
    prompt = f"""
Write a fast-paced documentary narration about:

{topic.title}

Create approximately 190 words.

WORD REQUIREMENTS:
- Ideal: 185–195 words.
- Acceptable: 175–205 words.
- ABSOLUTE MAXIMUM: 205 words.
- Do not exceed 205 words.
- Do not deliberately add words just to reach the target.

STYLE:
- Short, clear sentences.
- Fast-paced vertical documentary/TikTok style.
- Strong curiosity-driven opening.
- Every sentence must move the story forward.
- No repetition.
- Do not repeat the opening later.
- Do not repeat the same fact using different words.
- End with a strong conclusion.
- Do not include a title.
- Do not include scene directions.
- Do not include labels such as "Hook:" or "Conclusion:".
- Return ONLY the narration.

STORY FLOW:
1. Hook / surprising question
2. Context
3. Main discovery or achievement
4. Lesser-known or unexpected detail
5. Why it matters
6. Strong conclusion

FACTUAL RULES:
- Use only information supported by the research.
- Never invent people, dates, places, technologies, buildings,
  archaeological discoveries, quotations, or sources.
- Clearly distinguish mythology, oral tradition and scholarly
  interpretation from established facts.
- Do not make colonization the central subject.

RESEARCH:
{research}
"""

    return clean_narration(
        qwen.generate(
            prompt,
            max_new_tokens=GENERATION_TOKENS,
            temperature=0.65,
        )
    )


def revise_length(text, target_instruction, qwen):
    prompt = f"""
Revise the narration below.

{target_instruction}

IMPORTANT:
- Preserve the important historical information.
- Preserve the hook and conclusion.
- Keep the story chronological/logical.
- Remove repetition.
- Do not introduce new facts.
- Do not invent information.
- Return ONLY the revised narration.
- No title.
- No explanation.
- No labels.

NARRATION:
{text}
"""

    return clean_narration(
        qwen.generate(
            prompt,
            max_new_tokens=REVISION_TOKENS,
            temperature=0.45,
        )
    )


def remove_repeated_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    seen = set()
    output = []

    for sentence in sentences:
        normalized = re.sub(r"[^a-z0-9 ]", "", sentence.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        output.append(sentence.strip())

    return " ".join(output)


def run(paths, job_id, topic, verified, config, qwen):

    research = verified.get("research", verified)

    # ---------------------------------------------------------
    # STEP 1 — Generate the first narration
    # ---------------------------------------------------------
    print(f"[SCRIPT] {job_id} | STAGE 1/4 | Generating ~190-word narration")

    text = generate_initial(
        topic,
        research,
        config,
        qwen,
    )

    if not text:
        raise ValueError("Qwen returned an empty narration.")

    text = remove_repeated_sentences(text)

    count = word_count(text)

    print(
        f"[SCRIPT] {job_id} | Initial narration: "
        f"{count} words"
    )

    # ---------------------------------------------------------
    # STEP 2 — Correct excessive length
    # ---------------------------------------------------------
    if count > ACCEPT_MAX:

        print(
            f"[SCRIPT] {job_id} | STAGE 2/4 | "
            f"Too long ({count}) → shortening"
        )

        text = revise_length(
            text,
            f"Shorten this to approximately 190 words. "
            f"Final result must be between {IDEAL_MIN} and "
            f"{IDEAL_MAX} words if possible, and MUST NOT exceed "
            f"{ACCEPT_MAX} words.",
            qwen,
        )

        text = remove_repeated_sentences(text)
        count = word_count(text)

        print(
            f"[SCRIPT] {job_id} | After shortening: "
            f"{count} words"
        )

    # ---------------------------------------------------------
    # STEP 3 — Correct very short narration
    # ---------------------------------------------------------
    elif count < ACCEPT_MIN:

        print(
            f"[SCRIPT] {job_id} | STAGE 3/4 | "
            f"Too short ({count}) → expanding"
        )

        text = revise_length(
            text,
            f"Expand this naturally to approximately 190 words. "
            f"Final result should preferably be between "
            f"{IDEAL_MIN} and {IDEAL_MAX} words. "
            f"Only expand using information already present in "
            f"the supplied research. Do not invent facts.",
            qwen,
        )

        text = remove_repeated_sentences(text)
        count = word_count(text)

        print(
            f"[SCRIPT] {job_id} | After expansion: "
            f"{count} words"
        )

    # ---------------------------------------------------------
    # STEP 4 — Final safety correction
    # ---------------------------------------------------------
    if count > ACCEPT_MAX:

        print(
            f"[SCRIPT] {job_id} | STAGE 4/4 | "
            f"Still too long ({count}) → final compression"
        )

        text = revise_length(
            text,
            f"Compress this aggressively to 190 words or fewer. "
            f"Absolute maximum is {ACCEPT_MAX} words. "
            f"Remove repeated ideas, unnecessary adjectives, "
            f"filler and redundant explanations. "
            f"Do not remove the central historical facts.",
            qwen,
        )

        text = remove_repeated_sentences(text)
        count = word_count(text)

        print(
            f"[SCRIPT] {job_id} | Final count: "
            f"{count} words"
        )

    # ---------------------------------------------------------
    # FINAL VALIDATION
    # ---------------------------------------------------------
    if not text:
        raise ValueError("Narration became empty after processing.")

    if count > ACCEPT_MAX:
        raise ValueError(
            f"Narration failed validation: {count} words "
            f"(maximum {ACCEPT_MAX})."
        )

    if count < ACCEPT_MIN:
        print(
            f"[SCRIPT] {job_id} | WARNING: "
            f"{count} words is below preferred range, "
            f"but accepting the narration."
        )

    # ---------------------------------------------------------
    # SAVE OUTPUT
    # ---------------------------------------------------------
    write_text_atomic(
        paths.narration(job_id),
        text,
    )

    write_json_atomic(
        paths.script(job_id),
        {
            "word_count": count,
            "ideal_range": [IDEAL_MIN, IDEAL_MAX],
            "accepted_range": [ACCEPT_MIN, ACCEPT_MAX],
            "target_seconds": config.target_video_seconds,
            "format": "fast-paced vertical documentary",
            "status": "ready",
            "hook": (
                text.splitlines()[0]
                if text.splitlines()
                else ""
            ),
        },
    )

    print(
        f"[SCRIPT] {job_id} | COMPLETE | "
        f"{count} words"
    )

    return text
