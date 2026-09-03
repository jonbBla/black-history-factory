from __future__ import annotations

import re

MIN_NARRATION_WORDS = 170
MAX_NARRATION_WORDS = 220
TARGET_NARRATION_WORDS = 195
MAX_NARRATION_ATTEMPTS = 6


def count_words(text):
    return len(re.findall(r"\b[\w’'-]+\b", text))


def clean_narration(text):
    text = text.strip()
    text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(narration|script)\s*:\s*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def valid_length(text):
    n = count_words(text)
    return MIN_NARRATION_WORDS <= n <= MAX_NARRATION_WORDS


def prompt_for(topic, research, fact_check, visual_bible, correction=""):
    return f"""
Write the final narration for a fast-paced 80–100 second historical documentary.

TOPIC:
{topic.title}

EVIDENCE DOSSIER:
{research}

FACT-CHECK / VALIDATION:
{fact_check}

VISUAL CONTEXT:
{visual_bible}

LENGTH:
Target approximately {TARGET_NARRATION_WORDS} words.
HARD minimum: {MIN_NARRATION_WORDS} words.
HARD maximum: {MAX_NARRATION_WORDS} words.

STYLE:
- Begin with a question, surprising detail, contradiction, or unresolved puzzle.
- Keep the story moving through context, evidence, explanation, unexpected detail,
  significance and conclusion.
- Use only claims supported by the evidence dossier/fact-check.
- Never present mythology, oral tradition, interpretation or uncertainty as fact.
- Avoid colonization-centered framing.
- Natural spoken English and punctuation.
- No headings, scene numbers, visual directions or production notes.
- End naturally so a blank source card can follow.

Return ONLY the narration.
{correction}
"""


def generate_narration(qwen, topic, research, fact_check, visual_bible):
    best = None
    best_distance = float("inf")

    previous_count = None

    for attempt in range(1, MAX_NARRATION_ATTEMPTS + 1):
        correction = ""

        if previous_count is not None:
            if previous_count < MIN_NARRATION_WORDS:
                correction = f"""
CORRECTION: The previous attempt was {previous_count} words.
It was too short. Rewrite it as a complete, useful narration of
{MIN_NARRATION_WORDS}–{MAX_NARRATION_WORDS} words. Add substantive context,
not filler.
"""
            elif previous_count > MAX_NARRATION_WORDS:
                correction = f"""
CORRECTION: The previous attempt was {previous_count} words.
It was too long. Rewrite it to {MIN_NARRATION_WORDS}–{MAX_NARRATION_WORDS} words.
Remove repetition and filler while preserving important evidence.
"""

        print(f"[SCRIPT] ATTEMPT {attempt}/{MAX_NARRATION_ATTEMPTS} | Generating narration")

        raw = qwen.generate(
            prompt_for(topic, research, fact_check, visual_bible, correction),
            max_new_tokens=800,
            temperature=0.35,
        )
        script = clean_narration(raw)
        n = count_words(script)
        print(f"[SCRIPT] Attempt {attempt}: {n} words")

        distance = abs(n - TARGET_NARRATION_WORDS)
        if distance < best_distance:
            best = script
            best_distance = distance

        if valid_length(script):
            print(f"[SCRIPT] ACCEPTED | {n} words")
            return script

        previous_count = n

    raise ValueError(
        f"Narration generation failed after {MAX_NARRATION_ATTEMPTS} attempts. "
        f"Best attempt was {count_words(best) if best else 0} words; "
        f"required {MIN_NARRATION_WORDS}-{MAX_NARRATION_WORDS}."
    )


def run(paths, job_id, topic, research, fact_check, visual_bible, config, qwen):
    print(f"[QWEN] {job_id} | STAGE: NARRATION | writing and validating narration")
    narration = generate_narration(qwen, topic, research, fact_check, visual_bible)
    with open(paths.narration(job_id), "w", encoding="utf8") as f:
        f.write(narration)
    return narration
