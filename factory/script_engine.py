from __future__ import annotations

import re
import json

from .utils import write_text_atomic, write_json_atomic


# ---------------------------------------------------------
# BASIC TEXT UTILITIES
# ---------------------------------------------------------

def word_count(text: str) -> int:
    return len(str(text).split())


def normalize_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def incomplete(text: str) -> bool:
    text = normalize_text(text)

    if not text:
        return True

    words = text.rstrip(
        ' .,;:!?\'"”’'
    ).split()

    if not words:
        return True

    bad_endings = {
        "and",
        "but",
        "because",
        "while",
        "which",
        "that",
        "the",
        "a",
        "an",
        "to",
        "of",
        "with",
        "for",
        "from",
        "is",
        "was",
        "were",
        "as",
        "in",
        "on",
        "by",
    }

    if words[-1].lower() in bad_endings:
        return True

    # Must end naturally.
    if text[-1] not in ".!?'\"”’":
        return True

    return False


def repeated_content(text: str) -> bool:
    """
    Lightweight repetition detector.

    It catches obvious cases such as:
      "Did X happen?"
      ...
      "Did X happen?"

    and repeated multi-word phrases.
    """

    text = normalize_text(text).lower()

    if not text:
        return False

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned = []

    for sentence in sentences:
        sentence = re.sub(
            r"[^a-z0-9\s]",
            "",
            sentence
        )

        sentence = re.sub(
            r"\s+",
            " ",
            sentence
        ).strip()

        if sentence:
            cleaned.append(sentence)

    # Exact repeated sentences.
    seen = set()

    for sentence in cleaned:
        if sentence in seen:
            return True

        seen.add(sentence)

    # Repeated 5-word chunks.
    chunks = {}

    words = text.split()

    for i in range(len(words) - 4):
        chunk = " ".join(words[i:i + 5])

        if chunk in chunks:
            return True

        chunks[chunk] = True

    return False


# ---------------------------------------------------------
# PROMPT
# ---------------------------------------------------------

def build_prompt(
    topic,
    research,
    config,
    previous="",
    correction=""
):

    target_min = int(
        getattr(
            config,
            "narration_words_min",
            175
        )
    )

    target_max = int(
        getattr(
            config,
            "narration_words_max",
            220
        )
    )

    return f"""
Write ONE complete narration for a fast-paced
90-second vertical historical documentary.

TOPIC:
{topic.title}

TARGET:
{target_min}-{target_max} spoken words.

IMPORTANT LENGTH RULE:

Aim for about 190 spoken words.

Do NOT write a long essay.

Do NOT exceed approximately 240 words.

The narration must finish the entire story within this
short length.

STRUCTURE:

1. HOOK
Start with a surprising question, mystery, contradiction,
unexpected achievement, or detail that makes the viewer
immediately curious.

2. CONTEXT
Quickly explain who, what, where and when.

3. EVIDENCE
Explain what is actually known and what evidence supports it.

4. SURPRISING DETAIL
Give one genuinely interesting lesser-known detail.

5. SIGNIFICANCE
Explain why the subject matters.

6. CONCLUSION
Finish the story with a strong final sentence.

VERY IMPORTANT:

- Write ONE continuous narration.
- Do not use headings.
- Do not use bullet points.
- Do not number anything.
- Do not include scene numbers.
- Do not include a source card.
- Do not include citations.
- Do not repeat the opening hook.
- Do not repeat sentences.
- Do not repeat facts.
- Do not restart the story.
- Do not create multiple conclusions.
- Do not pad the narration to make it longer.
- Every paragraph must move the story forward.
- Keep sentences reasonably short for narration.
- Finish the conclusion before reaching 240 words.
- End with a complete sentence.

HISTORICAL ACCURACY:

- Established facts must be presented as facts.
- Archaeological evidence must be described as evidence.
- Scholarly interpretations must be identified as interpretations.
- Oral traditions must be identified as oral traditions.
- Mythology must be identified as mythology.
- Uncertain claims must be qualified.
- Never invent sources.
- Never invent dates.
- Never invent people.
- Never invent archaeological discoveries.
- Never invent technologies.
- Never invent quotations.
- Never invent buildings or artifacts.
- Do not make colonization the central framing.

{correction}

RESEARCH:

{research}

PREVIOUS DRAFT:

{previous}

RETURN ONLY THE FINAL NARRATION.
"""


# ---------------------------------------------------------
# SAFE GENERATION
# ---------------------------------------------------------

def _generate_attempt(
    qwen,
    prompt,
    temperature
):
    """
    Keep the token budget intentionally small.

    360 tokens is enough for a ~190-220 word narration
    while preventing the 1.5B model from wandering into
    400+ word outputs.
    """

    text = qwen.generate(
        prompt,
        max_new_tokens=360,
        temperature=temperature
    )

    return normalize_text(text)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def run(
    paths,
    job_id,
    topic,
    verified,
    config,
    qwen
):

    research = verified.get(
        "research",
        verified
    )

    target_min = int(
        getattr(
            config,
            "narration_words_min",
            175
        )
    )

    target_max = int(
        getattr(
            config,
            "narration_words_max",
            220
        )
    )

    # Flexible acceptance range.
    #
    # This means the narration does NOT need to hit exactly
    # 175-220 words.
    #
    # 150-250 is acceptable for the factory.
    minimum = max(
        140,
        target_min - 25
    )

    maximum = min(
        250,
        target_max + 30
    )

    previous = ""
    last_error = ""
    final_text = ""

    for attempt in range(100):

        correction = ""

        if attempt == 1:
            correction = f"""
STRICT CORRECTION:

Your previous narration was not suitable.

Previous word count:
{word_count(previous)}

Write the ENTIRE narration again.

Target:
{target_min}-{target_max} words.

Hard maximum:
{maximum} words.

Do not add more information.
Compress the story.

Make sure the conclusion happens before the narration ends.
"""

        elif attempt == 2:
            correction = f"""
EXTREME LENGTH CORRECTION:

Rewrite the entire narration.

The previous draft was too long.

Previous word count:
{word_count(previous)}

You MUST keep the new narration between
{minimum} and {maximum} words.

Use only the strongest facts.

Remove:
- repeated ideas
- unnecessary descriptions
- filler
- repeated context
- repeated conclusions

Keep:
- hook
- context
- evidence
- one surprising detail
- significance
- conclusion

Finish the story completely.
"""

        elif attempt == 3:
            correction = f"""
FINAL CORRECTION:

Produce a SHORT documentary narration.

Maximum:
{maximum} words.

Ideal:
190 words.

Do not explain everything.
Choose only the most important and interesting facts.

The final sentence MUST conclude the story.

Do not repeat anything from the previous draft.
"""

        prompt = build_prompt(
            topic,
            research,
            config,
            previous=previous,
            correction=correction
        )

        temperature = (
            0.65
            if attempt == 0
            else 0.45
        )

        try:
            text = _generate_attempt(
                qwen,
                prompt,
                temperature
            )
        except Exception as e:
            last_error = str(e)
            continue

        previous = text
        final_text = text

        count = word_count(text)

        errors = []

        if count < minimum:
            errors.append(
                f"too short ({count} words)"
            )

        if count > maximum:
            errors.append(
                f"too long ({count} words)"
            )

        if incomplete(text):
            errors.append(
                "incomplete ending"
            )

        if repeated_content(text):
            errors.append(
                "repeated content"
            )

        if not errors:
            break

        last_error = "; ".join(errors)

    else:
        raise ValueError(
            "Narration failed validation after "
            f"4 attempts. Final count: "
            f"{word_count(final_text)} words. "
            f"Reason: {last_error}"
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
            "word_count": word_count(final_text),
            "target_words": {
                "min": target_min,
                "max": target_max
            },
            "accepted_range": {
                "min": minimum,
                "max": maximum
            },
            "target_seconds": int(
                getattr(
                    config,
                    "target_video_seconds",
                    90
                )
            ),
            "format": (
                "fast-paced vertical documentary"
            ),
            "hook": (
                final_text.splitlines()[0]
                if final_text.splitlines()
                else ""
            ),
            "validated": True
        }
    )

    return final_text
