import json
from .utils import write_text_atomic, write_json_atomic


def word_count(text):
    return len(text.split())


def incomplete(text):
    text = text.strip()

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
        "from"
    }

    if words[-1].lower() in bad_endings:
        return True

    # Narration should normally finish with punctuation.
    if text[-1] not in ".!?'\"”’":
        return True

    return False


def build_prompt(
    topic,
    research,
    config,
    previous="",
    correction=""
):
    return f"""
Write ONE complete, engaging, fast-paced vertical documentary narration.

TOPIC:
{topic.title}

TARGET LENGTH:
Approximately {config.narration_words_min}-{config.narration_words_max}
spoken words.

IMPORTANT:
The word count is a target, NOT a hard requirement.
Natural storytelling is more important than hitting the exact number.

A narration between roughly
{max(120, config.narration_words_min - 20)}
and
{config.narration_words_max + 20}
words is acceptable.

STRUCTURE:

1. Strong curiosity hook.
2. The question, mystery, contradiction, or surprising detail.
3. Historical context.
4. Evidence or discovery.
5. Explanation.
6. Unexpected or lesser-known detail.
7. Why it matters.
8. Strong conclusion.

RULES:

- This must be ONE continuous narration.
- Start with something that makes the viewer want to keep watching.
- Do NOT repeat the hook later.
- Do NOT repeat paragraphs.
- Do NOT repeat the same fact unnecessarily.
- Do NOT restart the story halfway through.
- Do NOT use headings.
- Do NOT use bullet points.
- Do NOT include scene numbers.
- Do NOT include a source card.
- Do NOT include citations inside the narration.
- End with a complete sentence.
- Mythology must be identified as mythology.
- Oral traditions must be identified as oral traditions.
- Scholarly interpretations must be identified as interpretations.
- Uncertain claims must be clearly qualified.
- Never invent people, places, dates, artifacts, technologies,
  sources, quotations, or archaeological discoveries.
- Do not make colonization the central framing.
- Keep the narration conversational and compelling.

{correction}

RESEARCH:
{research}

PREVIOUS DRAFT:
{previous}

Return ONLY the narration text.
"""


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

    previous = ""
    text = ""

    # Give Qwen up to three attempts.
    for attempt in range(3):

        correction = ""

        if attempt:
            correction = f"""
The previous draft failed validation.

Previous word count:
{word_count(previous)}

Previous draft:
{previous}

Rewrite the ENTIRE narration.

Do not simply append text.

Make sure the new narration:
- is complete
- does not repeat itself
- has a strong hook
- flows from beginning to end
- ends with a complete sentence
- is approximately {config.narration_words_min}-{config.narration_words_max} words
"""

        prompt = build_prompt(
            topic,
            research,
            config,
            previous=previous,
            correction=correction
        )

        text = qwen.generate(
            prompt,
            max_new_tokens=1200,
            temperature=0.70 if attempt == 0 else 0.55
        ).strip()

        previous = text

        count = word_count(text)

        # Reasonable range rather than an exact count.
        minimum = max(
            120,
            config.narration_words_min - 20
        )

        maximum = (
            config.narration_words_max + 20
        )

        if (
            minimum <= count <= maximum
            and not incomplete(text)
        ):
            break

    else:
        raise ValueError(
            "Narration failed validation after "
            f"3 attempts. Final word count: "
            f"{word_count(text)}"
        )

    write_text_atomic(
        paths.narration(job_id),
        text
    )

    write_json_atomic(
        paths.script(job_id),
        {
            "word_count": word_count(text),
            "target_seconds": config.target_video_seconds,
            "format": "fast-paced vertical documentary",
            "hook": (
                text.splitlines()[0]
                if text.splitlines()
                else ""
            ),
            "validated": True
        }
    )

    return text
