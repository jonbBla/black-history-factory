from .utils import (
    write_text_atomic,
    write_json_atomic
)


def word_count(text):

    return len(
        str(text).split()
    )


def is_incomplete(text):

    text = str(
        text or ""
    ).strip()

    if not text:
        return True

    words = text.split()

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
        "as",
        "or"
    }

    if words[-1].lower().strip(
        ".,!?;:"
    ) in bad_endings:

        return True

    return text[-1] not in (
        ".",
        "!",
        "?",
        "\"",
        "'",
        "”",
        "’"
    )


def build_prompt(
    topic,
    research,
    config,
    previous="",
    correction=""
):

    minimum = max(
        120,
        int(config.narration_words_min) - 25
    )

    maximum = (
        int(config.narration_words_max) + 25
    )

    return f"""
Write ONE complete narration for a fast-paced
vertical historical documentary.

TOPIC:
{topic.title}

REGION:
{topic.region}

PERIOD:
{topic.period}

TARGET:
Approximately {config.narration_words_min}-
{config.narration_words_max} words.

ACCEPTABLE RANGE:
{minimum}-{maximum} words.

The target is flexible.
Natural storytelling is more important than
hitting an exact number.

STORY STRUCTURE:

1. Curiosity hook
2. The question or mystery
3. Historical context
4. Evidence or discovery
5. Explanation
6. Unexpected detail
7. Why it matters
8. Strong conclusion

IMPORTANT:

- Write one continuous narration.
- Do not use headings.
- Do not use bullet points.
- Do not use scene numbers.
- Do not repeat the opening hook.
- Do not repeat sentences.
- Do not repeat the same fact.
- Do not restart the story.
- Do not write a second conclusion.
- Do not include a source card.
- Do not include citations.
- Do not include stage directions.
- End with a complete sentence.
- Do not stop halfway through a thought.
- Mythology must be identified as mythology.
- Oral tradition must be identified as oral tradition.
- Scholarly interpretation must be identified as interpretation.
- Uncertain claims must be identified as uncertain.
- Never invent sources, dates, people, places,
  artifacts, technologies or quotations.
- Do not make colonization the central focus.

The narration should feel like a human documentary
voiceover rather than an academic article.

{correction}

RESEARCH:

{research}

PREVIOUS DRAFT:

{previous}

Return ONLY the narration.
"""


def run(
    paths,
    job_id,
    topic,
    verified,
    config,
    qwen
):

    # Fact checker normally returns the research
    # fields directly. This also supports a nested
    # "research" structure if one appears later.

    if isinstance(
        verified,
        dict
    ):

        research = verified.get(
            "research",
            verified
        )

    else:

        research = {}


    previous = ""
    final_text = ""


    for attempt in range(3):

        correction = ""

        if attempt > 0:

            correction = f"""
The previous narration failed validation.

Previous word count:
{word_count(previous)}

Previous draft:
{previous}

Generate a COMPLETELY NEW version.

Do NOT append another paragraph.

Do NOT repeat the same sentences.

Make sure the narration reaches a natural
conclusion and ends with a complete sentence.

Keep it around {config.narration_words_min}-
{config.narration_words_max} words.
"""


        prompt = build_prompt(
            topic,
            research,
            config,
            previous=previous,
            correction=correction
        )


        final_text = qwen.generate(
            prompt,
            max_new_tokens=1200,
            temperature=(
                0.70
                if attempt == 0
                else 0.55
            )
        ).strip()


        previous = final_text

        count = word_count(
            final_text
        )


        minimum = max(
            120,
            int(config.narration_words_min) - 25
        )

        maximum = (
            int(config.narration_words_max) + 25
        )


        if (
            minimum <= count <= maximum
            and not is_incomplete(final_text)
        ):

            break


    else:

        raise ValueError(
            "Narration failed validation after "
            f"3 attempts. Final count: "
            f"{word_count(final_text)} words."
        )


    write_text_atomic(
        paths.narration(job_id),
        final_text
    )


    write_json_atomic(
        paths.script(job_id),
        {
            "word_count": word_count(
                final_text
            ),
            "target_seconds": (
                config.target_video_seconds
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
