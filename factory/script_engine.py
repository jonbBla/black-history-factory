import json
from .utils import write_text_atomic, write_json_atomic

def word_count(text):
    return len(text.split())

def looks_incomplete(text):
    text = text.strip()
    if not text:
        return True
    endings = {"and","but","because","while","which","that","the","a","an","to","of","with","for","from"}
    words = text.rstrip(" .,;:!?").split()
    return not words or words[-1].lower() in endings or text[-1] not in ".!?\"'”’"

def build_prompt(topic, research, config):
    return f"""Write ONE complete fast-paced vertical documentary narration.

TOPIC:
{topic.title}

TARGET: {config.narration_words_min}-{config.narration_words_max} spoken words.
TARGET LENGTH: {config.min_video_seconds}-{config.max_video_seconds} seconds.

STRUCTURE:
1. Stop-scroll hook.
2. Question or mystery.
3. Historical context.
4. Evidence/discovery.
5. Explanation.
6. Unexpected or lesser-known detail.
7. Significance.
8. Strong conclusion.

IMPORTANT:
- This is ONE continuous narration.
- Do not repeat the hook, paragraphs, or facts unnecessarily.
- No headings, bullet points, scene numbers, or source cards.
- Cover the story continuously from beginning to end.
- End naturally because a separate source card will be added later.
- Never present mythology, oral tradition, scholarly interpretation, or uncertainty as established fact.
- Clearly qualify disputed claims.
- Never invent people, places, artifacts, dates, technologies, sources, or quotations.
- Do not make colonization the central framing.

RESEARCH:
{json.dumps(research, ensure_ascii=False)[:18000]}

Return ONLY the narration."""

def generate_valid_narration(topic, research, config, qwen, attempts=3):
    prompt = build_prompt(topic, research, config)
    last_text = ""
    for attempt in range(attempts):
        current_prompt = prompt if attempt == 0 else prompt + f"""
The previous narration failed validation.
Previous word count: {word_count(last_text)}
Previous draft:
{last_text}

Rewrite the ENTIRE narration.
Requirements:
- {config.narration_words_min}-{config.narration_words_max} words.
- Complete final sentence.
- No repeated sections.
- No source card.
- No scene labels.
"""
        text = qwen.generate(current_prompt, max_new_tokens=1200,
                             temperature=0.65 if attempt else 0.75).strip()
        last_text = text
        count = word_count(text)
        if config.narration_words_min <= count <= config.narration_words_max and not looks_incomplete(text):
            return text
    raise ValueError(f"Narration failed validation after {attempts} attempts. Final word count: {word_count(last_text)}")

def run(paths, job_id, topic, verified, config, qwen):
    research = verified.get("research", verified)
    text = generate_valid_narration(topic, research, config, qwen)
    write_text_atomic(paths.narration(job_id), text)
    write_json_atomic(paths.script(job_id), {
        "word_count": word_count(text),
        "target_seconds": config.target_video_seconds,
        "format": "fast-paced vertical documentary",
        "hook": text.splitlines()[0] if text.splitlines() else "",
        "validated": True
    })
    return text
