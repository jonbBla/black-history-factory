from __future__ import annotations
import re
from .utils import write_text_atomic, write_json_atomic

def word_count(text):
    return len(re.findall(r"\b\w+[’'\-]?\w*\b", str(text)))

def repeated_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.lower().strip())
    cleaned = [re.sub(r"[^a-z0-9 ]", "", s).strip() for s in sentences if s.strip()]
    return len(cleaned) != len(set(cleaned))

def valid_end(text):
    return bool(text and text.strip()[-1] in ".!?\"”’")

def build_prompt(topic, research, config, previous="", correction=""):
    return f"""Write ONE fast-paced documentary narration about {topic.title}.
Target {config.narration_words_min}-{config.narration_words_max} spoken words; ideal about 190. HARD MAXIMUM: 240 words.
Start with a curiosity hook, then context, evidence, one unexpected detail, significance and conclusion.
One continuous narration only. No headings, bullets, scene labels, citations or source card.
Never repeat the hook, sentences or facts. Mythology, oral tradition, scholarly interpretation and uncertainty must be labeled.
Never invent sources, dates, people, artifacts, technologies or quotations. Avoid colonization-centered framing.
Finish the story before 240 words.
{correction}
RESEARCH:
{research}
PREVIOUS DRAFT:
{previous}
RETURN ONLY THE NARRATION."""

def run(paths, job_id, topic, verified, config, qwen):
    research = verified.get("research", verified) if isinstance(verified, dict) else verified
    previous = ""
    text = ""
    minimum = max(150, int(config.narration_words_min) - 25)
    maximum = min(240, int(config.narration_words_max) + 20)
    for attempt in range(4):
        correction = "" if attempt == 0 else (
            f"Rewrite the ENTIRE draft from scratch. Previous draft had {word_count(previous)} words. "
            f"Keep the new draft between {minimum} and {maximum} words. Remove repetition and finish the conclusion early."
        )
        text = qwen.generate(build_prompt(topic, research, config, previous, correction), max_new_tokens=380,
                             temperature=0.55 if attempt else 0.7).strip()
        previous = text
        n = word_count(text)
        if minimum <= n <= maximum and valid_end(text) and not repeated_sentences(text):
            break
    else:
        raise ValueError(f"Narration failed validation after 4 attempts. Final word count: {word_count(text)}")
    write_text_atomic(paths.narration(job_id), text)
    write_json_atomic(paths.script(job_id), {"word_count": word_count(text), "target_seconds": config.target_video_seconds, "validated": True})
    return text
