from __future__ import annotations
import re
MIN_NARRATION_WORDS=170
TARGET_NARRATION_WORDS=220
MAX_NARRATION_WORDS=270
MAX_ATTEMPTS=6

def count_words(text): return len(re.findall(r"\b[\w’'-]+\b",text or ""))
def clean_narration(text):
    text=(text or "").strip(); text=re.sub(r"^```(?:text)?\s*","",text,flags=re.I); text=re.sub(r"\s*```$","",text); text=re.sub(r"^(narration|script)\s*:\s*","",text,flags=re.I); return re.sub(r"\s+"," ",text).strip()
def _prompt(topic,research,fact_check,correction=""):
    return f'''Write the final spoken narration for a fast-paced vertical historical documentary.

TOPIC: {topic.title}
RESEARCH:
{research}
FACT CHECK:
{fact_check}

LENGTH: target about {TARGET_NARRATION_WORDS} words; minimum {MIN_NARRATION_WORDS}; maximum {MAX_NARRATION_WORDS}. The final video may be under two minutes or slightly over two minutes when the story needs it. Do not pad for length.

STYLE:
- Open with a question, surprise, contradiction, mystery, or striking fact.
- Move quickly through context, evidence, culture, people/places, an unexpected detail, significance, and conclusion.
- Include useful visual/historical details when supported, especially attire/textiles, architecture, technology, art, food/daily life, and customs.
- Use only claims verified/supported by the research and fact check.
- Clearly phrase uncertainty, oral tradition, mythology, and scholarly disagreement.
- Do not make colonization the central framing.
- Natural spoken English. No headings, labels, scene directions, citations, or production notes.
- End naturally before the source card.
Return ONLY the narration.
{correction}'''
def run(paths,job_id,topic,research,fact_check,config,qwen):
    previous=None
    for attempt in range(1,MAX_ATTEMPTS+1):
        correction=""
        if previous is not None:
            correction=(f"The previous draft was {previous} words. Rewrite it with more substantive supported detail and make it {MIN_NARRATION_WORDS}-{MAX_NARRATION_WORDS} words." if previous<MIN_NARRATION_WORDS else f"The previous draft was {previous} words. Rewrite it shorter, removing repetition, and keep it {MIN_NARRATION_WORDS}-{MAX_NARRATION_WORDS} words.")
        print(f"[SCRIPT] ATTEMPT {attempt}/{MAX_ATTEMPTS}")
        text=clean_narration(qwen.generate(_prompt(topic,research,fact_check,correction),max_new_tokens=1100,temperature=0.35)); n=count_words(text); print(f"[SCRIPT] {n} words")
        if MIN_NARRATION_WORDS<=n<=MAX_NARRATION_WORDS:
            with open(paths.narration(job_id),"w",encoding="utf-8") as f:f.write(text)
            return text
        previous=n
    raise ValueError(f"Narration failed after {MAX_ATTEMPTS} attempts; last length {previous} words.")
