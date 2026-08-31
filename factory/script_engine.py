import json
from .utils import write_text_atomic,write_json_atomic

def run(paths,job_id,topic,verified,config,qwen):
    research=verified.get('research',verified)
    prompt=f'''Write a fast-paced ~90-second vertical documentary narration about {topic.title}. Target {config.narration_words_min}-{config.narration_words_max} spoken words and {config.min_video_seconds}-{config.max_video_seconds} seconds. Start with a question, surprising detail, contradiction, or unresolved puzzle that makes viewers stop. Then answer it through the end. Beat flow: hook, question, context, discovery, explanation, unexpected detail, significance, conclusion. Be concise. Never present mythology, oral tradition, scholarly interpretation or uncertainty as established fact. Avoid colonization-centered framing. End naturally so a blank source card can follow. Research: {json.dumps(research,ensure_ascii=False)[:15000]}'''
    text=qwen.generate(prompt,max_new_tokens=1800,temperature=.75).strip()
    words=len(text.split())
    if not config.narration_words_min<=words<=config.narration_words_max:
        text=qwen.generate(prompt+f'\nRewrite the draft to fit {config.narration_words_min}-{config.narration_words_max} words. Previous draft was {words} words.',max_new_tokens=1800,temperature=.65).strip()
    write_text_atomic(paths.narration(job_id),text)
    write_json_atomic(paths.script(job_id),{'word_count':len(text.split()),'target_seconds':config.target_video_seconds,'format':'fast-paced vertical documentary','hook':text.splitlines()[0] if text.splitlines() else ''})
    return text
