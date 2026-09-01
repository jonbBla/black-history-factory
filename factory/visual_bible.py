from __future__ import annotations
import os
PROMPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")

def run(topic, research, config, qwen):
    template = open(os.path.join(PROMPTS, "visual_bible.txt"), encoding="utf8").read()
    prompt = template.format(
        topic_title=topic.title, region=topic.region, period=topic.period,
        research_summary=str(research.get("overview", ""))[:1800],
        art_style=config.art_style_text,
    )
    result = qwen.generate_json(prompt, max_new_tokens=700)
    out = {
        "period": topic.period or "unspecified", "region": topic.region or "unspecified",
        "architecture": "", "clothing": "", "materials": "", "environment": "",
        "people": "", "lighting": "dramatic natural lighting", "style": config.art_style_text,
    }
    if isinstance(result, dict):
        for key in ("architecture", "clothing", "materials", "environment", "people", "lighting"):
            if isinstance(result.get(key), str) and result[key].strip():
                out[key] = result[key].strip()
    return out
