"""Real narration-generation implementation.

Contract:
  input:  verified research package + topic title + config
          (target_video_minutes) + a QwenClient
  output: paths.narration_txt(job_id) -- plain text documentary narration
          following the beat structure in prompts/narration.txt (Hook,
          Context, Story, Unexpected discovery, Explanation, Significance,
          Conclusion), written as flowing narration rather than labeled
          sections.

This is a plain-text generation call (qwen.generate, not generate_json) --
narration is prose, not structured data.
"""

from __future__ import annotations
import json
import os
from .utils import now_iso

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")


def _load_template(name: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(research: dict, config) -> str:
    template = _load_template("narration.txt")
    return template.format(
        target_minutes=config.target_video_minutes,
        research_json=json.dumps(research, indent=2),
    )


def run(paths, job_id: str, title: str, research: dict, config, qwen=None) -> str:
    out = paths.narration_txt(job_id)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if qwen is None:
        text = (
            f"[NO MODEL LOADED] Narration not generated for '{title}'.\n"
            f"generated_at: {now_iso()}\n"
        )
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        return text

    prompt = build_prompt(research, config)
    try:
        text = qwen.generate(prompt, max_new_tokens=2500, temperature=0.8)
    except Exception as e:
        raise RuntimeError(f"Narration generation failed for '{title}': {e}") from e

    text = text.strip()
    if not text:
        raise RuntimeError(f"Narration generation returned empty text for '{title}'")

    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return text
