"""Phase D -- real implementation.

Contract (unchanged):
  input:  topic + verified research package + config (for the locked
          series-wide art_style) + a QwenClient
  output: a dict of visual rules shared by every scene's image prompt:
    { period, region, architecture, clothing, materials, environment,
      lighting, style }

`style` is ALWAYS taken from config.art_style, never from the model --
the whole point of a visual bible is series-wide consistency, and the art
style was a deliberate one-time decision (see config.py), not something to
re-derive per topic. `lighting` may be specialized per topic (e.g. "torchlit
interior" vs "open savanna at dusk") but stays within the locked style's
overall look. Only architecture/clothing/materials/environment are
genuinely topic-specific and come from the model.

If Qwen's response is malformed or the call fails, this falls back to
placeholders rather than failing the whole job -- visual bible detail is a
quality-of-life improvement for image consistency, not something worth
losing a completed research+narration pipeline over.
"""

from __future__ import annotations
import json
import os

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")

_DEFAULT_STYLE = (
    "historical cinematic oil realism, painterly brushwork, warm "
    "directional lighting, muted earth-tone palette"
)
_DEFAULT_LIGHTING = "warm directional, late-afternoon or torchlight, strong chiaroscuro"


def _load_template(name: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(topic, research: dict) -> str:
    template = _load_template("visual_bible.txt")
    overview = (research or {}).get("overview", "") if isinstance(research, dict) else ""
    return template.format(
        topic_title=topic.title,
        region=topic.region,
        period=topic.period,
        research_summary=overview[:1500],  # keep the prompt bounded
    )


def run(topic, research: dict, config=None, qwen=None) -> dict:
    art_style = getattr(config, "art_style", None) if config else None
    art_style = art_style or _DEFAULT_STYLE

    base = {
        "period": getattr(topic, "period", "") or "unspecified",
        "region": getattr(topic, "region", "") or "unspecified",
        "architecture": "",
        "clothing": "",
        "materials": "",
        "environment": "",
        "lighting": _DEFAULT_LIGHTING,
        "style": art_style,   # locked -- never overwritten below
    }

    if qwen is not None:
        prompt = build_prompt(topic, research)
        try:
            result = qwen.generate_json(prompt, max_new_tokens=800)
        except ValueError:
            result = {}
        if isinstance(result, dict):
            for key in ("architecture", "clothing", "materials", "environment"):
                val = result.get(key)
                if isinstance(val, str) and val.strip():
                    base[key] = val.strip()
            lighting = result.get("lighting")
            if isinstance(lighting, str) and lighting.strip():
                base["lighting"] = lighting.strip()
            # `style` intentionally ignored even if the model returns one --
            # config.art_style is the single source of truth for the look.

    for key in ("architecture", "clothing", "materials", "environment"):
        if not base[key]:
            base[key] = "(not specified)"

    return base
