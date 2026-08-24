# Series Art Style

**This is the one place to look for "what does this project's art style
look like, and where is it set in code."**

## The style

**Historical Cinematic Oil Realism**

- Painterly brushwork -- not photoreal, not a glossy 3D render
- Warm, directional lighting (late-afternoon sun or torchlight), strong
  chiaroscuro (high contrast between light and shadow)
- Muted earth-tone palette with selective warm accent colors
- Wide, cinematic compositions -- documentary establishing-shot framing
- Textures emphasized (cloth, stone, metal, skin) over a glossy finish

## Where it lives in code

**`factory/config.py` -> `DEFAULTS["art_style"]`** is the single source of
truth. It's also written into `00_CONFIG/config.json` on Drive (both at
first run, and automatically if an older config file on Drive is missing
the key), so you can edit it there too.

From there:
1. `factory/visual_bible.py` reads `config.art_style` and puts it in the
   `style` field of the visual bible -- and ONLY `config.art_style`; even
   if the research model tries to suggest a different style, that value is
   discarded.
2. `factory/scene_engine.py`'s `_compose_image_prompt()` puts the visual
   bible's `style` first in every scene's `image_prompt` string (trimmed
   to fit the active image backend's token budget -- see that file's
   module docstring for the current SDXL-Lightning-tuned budget).

## To change the look of every future video

Edit `art_style` in EITHER `factory/config.py` (`DEFAULTS["art_style"]`)
or `00_CONFIG/config.json` on your Drive. Both flow through the same path
automatically.
