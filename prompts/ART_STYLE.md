# Series Art Style

**This is the one place to look for "what does this project's art style
look like, and where is it set in code."**

## The style

**Historical Cinematic Oil Realism**

- Painterly brushwork — not photoreal, not a glossy 3D render
- Warm, directional lighting (late-afternoon sun or torchlight), strong
  chiaroscuro (high contrast between light and shadow)
- Muted earth-tone palette with selective warm accent colors
- Wide, cinematic compositions — documentary establishing-shot framing
- Textures emphasized (cloth, stone, metal, skin) over a glossy finish

Chosen over photoreal/3D because it reads as "historical reconstruction"
rather than uncanny-valley photo-fakery, and gives FLUX something painterly
diffusion models are reliably good at, instead of fighting it toward
photoreal skin/fabric rendering it's weaker at.

## Where it lives in code

**`factory/config.py` → `DEFAULTS["art_style"]`** is the single source of
truth. It's also written into `00_CONFIG/config.json` on Drive (both at
first run, and automatically if an older config file on Drive is missing
the key), so you can edit it there too — either place works, they're the
same value once loaded.

From there:

1. `factory/visual_bible.py` reads `config.art_style` and puts it in the
   `style` field of the visual bible — and **only** `config.art_style`;
   even if the research model tries to suggest a different style, that
   value is discarded (see the comment in `visual_bible.py`). This is
   deliberate: the whole point of a visual bible is series-wide
   consistency, and style was a one-time decision, not something to
   re-derive per topic.
2. `factory/scene_engine.py`'s `_compose_image_prompt()` puts the visual
   bible's `style` first in every single scene's `image_prompt` string —
   so every image FLUX generates, across every episode, starts from the
   same style description.

## To change the look of every future video

Edit `art_style` in **one** of:
- `factory/config.py` (`DEFAULTS["art_style"]`) — changes the default for
  any fresh Drive setup
- `00_CONFIG/config.json` on your Drive — changes it for your existing
  project without touching code

Both flow through the same path above automatically; nothing else needs
editing.
