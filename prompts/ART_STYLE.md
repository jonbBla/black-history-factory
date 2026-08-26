# Series Art Style

**This is the one place to look for "what does this project's art style
look like, and where is it set in code."**

## The style

**Digital Concept Art / Matte Painting**

- Richly detailed environments -- lighting, atmosphere, texture
- Warm cinematic lighting, dramatic atmospheric depth
- Not photoreal, not flat cartoon

## Why this style, specifically for SD-Turbo

An earlier version of this style was "historical cinematic oil realism"
(painterly brushwork, chiaroscuro). That was changed after real generated
output showed two problems specific to running on SD-Turbo (the default
image backend):

1. **Oil-painting brushwork is stochastic, high-frequency texture** that
   needs several denoising steps to resolve cleanly. SD-Turbo only runs
   1-4 steps by design -- asking it for brushwork texture at that step
   count produced muddy, undefined results rather than clean detail.
2. **"Oil painting + warm lighting + old buildings" is a heavily
   over-represented genre** in general text-to-image training data
   (atmospheric European city paintings are extremely common on art
   sites). Combined with SD-Turbo's weak prompt alignment, this pulled
   generations toward a generic "old European street at dusk" look
   regardless of the scene's actual intended subject/region.

Digital concept art / matte painting renders with cleaner shape
definition even at very few steps, stays genuinely detailed, and isn't
as strongly pre-associated with one specific cultural/regional archetype.

**This does not fully fix subject/content fidelity.** SD-Turbo's weak
prompt-following is a separate, more fundamental limitation from style
choice -- a style change makes output cleaner and more detailed, but
doesn't guarantee the model reliably depicts specific cultural/regional
content (e.g. correctly rendering a Moroccan setting rather than a
generic one) every time. If that fidelity matters more than the resource
savings, the next tier up (SDXL-Lightning, see image_engine.py) has
meaningfully stronger prompt adherence.

## Where it lives in code

**`factory/config.py` -> `DEFAULTS["art_style"]`** is the single source of
truth. It's ALSO written into `00_CONFIG/config.json` on Drive at first
run -- but only if that key is missing there. **If `config.json` already
exists on your Drive with an older art_style value, changing the code
default alone will NOT update it** -- edit `art_style` directly in your
Drive's `config.json`, or delete just that key from the file so
`Config.load()`'s self-healing fills it back in with the current default.

From there:
1. `factory/visual_bible.py` reads `config.art_style` and puts it in the
   `style` field of the visual bible -- and ONLY `config.art_style`; even
   if the research model tries to suggest a different style, that value is
   discarded.
2. `factory/scene_engine.py`'s `_compose_image_prompt()` puts the visual
   bible's `style` first in every scene's `image_prompt` string, cut at a
   comma-boundary (not mid-word) to whatever length keeps the whole prompt
   short and SD-Turbo-friendly (see that file's module docstring).

## To change the look of every future video

Edit `art_style` in `factory/config.py` (`DEFAULTS["art_style"]`) AND in
`00_CONFIG/config.json` on your Drive (both -- see the note above about
the code default alone not retroactively updating an existing Drive file).
