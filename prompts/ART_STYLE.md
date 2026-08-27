# Series Art Style

**This is the one place to look for "what does this project's art style
look like, and where is it set in code."**

## The style

**Cinematic 3D Render / CGI**

- Octane render, Unreal Engine style -- clean, well-defined surfaces
- Volumetric lighting, dramatic atmosphere
- Highly detailed textures, sharp focus
- Not flat cartoon

## Why this style, specifically for SD-Turbo

This went through two earlier versions before landing here, each changed
after real generated output revealed a problem:

1. **"Historical cinematic oil realism"** (painterly brushwork,
   chiaroscuro) -- oil-painting brushwork is stochastic, high-frequency
   texture that needs several denoising steps to resolve cleanly.
   SD-Turbo only runs 1-4 steps by design, so this came out muddy.
2. **"Digital concept art / matte painting"** -- cleaner than oil painting,
   but still a *painting* category, and still produced results that
   weren't detailed/sharp enough at the requested quality bar.
3. **This version: cinematic 3D render / CGI** -- a completely different
   rendering paradigm from painting. 3D/CGI-render prompts are a hugely
   well-represented category in general text-to-image training data,
   emphasizing defined surfaces, materials, and lighting rather than
   painterly noise -- they render cleanly and sharply even at SD-Turbo's
   very low step count. It's also not tied to the "European romantic
   painting" genre bias that pulled earlier attempts toward a generic
   look regardless of the scene's actual intended content.

**This does not fully fix subject/content fidelity on its own.**
SD-Turbo's weak prompt-following is a separate, more fundamental
limitation from style choice. Two things DO help fidelity and were fixed
alongside this style change:
- `factory/scene_engine.py`'s scene-planning prompt now actually includes
  the episode's locked visual bible (region, period, architecture,
  clothing) when asking Qwen to write each scene's `visual_focus` --
  previously it didn't, so Qwen was guessing at attire/setting blind.
- `factory/image_engine.py` now AI-upscales each generated image (Real-
  ESRGAN, falling back to Lanczos resize if unavailable) before saving,
  instead of leaving SD-Turbo's small native resolution for ffmpeg to
  stretch blurrily later in video_engine.py.

If narration-content accuracy still isn't reliable enough after these
fixes, the next tier up (SDXL-Lightning, see image_engine.py) has
meaningfully stronger prompt adherence -- that's a model-capability
ceiling, not something further prompt engineering on SD-Turbo can fully
close.

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
   comma-boundary (not mid-word) to keep the whole prompt short and
   SD-Turbo-friendly (see that file's module docstring).

## To change the look of every future video

Edit `art_style` in `factory/config.py` (`DEFAULTS["art_style"]`) AND in
`00_CONFIG/config.json` on your Drive (both -- see the note above about
the code default alone not retroactively updating an existing Drive file).
