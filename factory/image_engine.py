from __future__ import annotations

import gc
import re
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

from .utils import read_json


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


# ============================================================
# PROMPT SETTINGS
# ============================================================

STYLE_PROMPT = """
cinematic 3D historical reconstruction, high-end AAA game cinematic,
Unreal Engine style, highly detailed CGI, realistic 3D geometry,
physically based materials, realistic skin texture, natural cloth
folds, detailed surfaces, cinematic lighting, volumetric atmosphere,
dramatic depth, epic composition, realistic proportions,
high micro-detail, historically immersive environment
"""

NEGATIVE_PROMPT = """
cartoon, anime, illustration, painting, flat 2D art, low quality,
low resolution, blurry, pixelated, oversaturated, plastic skin,
waxy skin, deformed face, bad anatomy, extra fingers, missing fingers,
fused fingers, extra limbs, duplicate people, malformed hands,
distorted body, floating objects, impossible perspective, text,
logo, watermark, UI, cropped head, cropped feet, out of focus subject,
modern objects, modern clothing, modern architecture, anachronism
"""


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean(text):
    """Normalize whitespace WITHOUT truncating the text."""
    if text is None:
        return ""

    return re.sub(r"\s+", " ", str(text).strip())


def _clear_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _first(scene, *keys):
    """Return the first non-empty scene field."""
    for key in keys:
        value = scene.get(key)

        if value is not None and str(value).strip():
            return clean(value)

    return ""


# ============================================================
# STRUCTURED PROMPT
# ============================================================

def build_prompt(scene):
    """
    Convert the Qwen scene into the same structured prompt style
    used by the working Structured SDXL notebook.

    IMPORTANT:
    There is deliberately NO words[:55], character slicing,
    or other artificial truncation here.
    """

    visual_description = _first(
        scene,
        "visual_description",
        "image_description",
        "description"
    )

    image_prompt = _first(
        scene,
        "image_prompt"
    )

    camera = _first(
        scene,
        "camera",
        "camera_position"
    )

    narration = _first(
        scene,
        "narration"
    )

    # Main visual information supplied by Qwen.
    event_moment = visual_description or image_prompt

    if not event_moment:
        event_moment = (
            "a historically accurate reconstruction of the event "
            "described by the narration"
        )

    environment_setting = (
        "A historically accurate environment based on the scene, "
        "with region-specific architecture, terrain, vegetation, "
        "structures, materials, and spatial relationships. "
        "Use period-authentic details and avoid generic African "
        "architecture."
    )

    characters_appearance = (
        "People must appear historically and regionally appropriate, "
        "with realistic anatomy, historically appropriate hairstyles, "
        "clothing, textiles, jewelry, tools, posture, and activities. "
        "Do not introduce unjustified costumes or modern clothing."
    )

    lighting_atmosphere = (
        "dramatic natural cinematic lighting, realistic shadows, "
        "physically plausible illumination, volumetric atmosphere, "
        "atmospheric depth, detailed surfaces, dramatic historical "
        "mood and strong cinematic depth"
    )

    props_objects = (
        "Historically appropriate objects, tools, vessels, furniture, "
        "textiles, artifacts, architectural details, food and other "
        "objects specifically relevant to the scene. Avoid modern "
        "objects and anachronistic technology."
    )

    unreal_rendering = clean(STYLE_PROMPT)

    camera_position = camera or (
        "cinematic wide establishing composition, strong depth, "
        "clear focal subject, natural perspective and balanced framing"
    )

    # --------------------------------------------------------
    # This follows the proven notebook's structure:
    #
    # EVENT
    # ENVIRONMENT
    # CHARACTERS
    # LIGHTING
    # PROPS
    # RENDERING
    # CAMERA
    #
    # The complete prompt is passed to SDXL.
    # --------------------------------------------------------

    final_prompt = f"""
A cinematic historical scene depicting {event_moment}.

Environment and setting: {environment_setting}

Characters and appearance: {characters_appearance}

Lighting and atmosphere: {lighting_atmosphere}

Props and objects: {props_objects}

Rendering: {unreal_rendering}

Camera: {camera_position}

Historical context from the narration: {narration}

Prioritize physically believable anatomy, authentic materials,
realistic skin texture, natural cloth folds, accurate scale,
environmental storytelling, coherent spatial relationships,
period-specific materials, historically appropriate architecture,
realistic textures, sharp focal subject, cinematic composition,
subtle filmic color response, dramatic scale, high micro-detail,
and a highly detailed cinematic 3D historical reconstruction.
"""

    return clean(final_prompt), clean(NEGATIVE_PROMPT)


# ============================================================
# MODEL LOADER
# ============================================================

def load_sdxl_lightning(model_id=None):
    """
    Kept under the existing function name so the Image Processor
    notebook does not need to change.

    The actual model is stable SDXL Base.
    """

    model_id = model_id or MODEL_ID

    print(f"[IMAGE] Loading SDXL: {model_id}")

    dtype = torch.float16

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16",
        add_watermarker=False,
    )

    pipe.enable_model_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    _clear_memory()

    print("[IMAGE] SDXL loaded.")

    return pipe


# ============================================================
# SINGLE SCENE GENERATION
# ============================================================

def generate_scene(pipe, scene, output_path, config):
    """
    Generate one missing scene image.

    The complete structured prompt is passed to SDXL without
    artificial shortening.
    """

    final_prompt, negative_prompt = build_prompt(scene)

    width = int(getattr(config, "image_width", 768))
    height = int(getattr(config, "image_height", 1344))

    steps = int(getattr(config, "image_steps", 28))
    guidance = float(getattr(config, "image_guidance_scale", 7.0))

    # Generate one image at a time for Colab/T4 stability.
    seed = torch.randint(
        0,
        2**32 - 1,
        (1,),
        device="cpu"
    ).item()

    generator = torch.Generator(device="cpu").manual_seed(seed)

    print(f"[IMAGE] Prompt characters: {len(final_prompt)}")
    print(f"[IMAGE] Prompt words: {len(final_prompt.split())}")
    print(f"[IMAGE] Seed: {seed}")

    result = pipe(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )

    image = result.images[0]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image.save(output_path)

    _clear_memory()

    return output_path


# ============================================================
# IMAGE PROCESSOR
# ============================================================

def run(paths, jid, scenes, pipe, config, progress=None):
    """
    Generate only missing images.

    Existing images are NEVER regenerated.
    """

    total = len(scenes)

    for index, scene in enumerate(scenes, start=1):

        scene_number = int(
            scene.get(
                "scene_number",
                scene.get("scene_id", index)
            )
        )

        image_path = paths.image(jid, scene_number)

        # ----------------------------------------------------
        # RESUME SUPPORT
        # ----------------------------------------------------

        if image_path.exists() and image_path.stat().st_size > 0:
            print(
                f"[IMAGE] Scene {scene_number}/{total} "
                f"already exists — skipping."
            )

            if progress:
                progress(index, total)

            continue

        print(
            f"[IMAGE] Generating scene "
            f"{scene_number}/{total}"
        )

        # ----------------------------------------------------
        # SHOW PROMPT INFORMATION
        # ----------------------------------------------------

        final_prompt, _ = build_prompt(scene)

        print(
            f"[IMAGE] Prompt length: "
            f"{len(final_prompt.split())} words / "
            f"{len(final_prompt)} characters"
        )

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        generate_scene(
            pipe=pipe,
            scene=scene,
            output_path=image_path,
            config=config,
        )

        print(
            f"[IMAGE] Saved scene {scene_number}: "
            f"{image_path}"
        )

        if progress:
            progress(index, total)

        _clear_memory()

    return True
