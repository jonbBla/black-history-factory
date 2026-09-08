# factory/image_engine.py

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Callable, Optional

import torch
from diffusers import StableDiffusionXLPipeline

from .utils import read_json


# ============================================================
# MODEL
# ============================================================

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


# ============================================================
# MEMORY
# ============================================================

def _clear_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ============================================================
# LOAD MODEL
# ============================================================

def load_sdxl_lightning(model_id: Optional[str] = None):
    """
    Loads the stable SDXL Base pipeline.

    The function name is retained for compatibility with the
    existing Image Processor notebook.
    """

    model_id = model_id or MODEL_ID

    print(f"[IMAGE] Loading SDXL model: {model_id}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        add_watermarker=False,
    )

    # Important for Colab T4 / limited VRAM.
    pipe.enable_model_cpu_offload()

    # Reduce VAE memory usage.
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    _clear_memory()

    print("[IMAGE] SDXL loaded.")

    return pipe


# ============================================================
# ART STYLE
# ============================================================

DEFAULT_ART_STYLE = {
    "primary": "cinematic 3D historical reconstruction",

    "description": (
        "epic cinematic historical reconstruction, "
        "high-end AAA game cinematic, "
        "Unreal Engine style, "
        "Octane-style 3D rendering, "
        "detailed CGI environment, "
        "physically based 3D materials, "
        "dramatic natural lighting, "
        "volumetric atmosphere, "
        "strong depth, "
        "detailed surfaces, "
        "cinematic composition, "
        "realistic 3D geometry, "
        "realistic textures, "
        "dramatic scale, "
        "highly detailed environments, "
        "cinematic depth of field"
    ),

    "default_renderer_feel": (
        "high-end game cinematic, Unreal Engine style"
    ),
}


DEFAULT_VISUAL_RULES = {
    "prioritize_historical_accuracy": True,
    "avoid_anachronisms": True,
    "avoid_generic_african_architecture": True,
    "avoid_modern_objects": True,
    "avoid_unjustified_costumes": True,
    "use_region_specific_architecture": True,
    "use_period_specific_materials": True,
    "use_evidence_based_visual_details": True,
}


# ============================================================
# STYLE PROMPT
# ============================================================

def build_style_prompt(config) -> str:
    """
    Converts config.art_style into a strong SDXL style instruction.

    Qwen remains responsible for WHAT is shown.
    The image engine is responsible for HOW it is rendered.
    """

    art_style = getattr(config, "art_style", None)

    if not isinstance(art_style, dict):
        art_style = DEFAULT_ART_STYLE

    primary = str(
        art_style.get(
            "primary",
            DEFAULT_ART_STYLE["primary"]
        )
    )

    description = str(
        art_style.get(
            "description",
            DEFAULT_ART_STYLE["description"]
        )
    )

    renderer = str(
        art_style.get(
            "default_renderer_feel",
            DEFAULT_ART_STYLE["default_renderer_feel"]
        )
    )

    style = f"""
{primary},
{description},
{renderer},
cinematic 3D CGI,
fully rendered 3D scene,
three-dimensional geometry,
physically based rendering,
PBR materials,
detailed 3D surfaces,
realistic environmental geometry,
AAA video game cinematic,
epic visual storytelling,
cinematic lighting,
volumetric light,
atmospheric perspective,
dramatic depth,
cinematic depth of field,
high detail,
large-scale cinematic composition
"""

    return " ".join(style.split())


# ============================================================
# HISTORICAL VISUAL RULES
# ============================================================

def build_visual_rules_prompt(config) -> str:
    """
    Adds historical accuracy constraints without changing
    Qwen's actual scene content.
    """

    rules = getattr(config, "visual_rules", None)

    if not isinstance(rules, dict):
        rules = DEFAULT_VISUAL_RULES

    parts = []

    if rules.get("prioritize_historical_accuracy"):
        parts.append("historically accurate reconstruction")

    if rules.get("avoid_anachronisms"):
        parts.append("strictly avoid anachronistic objects and technology")

    if rules.get("avoid_generic_african_architecture"):
        parts.append(
            "use specific regionally appropriate architecture rather than generic African architecture"
        )

    if rules.get("avoid_modern_objects"):
        parts.append(
            "no modern objects, vehicles, clothing, buildings, tools, electronics or infrastructure"
        )

    if rules.get("avoid_unjustified_costumes"):
        parts.append(
            "period-appropriate clothing and textiles based on the historical setting"
        )

    if rules.get("use_region_specific_architecture"):
        parts.append(
            "region-specific architecture and construction methods"
        )

    if rules.get("use_period_specific_materials"):
        parts.append(
            "period-specific materials, tools and construction techniques"
        )

    if rules.get("use_evidence_based_visual_details"):
        parts.append(
            "evidence-based historical visual details"
        )

    return ", ".join(parts)


# ============================================================
# NEGATIVE PROMPT
# ============================================================

def build_negative_prompt() -> str:
    """
    Strongly discourages SDXL from interpreting the scene as
    ordinary photography or flat artwork.
    """

    return """
photograph,
photography,
photorealistic photograph,
real photograph,
live action,
documentary photography,
news photograph,
studio photograph,
modern camera aesthetic,
DSLR photo,
portrait photography,
fashion photography,
film still,
cinematic photograph,
real person photograph,
flat illustration,
2D illustration,
digital painting,
painting,
watercolor,
sketch,
drawing,
cartoon,
anime,
manga,
comic,
cel shading,
vector art,
flat colors,
low detail,
low poly,
plastic toy,
doll,
figurine,
modern clothing,
modern architecture,
modern vehicles,
cars,
motorcycles,
smartphones,
laptops,
electric lights,
neon signs,
contemporary objects,
futuristic objects,
science fiction,
text,
letters,
words,
logos,
watermarks
"""

# Clean whitespace.
DEFAULT_NEGATIVE_PROMPT = " ".join(
    build_negative_prompt().split()
)


# ============================================================
# FINAL PROMPT
# ============================================================

def build_image_prompt(
    scene: dict,
    config,
) -> str:
    """
    Combines:

        1. Qwen's scene content
        2. Config art style
        3. Historical accuracy rules
        4. Strong 3D rendering direction

    Qwen controls the historical subject.
    The image engine controls the rendering medium.
    """

    # --------------------------------------------------------
    # Qwen's generated visual prompt
    # --------------------------------------------------------

    qwen_prompt = (
        scene.get("image_prompt")
        or scene.get("visual_description")
        or scene.get("image_description")
        or ""
    )

    qwen_prompt = str(qwen_prompt).strip()

    if not qwen_prompt:
        raise ValueError(
            f"Scene {scene.get('scene_number', scene.get('scene_id', '?'))} "
            "does not contain an image prompt or visual description."
        )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    camera = str(
        scene.get("camera", "")
    ).strip()

    # --------------------------------------------------------
    # Style
    # --------------------------------------------------------

    style_prompt = build_style_prompt(config)

    # --------------------------------------------------------
    # Historical rules
    # --------------------------------------------------------

    historical_prompt = build_visual_rules_prompt(config)

    # --------------------------------------------------------
    # Camera instruction
    # --------------------------------------------------------

    camera_prompt = ""

    if camera:
        camera_prompt = f"""
Camera direction:
{camera}
"""

    # --------------------------------------------------------
    # Final prompt
    # --------------------------------------------------------

    final_prompt = f"""
HISTORICAL SUBJECT AND SCENE:

{qwen_prompt}

VISUAL MEDIUM AND RENDERING:

{style_prompt}

HISTORICAL ACCURACY:

{historical_prompt}

{camera_prompt}

The image must look like a deliberately created
high-end 3D historical reconstruction rather than a photograph.

Render the people, architecture, landscape, clothing,
objects and environment as fully modeled three-dimensional
CGI assets with physically based materials.

Use convincing 3D geometry, detailed surfaces,
natural material response, cinematic volumetric lighting,
atmospheric depth, dramatic scale and strong visual hierarchy.

Preserve the specific historical location, culture,
period, architecture, clothing, tools and objects
described in the scene.

Do not replace historically specific details with
generic African visual stereotypes.

The result should resemble an expensive AAA historical
video game cinematic rendered in Unreal Engine,
with cinematic composition and epic visual storytelling.
"""

    return " ".join(final_prompt.split())


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image(
    pipe,
    prompt: str,
    output_path: Path,
    config,
    seed: Optional[int] = None,
):
    """
    Generate a single vertical SDXL image.
    """

    width = int(
        getattr(config, "image_width", 768)
    )

    height = int(
        getattr(config, "image_height", 1344)
    )

    steps = int(
        getattr(config, "image_steps", 28)
    )

    guidance = float(
        getattr(config, "image_guidance_scale", 7.0)
    )

    # --------------------------------------------------------
    # Generator
    # --------------------------------------------------------

    generator = None

    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed)
        )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    result = pipe(
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )

    image = result.images[0]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(output_path)

    # --------------------------------------------------------
    # Memory cleanup
    # --------------------------------------------------------

    del result
    del image

    _clear_memory()

    return output_path


# ============================================================
# SCENE IMAGE PATH
# ============================================================

def _scene_number(scene: dict, fallback: int) -> int:
    """
    Safely determine scene number.
    """

    value = (
        scene.get("scene_number")
        or scene.get("scene_id")
        or fallback
    )

    try:
        return int(value)
    except Exception:
        return fallback


# ============================================================
# RUN IMAGE PROCESSOR
# ============================================================

def run(
    paths,
    job_id: str,
    scenes: list,
    pipe,
    config,
    progress: Optional[Callable[[int, int], None]] = None,
):
    """
    Generate only missing scene images.

    Existing images are never regenerated.
    """

    total = len(scenes)

    if total == 0:
        raise ValueError(
            f"Job {job_id} contains no scenes."
        )

    generated = 0
    skipped = 0

    print(
        f"[IMAGE] Processing {total} scenes..."
    )

    # --------------------------------------------------------
    # Process each scene
    # --------------------------------------------------------

    for index, scene in enumerate(
        scenes,
        start=1,
    ):

        scene_number = _scene_number(
            scene,
            index,
        )

        output_path = Path(
            paths.image(
                job_id,
                scene_number,
            )
        )

        # ----------------------------------------------------
        # Resume support
        # ----------------------------------------------------

        if output_path.exists() and output_path.stat().st_size > 0:

            print(
                f"[IMAGE] Scene {scene_number}/{total} "
                f"already exists — skipping."
            )

            skipped += 1

            if progress:
                progress(
                    index,
                    total,
                )

            continue

        # ----------------------------------------------------
        # Build prompt
        # ----------------------------------------------------

        prompt = build_image_prompt(
            scene,
            config,
        )

        print(
            f"[IMAGE] Generating scene "
            f"{scene_number}/{total}..."
        )

        # Optional: print prompt for debugging.
        print(
            f"[IMAGE] Prompt preview: "
            f"{prompt[:300]}..."
        )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        generate_image(
            pipe=pipe,
            prompt=prompt,
            output_path=output_path,
            config=config,
        )

        generated += 1

        print(
            f"[IMAGE] Saved: {output_path}"
        )

        if progress:
            progress(
                index,
                total,
            )

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    _clear_memory()

    print(
        f"[IMAGE] Finished | "
        f"generated={generated} | "
        f"skipped={skipped} | "
        f"total={total}"
    )

    return {
        "generated": generated,
        "skipped": skipped,
        "total": total,
    }
