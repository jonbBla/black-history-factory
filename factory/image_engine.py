# factory/image_engine.py

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable, Optional

import torch
from diffusers import StableDiffusionXLPipeline


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
# LOAD SDXL
# ============================================================

def load_sdxl_lightning(model_id: Optional[str] = None):
    """
    Load SDXL Base.

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
# COMPACT STYLE PROMPT
# ============================================================

STYLE_PROMPT = (
    "cinematic 3D historical reconstruction, "
    "high-end AAA game cinematic, "
    "Unreal Engine style, "
    "detailed CGI, "
    "realistic 3D geometry, "
    "physically based materials, "
    "detailed textures, "
    "cinematic lighting, "
    "volumetric atmosphere, "
    "dramatic depth, "
    "epic composition"
)


# ============================================================
# COMPACT NEGATIVE PROMPT
# ============================================================

NEGATIVE_PROMPT = (
    "photograph, photography, live action, "
    "documentary photo, flat illustration, "
    "2D art, painting, cartoon, anime, manga, "
    "modern objects, modern clothing, "
    "modern architecture, futuristic, "
    "low quality, low detail, low poly, "
    "text, letters, logo, watermark"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def _clean_text(text: str) -> str:
    """
    Normalize text into a compact prompt-friendly string.
    """

    if not text:
        return ""

    text = str(text)

    # Remove excessive whitespace.
    text = " ".join(text.split())

    return text.strip()


# ============================================================
# EXTRACT IMPORTANT SCENE INFORMATION
# ============================================================

def build_subject_prompt(scene: dict) -> str:
    """
    Build the first SDXL prompt.

    Qwen's detailed scene is preserved in scenes.json, but only
    the most visually important portion is sent to CLIP.

    We intentionally keep this compact.
    """

    visual = _clean_text(
        scene.get("image_prompt")
        or scene.get("visual_description")
        or scene.get("image_description")
        or ""
    )

    if not visual:
        raise ValueError(
            f"Scene {scene.get('scene_number', scene.get('scene_id', '?'))} "
            "does not contain an image prompt or visual description."
        )

    # --------------------------------------------------------
    # Basic intelligent compression
    # --------------------------------------------------------
    #
    # Qwen normally puts the most important visual information
    # near the beginning of its description.
    #
    # We keep a bounded amount of text so CLIP does not receive
    # hundreds of tokens.
    #
    # The limit is character-based rather than token-based,
    # giving us a safe margin below CLIP's 77-token limit.
    # --------------------------------------------------------

    words = visual.split()

    # Approximately 45-55 words depending on tokenization.
    words = words[:55]

    compact_visual = " ".join(words)

    return compact_visual


# ============================================================
# CAMERA PROMPT
# ============================================================

def build_camera_prompt(scene: dict) -> str:
    """
    Convert the camera instruction into a compact visual phrase.
    """

    camera = _clean_text(
        scene.get("camera", "")
    )

    if not camera:
        return ""

    # Keep camera information short.
    camera_words = camera.split()[:12]

    return " ".join(camera_words)


# ============================================================
# HISTORICAL STYLE PROMPT
# ============================================================

def build_historical_prompt(config) -> str:
    """
    Compact historical constraints.

    These are deliberately keywords rather than long sentences.
    """

    rules = getattr(config, "visual_rules", None)

    if not isinstance(rules, dict):
        rules = {}

    parts = []

    if rules.get("prioritize_historical_accuracy", True):
        parts.append("historically accurate")

    if rules.get("avoid_anachronisms", True):
        parts.append("period-accurate")

    if rules.get("avoid_generic_african_architecture", True):
        parts.append("region-specific architecture")

    if rules.get("avoid_modern_objects", True):
        parts.append("no modern objects")

    if rules.get("avoid_unjustified_costumes", True):
        parts.append("period-appropriate clothing")

    if rules.get("use_region_specific_architecture", True):
        parts.append("regional construction")

    if rules.get("use_period_specific_materials", True):
        parts.append("period-specific materials")

    if rules.get("use_evidence_based_visual_details", True):
        parts.append("evidence-based details")

    return ", ".join(parts)


# ============================================================
# BUILD SDXL PROMPTS
# ============================================================

def build_image_prompts(scene: dict, config):
    """
    Build the two prompts used by SDXL.

    prompt:
        Historical subject and visual content.

    prompt_2:
        Rendering style and visual medium.

    Both are deliberately compact to avoid the CLIP 77-token
    limitation.
    """

    subject = build_subject_prompt(scene)

    camera = build_camera_prompt(scene)

    historical = build_historical_prompt(config)

    # --------------------------------------------------------
    # PROMPT 1
    # --------------------------------------------------------

    prompt_parts = [
        subject,
    ]

    if historical:
        prompt_parts.append(historical)

    if camera:
        prompt_parts.append(camera)

    prompt = ", ".join(
        part for part in prompt_parts
        if part
    )

    # --------------------------------------------------------
    # PROMPT 2
    # --------------------------------------------------------

    prompt_2 = STYLE_PROMPT

    return prompt, prompt_2


# ============================================================
# GENERATE IMAGE
# ============================================================

def generate_image(
    pipe,
    prompt: str,
    prompt_2: str,
    output_path: Path,
    config,
    seed: Optional[int] = None,
):
    """
    Generate one vertical SDXL image.

    SDXL receives:
        prompt   = scene content
        prompt_2 = visual/rendering style
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
        generator = torch.Generator(
            device="cpu"
        ).manual_seed(
            int(seed)
        )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    result = pipe(
        prompt=prompt,
        prompt_2=prompt_2,
        negative_prompt=NEGATIVE_PROMPT,
        negative_prompt_2=NEGATIVE_PROMPT,
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
    # Cleanup
    # --------------------------------------------------------

    del result
    del image

    _clear_memory()

    return output_path


# ============================================================
# SCENE NUMBER
# ============================================================

def _scene_number(
    scene: dict,
    fallback: int,
) -> int:

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
    progress: Optional[
        Callable[[int, int], None]
    ] = None,
):
    """
    Generate only missing scene images.

    Existing images are skipped.

    This makes the image processor safe to resume even if the
    manifest says IMAGES_READY but files have been deleted.
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
    # Process scenes
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
        # Missing-image detection
        # ----------------------------------------------------

        if (
            output_path.exists()
            and output_path.stat().st_size > 0
        ):

            print(
                f"[IMAGE] Scene "
                f"{scene_number}/{total} "
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
        # Build compact prompts
        # ----------------------------------------------------

        prompt, prompt_2 = build_image_prompts(
            scene,
            config,
        )

        print(
            f"[IMAGE] Generating scene "
            f"{scene_number}/{total}..."
        )

        print(
            f"[IMAGE] Prompt 1: "
            f"{prompt}"
        )

        print(
            f"[IMAGE] Prompt 2: "
            f"{prompt_2}"
        )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        generate_image(
            pipe=pipe,
            prompt=prompt,
            prompt_2=prompt_2,
            output_path=output_path,
            config=config,
        )

        generated += 1

        print(
            f"[IMAGE] Saved: "
            f"{output_path}"
        )

        if progress:
            progress(
                index,
                total,
            )

    # --------------------------------------------------------
    # Cleanup
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
