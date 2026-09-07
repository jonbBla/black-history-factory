from __future__ import annotations

import gc
import os

import torch


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def _clear_memory():
    """Release unused CPU/GPU memory."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_sdxl_lightning(model_id=None):
    """
    Load the stable SDXL Base 1.0 pipeline.

    The function name is kept as load_sdxl_lightning() so the existing
    Image Processor notebook does not need to change.

    Uses the same loading configuration as the proven
    Structured SDXL Image Generator notebook:
      - SDXL Base 1.0
      - FP16
      - safetensors
      - CPU model offloading
      - VAE slicing
      - VAE tiling
    """

    from diffusers import StableDiffusionXLPipeline

    if torch.cuda.is_available():
        dtype = torch.float16
        print(f"[IMAGE] GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"[IMAGE] VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    else:
        dtype = torch.float32
        print("[IMAGE] WARNING: CUDA GPU not detected.")

    _clear_memory()

    print("[IMAGE] Loading Stable Diffusion XL 1.0 Base...")
    print(f"[IMAGE] Model: {MODEL_ID}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if torch.cuda.is_available() else None,
        add_watermarker=False,
    )

    print("[IMAGE] Enabling CPU model offload...")

    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()

        # These significantly reduce VAE memory usage.
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    else:
        pipe.to("cpu")

    pipe.set_progress_bar_config(disable=True)

    _clear_memory()

    print("[IMAGE] Stable Diffusion XL loaded and ready.")

    return pipe


def _scene_prompt(scene):
    """
    Get the image prompt produced by the Qwen scene processor.

    Preference:
      1. image_prompt
      2. visual_description
      3. image_description
    """

    prompt = (
        scene.get("image_prompt")
        or scene.get("visual_description")
        or scene.get("image_description")
    )

    if not prompt:
        sid = scene.get("scene_id", "?")
        raise ValueError(
            f"Scene {sid} has no image_prompt, "
            f"visual_description, or image_description."
        )

    return str(prompt).strip()


def run(paths, job_id, scenes, pipe, config, progress=None):
    """
    Generate missing scene images only.

    Existing images are skipped so the processor can safely resume
    after a Colab reset or interruption.
    """

    out = paths.images_dir(job_id)
    os.makedirs(out, exist_ok=True)

    files = []
    total = len(scenes)

    width = int(config.image_width)
    height = int(config.image_height)

    # Use the existing config values when present.
    # The stable SDXL notebook uses 28 steps and CFG 7.
    steps = int(getattr(config, "image_steps", 28))
    guidance = float(
        getattr(config, "image_guidance_scale", 7.0)
    )

    for i, scene in enumerate(scenes, 1):

        sid = int(scene["scene_id"])

        output_path = os.path.join(
            out,
            f"scene_{sid:03d}.png",
        )

        # ---------------------------------------------------------
        # Resume support
        # ---------------------------------------------------------

        if os.path.exists(output_path):

            print(
                f"[IMAGE] {job_id} | "
                f"scene {i}/{total} | exists, skip"
            )

            files.append(output_path)

            if progress:
                progress(i, total)

            continue

        # ---------------------------------------------------------
        # Get Qwen's visual prompt
        # ---------------------------------------------------------

        prompt = _scene_prompt(scene)

        print(
            f"[IMAGE] {job_id} | "
            f"scene {i}/{total} | generating"
        )

        # CPU generator matches the proven working notebook.
        generator = torch.Generator(
            device="cpu"
        ).manual_seed(sid)

        # ---------------------------------------------------------
        # Generate ONE image at a time
        # ---------------------------------------------------------

        with torch.inference_mode():

            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )

            image = result.images[0]

            image.save(
                output_path,
                format="PNG",
            )

            del image
            del result

        del generator

        _clear_memory()

        files.append(output_path)

        if progress:
            progress(i, total)

    return files
