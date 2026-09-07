from __future__ import annotations

import gc
import os

import torch


# ---------------------------------------------------------------------
# SDXL + SDXL-Lightning configuration
# ---------------------------------------------------------------------

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
LIGHTNING_CHECKPOINT = "sdxl_lightning_4step_unet.safetensors"


# ---------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------

def _clear_memory():
    """Release temporary Python/CUDA memory."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_sdxl_lightning(model_id=None):
    """
    Load SDXL Base 1.0 with the official SDXL-Lightning 4-step UNet.

    Designed for Google Colab T4:
    - FP16 on CUDA
    - CPU model offloading
    - VAE slicing
    - VAE tiling
    - Lightning 4-step inference
    """

    from diffusers import (
        EulerDiscreteScheduler,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device = "cuda" if use_cuda else "cpu"

    print("[IMAGE] Loading SDXL-Lightning 4-step...")
    print(f"[IMAGE] Device: {device}")
    print(f"[IMAGE] Base model: {BASE_MODEL}")
    print(f"[IMAGE] Lightning checkpoint: {LIGHTNING_CHECKPOINT}")

    _clear_memory()

    # -------------------------------------------------------------
    # 1. Build the SDXL UNet architecture from SDXL Base
    # -------------------------------------------------------------

    print("[IMAGE] Creating UNet architecture...")

    unet = UNet2DConditionModel.from_config(
        BASE_MODEL,
        subfolder="unet",
    )

    # -------------------------------------------------------------
    # 2. Download the official 4-step Lightning UNet
    # -------------------------------------------------------------

    print("[IMAGE] Downloading/loading Lightning checkpoint...")

    checkpoint_path = hf_hub_download(
        repo_id=LIGHTNING_REPO,
        filename=LIGHTNING_CHECKPOINT,
    )

    # Load checkpoint directly to CPU first.
    #
    # This avoids creating an unnecessary second GPU copy while
    # loading the large ~5 GB UNet checkpoint.
    state_dict = load_file(
        checkpoint_path,
        device="cpu",
    )

    print("[IMAGE] Applying Lightning weights...")

    unet.load_state_dict(
        state_dict,
        strict=True,
    )

    # The state dictionary is no longer needed after loading.
    del state_dict

    _clear_memory()

    # Convert the UNet to FP16 before putting it under the pipeline.
    unet = unet.to(dtype=dtype)

    _clear_memory()

    # -------------------------------------------------------------
    # 3. Create the SDXL pipeline
    # -------------------------------------------------------------

    print("[IMAGE] Loading SDXL Base pipeline...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        unet=unet,
        torch_dtype=dtype,
        variant="fp16" if use_cuda else None,
        use_safetensors=True,
        add_watermarker=False,
    )

    # The separate UNet object is now owned by the pipeline.
    del unet

    _clear_memory()

    # -------------------------------------------------------------
    # 4. Lightning requires the trailing Euler scheduler
    # -------------------------------------------------------------

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    # -------------------------------------------------------------
    # 5. T4 memory optimizations
    # -------------------------------------------------------------

    if use_cuda:
        print("[IMAGE] Enabling CPU model offload...")

        # IMPORTANT:
        # Do NOT use pipe.to("cuda") here.
        #
        # CPU offloading keeps only the currently required model
        # components on the T4, greatly reducing VRAM usage.
        pipe.enable_model_cpu_offload()

        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    else:
        pipe.to(device)

    pipe.set_progress_bar_config(disable=True)

    _clear_memory()

    print("[IMAGE] SDXL-Lightning 4-step ready.")

    return pipe


# ---------------------------------------------------------------------
# Prompt handling
# ---------------------------------------------------------------------

def _scene_prompt(scene):
    """
    Select the best visual prompt available in scenes.json.

    Qwen already creates the visual information, so the image
    processor does not create a second prompt-generation stage.
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


# ---------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------

def run(paths, job_id, scenes, pipe, config, progress=None):
    """
    Generate only missing scene images.

    Existing images are never regenerated, allowing safe resume
    after a Colab reset/interruption.
    """

    out = paths.images_dir(job_id)
    os.makedirs(out, exist_ok=True)

    files = []
    total = len(scenes)

    width = int(config.image_width)
    height = int(config.image_height)
    steps = int(config.image_steps)
    guidance = float(config.image_guidance_scale)

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
        # Prompt
        # ---------------------------------------------------------

        prompt = _scene_prompt(scene)

        print(
            f"[IMAGE] {job_id} | "
            f"scene {i}/{total} | generating"
        )

        # ---------------------------------------------------------
        # Deterministic per-scene generator
        #
        # A separate generator prevents the whole job from depending
        # on one global random state.
        # ---------------------------------------------------------

        generator = None

        if torch.cuda.is_available():
            generator = torch.Generator(device="cuda").manual_seed(
                int(sid)
            )
        else:
            generator = torch.Generator().manual_seed(
                int(sid)
            )

        # ---------------------------------------------------------
        # Generate ONE image at a time.
        # ---------------------------------------------------------

        with torch.inference_mode():

            result = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
            )

            image = result.images[0]

            image.save(output_path)

            del result
            del image
            del generator

        _clear_memory()

        files.append(output_path)

        if progress:
            progress(i, total)

    return files
