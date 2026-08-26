"""Real image-generation implementation.

Contract:
  input:  scenes list (see scene_engine.py) + config (image_width,
          image_height) + a loaded image-generation pipeline (FLUX via
          load_flux(), SDXL-Lightning via load_sdxl_lightning(), OR
          SD-Turbo via load_sd_turbo() -- pick ONE in Cell 4 depending on
          available VRAM/disk, pass whichever you loaded as
          models["flux"] either way; run() below detects which kind it
          got automatically)
  output: paths.images_dir(job_id)/scene_{NNN}.png -- one file per scene.

Checkpointing: an on_progress(completed_count) callback is invoked after
EVERY image (not just per-job) so main.py can advance+push the checkpoint
per image. Any scene whose PNG already exists on Drive is skipped without
calling the model again, so resuming after a crash never regenerates
finished images.
"""

from __future__ import annotations
import os
import base64

_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42"
    "YAAAAASUVORK5CYII="
)


def load_flux(device: str = "cuda", low_vram: bool = True, load_in_4bit: bool = True):
    """Call once in Colab Cell 4, e.g.:
        from factory.image_engine import load_flux
        models["flux"] = load_flux()
    FLUX.1 Schnell is a distilled model designed for 1-4 inference steps.

    VRAM note: the 12B-parameter transformer ALONE is ~24GB in bf16 --
    already more than a free-tier T4's ~14.56GB by itself, before the
    ~9.4GB T5 text encoder, CLIP, or VAE are even considered (full
    pipeline download is ~34GB total). enable_model_cpu_offload() only
    helps by keeping OTHER components off the GPU while one is active --
    it can't shrink the transformer itself, so on a T4 it isn't enough on
    its own.

    load_in_4bit=True (the default) fixes the actual cause: it loads the
    transformer AND the T5 text encoder with bitsandbytes NF4
    quantization, cutting the transformer to roughly 6GB and the T5
    encoder to roughly 2.5GB. Set load_in_4bit=False only on a GPU with
    enough VRAM for the full-precision pipeline (A100 40GB or similar).

    Given the disk (~34GB download) and VRAM cost even quantized, consider
    load_sdxl_lightning() instead unless you specifically need FLUX's
    stronger prompt adherence.
    """
    from diffusers import FluxPipeline
    import torch

    if load_in_4bit and device == "cuda":
        from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
        from diffusers import FluxTransformer2DModel
        from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
        from transformers import T5EncoderModel

        model_id = "black-forest-labs/FLUX.1-schnell"

        transformer_4bit_config = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer",
            quantization_config=transformer_4bit_config,
            torch_dtype=torch.bfloat16,
        )

        text_encoder_4bit_config = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder_2",
            quantization_config=text_encoder_4bit_config,
            torch_dtype=torch.bfloat16,
        )

        pipe = FluxPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16,
        )
        # Components are already quantized down to a few GB each and load
        # directly onto the GPU -- no need to also layer
        # enable_model_cpu_offload() on top, which has known rough edges
        # when mixed with bitsandbytes-quantized submodules in diffusers.
        pipe.to(device)
    else:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        )
        if low_vram:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
    return pipe


def load_sdxl_lightning(device: str = "cuda", num_inference_steps: int = 4):
    """Alternative to load_flux() -- call at most ONE of
    load_flux()/load_sdxl_lightning() in Cell 4, whichever fits your Colab
    session, and pass its result as models["flux"] either way (the run()
    dispatcher below detects which kind of pipeline it got automatically).

    Recommended default: SDXL-Lightning is a distilled Stable Diffusion XL
    variant supporting 1-8 step generation, built on SDXL's ~2.6B-parameter
    UNet rather than FLUX's 12B transformer. Full pipeline is ~7GB in
    fp16 -- comfortably fits a T4 alongside Qwen with NO quantization
    tricks needed, and downloads roughly 5x less data than FLUX.

    Tradeoffs vs FLUX:
    - Prompt length: SDXL uses CLIP-only text encoding with a hard
      77-token limit; FLUX additionally uses a T5 encoder handling up to
      256 tokens. scene_engine.py's prompt composition is tuned for this
      tighter SDXL budget by default.
    - Photorealism and complex prompt adherence: FLUX is generally
      regarded as stronger.
    - Painterly/stylized output: SDXL has a much larger ecosystem built
      around exactly this kind of non-photoreal, artistic style (which is
      what this project's locked art_style calls for).
    """
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import torch

    if num_inference_steps not in (1, 2, 4, 8):
        raise ValueError("SDXL-Lightning checkpoints are published for 1, 2, 4, or 8 steps.")

    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    lightning_repo = "ByteDance/SDXL-Lightning"
    ckpt_name = f"sdxl_lightning_{num_inference_steps}step_unet.safetensors"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, variant="fp16"
    ).to(device)
    # Load the Lightning UNet weights to CPU first, then let load_state_dict
    # copy each tensor into the already-allocated (already-on-GPU) UNet
    # parameters in place. Loading directly to GPU here instead would
    # briefly hold BOTH the original UNet (already resident from .to(device)
    # above) and this new state dict as a full second GPU-resident copy at
    # the same time -- exactly the kind of transient memory spike that OOMs
    # a 14.56GB T4 regardless of how small either model is on its own.
    lightning_state_dict = load_file(hf_hub_download(lightning_repo, ckpt_name), device="cpu")
    pipe.unet.load_state_dict(lightning_state_dict)
    del lightning_state_dict
    if device == "cuda":
        torch.cuda.empty_cache()
    # Lightning's distillation requires this specific scheduler config --
    # using the base SDXL scheduler unmodified produces poor results.
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe._num_inference_steps = num_inference_steps  # stashed for _generate_one below
    return pipe


def load_sd_turbo(device: str = "cuda", num_inference_steps: int = 4):
    """Alternative to load_flux()/load_sdxl_lightning() for the tightest
    possible resource budget -- call at most ONE image-loader function in
    Cell 4, whichever you pick lands in models["flux"], and run() below
    detects which kind it got automatically (no special-casing needed:
    SD-Turbo's pipeline, like SDXL-Lightning's, has no `.transformer`
    attribute, so it falls into the same dispatch branch in
    _generate_one() below).

    SD-Turbo is built on Stable Diffusion 2.1's architecture (a single
    ~865M-parameter UNet and a single CLIP text encoder) rather than
    SDXL's 2.6B-parameter UNet with dual text encoders. Total pipeline
    download is roughly 2.5-3GB -- smaller than SDXL-Lightning's ~7GB,
    dramatically smaller than FLUX's ~34GB. This is the lightest of the
    three image backends on storage, VRAM, and generation speed (1-2
    steps is enough).

    Real quality cost: SD-Turbo's native/best resolution is 512x512.
    Generating at this project's configured portrait resolution (e.g.
    896x1600) works, but with more visible quality degradation than
    either FLUX or SDXL-Lightning would show at the same resolution.
    Use this when disk/VRAM/RAM headroom matters more than image
    fidelity.
    """
    from diffusers import AutoPipelineForText2Image
    import torch

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to(device)
    pipe._num_inference_steps = num_inference_steps  # stashed for _generate_one below
    pipe._native_resolution = 512  # stashed for _dims_for_generation below
    return pipe


def _dims_for_generation(pipe, config_width: int, config_height: int):
    """Picks the actual width/height to hand to the model, which is NOT
    always this project's configured portrait canvas size.

    SD-Turbo is trained/optimized around 512x512. Forcing it to generate
    directly at a tall portrait target (e.g. 896x1600, aspect ratio ~1.79)
    causes a well-documented diffusion artifact: the model doesn't know
    how to compose the full unfamiliar canvas, so it repeats a small
    learned pattern block to fill it -- visible as a repetitive tiled
    look (identical windows/balconies/lamps stacked over and over) rather
    than a naturally varied single scene.

    The fix generates at a resolution that keeps the SAME target aspect
    ratio (so video_engine.py's existing scale+crop step doesn't have to
    throw away an unreasonable fraction of the image) while keeping the
    total pixel count close to the model's comfortable ~512x512 budget --
    both matter: matching only total area (e.g. plain 512x512) would
    force a much more aggressive crop later, and matching only aspect
    ratio at full size is exactly the problem being fixed.

    FLUX and SDXL-Lightning don't get this treatment: FLUX handles
    arbitrary resolutions reasonably, and SDXL was trained across multiple
    aspect-ratio buckets including portrait, so config's actual target
    resolution is fine for both.
    """
    native = getattr(pipe, "_native_resolution", None)
    if not native or config_width <= 0 or config_height <= 0:
        return config_width, config_height

    aspect = config_height / config_width
    gen_width = int(round((native * native / aspect) ** 0.5))
    gen_height = int(round(gen_width * aspect))
    # Diffusion UNets require dimensions divisible by 8 (latent downsampling).
    gen_width = max(8, (gen_width // 8) * 8)
    gen_height = max(8, (gen_height // 8) * 8)
    return gen_width, gen_height


def _generate_one(pipe, prompt: str, width: int, height: int):
    if hasattr(pipe, "transformer"):
        # FLUX-style pipeline. Schnell is trained for guidance_scale=0 and
        # very few steps; max_sequence_length=256 uses the T5 encoder's
        # full available context.
        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
        )
    else:
        # SDXL-Lightning-style pipeline. Also distilled for guidance_scale=0
        # and few steps, but has no max_sequence_length knob -- SDXL's CLIP
        # text encoders truncate at 77 tokens regardless.
        steps = getattr(pipe, "_num_inference_steps", 4)
        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=0.0,
            num_inference_steps=steps,
        )
    return result.images[0]


def run(paths, job_id: str, scenes: list, config=None, flux=None, on_progress=None) -> list:
    out_dir = paths.images_dir(job_id)
    os.makedirs(out_dir, exist_ok=True)
    width = getattr(config, "image_width", 896) if config else 896
    height = getattr(config, "image_height", 1600) if config else 1600

    written = []
    for scene in scenes:
        fname = os.path.join(out_dir, f"scene_{scene['scene_id']:03d}.png")
        if os.path.exists(fname):
            written.append(fname)
            if on_progress:
                on_progress(len(written))
            continue

        if flux is None:
            with open(fname, "wb") as f:
                f.write(_PLACEHOLDER_PNG)
        else:
            prompt = scene.get("image_prompt", "")
            gen_width, gen_height = _dims_for_generation(flux, width, height)
            try:
                image = _generate_one(flux, prompt, gen_width, gen_height)
                image.save(fname)
            except Exception as e:
                raise RuntimeError(
                    f"Image generation failed for {job_id} scene {scene['scene_id']}: {e}"
                ) from e

        written.append(fname)
        if on_progress:
            on_progress(len(written))

    return written
