"""Phase E -- real implementation.

Contract (unchanged):
  input:  scenes list (see scene_engine.py) + config (image_width,
          image_height) + a loaded image-generation pipeline (FLUX via
          load_flux(), OR SDXL-Lightning via load_sdxl_lightning() -- pick
          ONE in Cell 4 depending on available VRAM/disk, pass whichever
          you loaded as models["flux"] either way; run() below detects
          which kind it got automatically)
  output: paths.images_dir(job_id)/scene_{NNN}.png -- one file per scene.

Checkpointing: an on_progress(completed_count) callback is invoked after
EVERY image (not just per-job) so main.py can advance+push the checkpoint
per image -- image generation is the slowest, most interruption-prone
stage, per the spec. Any scene whose PNG already exists on Drive is
skipped without calling the model again, so resuming after a crash never
regenerates finished images.
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
    FLUX.1 Schnell is a distilled model designed for 1-4 inference steps --
    that's what makes it viable to run per-scene inside a single Colab
    session instead of needing 20-50 steps like most diffusion models.

    VRAM note, corrected: the 12B-parameter transformer ALONE is ~24GB in
    bf16 -- already more than a free-tier T4's ~14.56GB by itself, before
    the ~9.4GB T5 text encoder, CLIP, or VAE are even considered (full
    pipeline download is ~34GB total, matching what Colab actually
    downloads). enable_model_cpu_offload() (low_vram, still applied below)
    only helps by keeping OTHER components off the GPU while one is
    active -- it can't shrink the transformer itself, so on a T4 it isn't
    enough on its own and the job will hit CUDA OOM the moment the
    transformer's turn comes up, exactly as it would fail identically
    whether Qwen is also loaded or not.

    load_in_4bit=True (the default) fixes the actual cause: it loads the
    transformer AND the T5 text encoder with bitsandbytes NF4 quantization
    (the same technique qwen_client.py uses for Qwen), cutting the
    transformer to roughly 6GB and the T5 encoder to roughly 2.5GB --
    small enough to coexist with 4-bit Qwen on a single T4. CLIP and the
    VAE are left at full precision since they're small enough not to
    matter. Set load_in_4bit=False only on a GPU with enough VRAM for the
    full-precision pipeline (A100 40GB or similar).
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
    """Alternative to load_flux() for tight-VRAM sessions -- call at most
    ONE of load_flux()/load_sdxl_lightning() in Cell 4, whichever fits your
    Colab session, and pass its result as models["flux"] either way (the
    image_engine.run() dispatcher below detects which kind of pipeline it
    got automatically).

    SDXL-Lightning is a distilled Stable Diffusion XL variant supporting
    1-8 step generation -- same speed idea as FLUX.1-schnell, but built on
    SDXL's ~2.6B-parameter UNet rather than FLUX's 12B transformer. Full
    pipeline is ~7GB in fp16 -- comfortably fits a T4 alongside Qwen with
    NO quantization tricks needed, and downloads roughly 5x less data than
    FLUX (~7GB vs ~34GB), directly helping the disk-space problem too.

    Tradeoffs vs FLUX, worth knowing before you pick:
    - Prompt length: SDXL uses CLIP-only text encoding with a hard 77-token
      limit; FLUX additionally uses a T5 encoder handling up to 256 tokens.
      Long, detail-heavy scene prompts (like this project's, which combine
      the visual bible + scene description) truncate more aggressively on
      SDXL. scene_engine.py's prompt composition was tightened alongside
      this to help, but SDXL will still drop more detail on long prompts.
    - Photorealism and complex prompt adherence: FLUX is generally regarded
      as stronger.
    - Painterly/stylized output: SDXL has a much larger ecosystem built
      around exactly this kind of non-photoreal, artistic style (which is
      what this project's locked art_style calls for), so quality for THIS
      specific use case may be comparable to or better than FLUX despite
      SDXL being the "smaller" model.
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
    pipe.unet.load_state_dict(
        load_file(hf_hub_download(lightning_repo, ckpt_name), device=device)
    )
    # Lightning's distillation requires this specific scheduler config --
    # using the base SDXL scheduler unmodified produces poor results.
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe._num_inference_steps = num_inference_steps  # stashed for _generate_one below
    return pipe


def _generate_one(pipe, prompt: str, width: int, height: int):
    if hasattr(pipe, "transformer"):
        # FLUX-style pipeline. Schnell is trained for guidance_scale=0 and
        # very few steps; max_sequence_length=256 uses the T5 encoder's
        # full available context rather than the shorter CLIP-only default.
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
    width = getattr(config, "image_width", 1024) if config else 1024
    height = getattr(config, "image_height", 1024) if config else 1024

    written = []
    for scene in scenes:
        fname = os.path.join(out_dir, f"scene_{scene['scene_id']:03d}.png")
        if os.path.exists(fname):
            written.append(fname)
            if on_progress:
                on_progress(len(written))
            continue

        if flux is None:
            # No model loaded -- write a placeholder so downstream stages
            # (audio duration matching, video assembly) can still run.
            with open(fname, "wb") as f:
                f.write(_PLACEHOLDER_PNG)
        else:
            prompt = scene.get("image_prompt", "")
            try:
                image = _generate_one(flux, prompt, width, height)
                image.save(fname)
            except Exception as e:
                raise RuntimeError(
                    f"Image generation failed for {job_id} scene {scene['scene_id']}: {e}"
                ) from e

        written.append(fname)
        if on_progress:
            on_progress(len(written))

    return written
  
