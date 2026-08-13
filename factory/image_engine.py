"""Phase E -- real implementation.

Contract (unchanged):
  input:  scenes list (see scene_engine.py) + config (image_width,
          image_height) + a loaded FLUX pipeline
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


def load_flux(device: str = "cuda"):
    """Call once in Colab Cell 4, e.g.:
        from factory.image_engine import load_flux
        models["flux"] = load_flux()
    FLUX.1 Schnell is a distilled model designed for 1-4 inference steps --
    that's what makes it viable to run per-scene inside a single Colab
    session instead of needing 20-50 steps like most diffusion models.
    """
    from diffusers import FluxPipeline
    import torch
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
    )
    pipe.to(device)
    return pipe


def _generate_one(flux, prompt: str, width: int, height: int):
    # Schnell is trained for guidance_scale=0 and very few steps.
    result = flux(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
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
