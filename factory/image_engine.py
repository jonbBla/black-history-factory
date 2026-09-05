from __future__ import annotations
import os

def load_sdxl_lightning(model_id="ByteDance/SDXL-Lightning"):
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    import torch
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    pipe=StableDiffusionXLPipeline.from_pretrained(model_id,torch_dtype=dtype,use_safetensors=True,variant="fp16" if torch.cuda.is_available() else None)
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
    pipe=pipe.to("cuda" if torch.cuda.is_available() else "cpu"); pipe.set_progress_bar_config(disable=True); return pipe

def run(paths,job_id,scenes,pipe,config,progress=None):
    out=paths.images_dir(job_id); os.makedirs(out,exist_ok=True); files=[]; total=len(scenes)
    for i,scene in enumerate(scenes,1):
        sid=int(scene["scene_id"]); p=os.path.join(out,f"scene_{sid:03d}.png")
        if not os.path.exists(p):
            prompt=scene.get("visual_description") or scene.get("image_prompt") or scene.get("image_description")
            if not prompt: raise ValueError(f"Scene {sid} has no visual description.")
            print(f"[IMAGE] {job_id} | scene {i}/{total} | generating from visual_description")
            image=pipe(prompt=prompt,num_inference_steps=int(config.image_steps),guidance_scale=float(config.image_guidance_scale),width=int(config.image_width),height=int(config.image_height)).images[0]; image.save(p)
        else: print(f"[IMAGE] {job_id} | scene {i}/{total} | exists, skip")
        files.append(p)
        if progress: progress(i,total)
    return files
