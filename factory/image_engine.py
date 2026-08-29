import os

def load_sdxl_lightning(model_id='ByteDance/SDXL-Lightning'):
    import torch
    from diffusers import StableDiffusionXLPipeline,LCMScheduler
    pipe=StableDiffusionXLPipeline.from_pretrained(model_id,torch_dtype=torch.float16,variant='fp16')
    pipe.scheduler=LCMScheduler.from_config(pipe.scheduler.config); pipe=pipe.to('cuda' if torch.cuda.is_available() else 'cpu'); pipe.set_progress_bar_config(disable=True); return pipe

def run(paths,job_id,scenes,pipe,config):
    out=paths.images_dir(job_id); os.makedirs(out,exist_ok=True); files=[]
    for s in scenes:
        p=os.path.join(out,f"scene_{int(s['scene_id']):03d}.png")
        if os.path.exists(p): files.append(p); continue
        image=pipe(prompt=s['image_prompt'],num_inference_steps=int(config.image_steps),guidance_scale=float(config.image_guidance_scale),width=int(config.image_width),height=int(config.image_height)).images[0]
        image.save(p); files.append(p)
    return files
