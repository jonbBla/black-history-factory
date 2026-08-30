import json,os
DEFAULTS={"project_name":"Black History Factory","language":"English","prepared_job_target":40,"target_video_seconds":90,"min_video_seconds":80,"max_video_seconds":100,"narration_words_min":175,"narration_words_max":220,"scene_count_min":20,"scene_count_max":40,"image_model":"sdxl-lightning","qwen_model":"Qwen/Qwen3-4B","qwen_load_in_4bit":True,"image_width":768,"image_height":1344,"image_steps":4,"image_guidance_scale":0.0,"video_width":1080,"video_height":1920,"video_fps":30,"enable_subtitles":True,"enable_music":True,"music_volume":0.10,"source_card_seconds":4.5,"art_style":"epic cinematic historical reconstruction, physically plausible materials, period-authentic details, dramatic natural lighting, volumetric atmosphere, strong depth, detailed surfaces, cinematic composition, realistic proportions","github_repo":"","github_dashboard_path":"dashboard/data"}
class Config:
 def __init__(self,values): self.values=values
 def __getattr__(self,n):
  if n in self.values:return self.values[n]
  raise AttributeError(n)
 @classmethod
 def load(cls,root):
  p=os.path.join(root,'00_CONFIG','config.json'); vals=dict(DEFAULTS)
  if os.path.exists(p):
   with open(p,encoding='utf8') as f: vals.update(json.load(f))
  else:
   os.makedirs(os.path.dirname(p),exist_ok=True); open(p,'w').write(json.dumps(vals,indent=2))
  return cls(vals)
