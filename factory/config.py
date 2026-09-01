from __future__ import annotations
import json, os
DEFAULTS={"project_name":"Black History Factory","language":"English","prepared_job_target":1,"target_video_seconds":90,"min_video_seconds":80,"max_video_seconds":100,"narration_words_min":175,"narration_words_max":220,"scene_count_min":20,"scene_count_max":24,"max_scene_words":14,"image_model":"ByteDance/SDXL-Lightning","image_width":768,"image_height":1344,"image_steps":4,"image_guidance_scale":0.0,"video_width":1080,"video_height":1920,"video_fps":30,"enable_subtitles":True,"enable_music":True,"music_volume":0.1,"enable_ambience":True,"enable_sfx":True,"ambience_volume":0.08,"sfx_volume":0.15,"source_card_enabled":True,"source_card_seconds":4.5,"art_style":{"primary":"cinematic 3D historical reconstruction","description":"epic cinematic historical reconstruction, physically plausible materials, period-authentic details, dramatic natural lighting, volumetric atmosphere, strong depth, detailed surfaces, cinematic composition, realistic proportions, highly detailed environments, realistic textures, dramatic scale, not flat cartoon","default_renderer_feel":"high-end game cinematic, Unreal Engine style"},"image_mode":"generate_missing_only","github_repo":"","github_dashboard_path":"dashboard/data"}
class Config:
    def __init__(self,values): self.values=values
    def __getattr__(self,n):
        if n in self.values:return self.values[n]
        raise AttributeError(n)
    @property
    def art_style_text(self):
        v=self.values.get('art_style',''); return (v.get('description') or v.get('primary')) if isinstance(v,dict) else str(v)
    @classmethod
    def load(cls,root):
        p=os.path.join(root,'00_CONFIG','config.json'); vals=dict(DEFAULTS)
        if os.path.exists(p): vals.update(json.load(open(p,encoding='utf8')))
        else:
            os.makedirs(os.path.dirname(p),exist_ok=True); json.dump(vals,open(p,'w',encoding='utf8'),indent=2)
        return cls(vals)
