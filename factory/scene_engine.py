import json
from .utils import write_json_atomic
CAMERAS=['zoom_in','zoom_out','pan_left','pan_right','slow_push','slow_pull']
def run(paths,job_id,narration,vb,config,qwen):
    prompt=f'''Break this narration into {config.scene_count_min}-{config.scene_count_max} visual scenes for a fast-paced 90-second vertical short. Each scene must align with its exact narration segment. Return JSON object {{"scenes":[...]}}. Fields: scene_id, narration, visual_focus, camera, transition, historical_role. Prefer 2-4 second visual beats; use hard cuts frequently and crossfade only for changes of place/time/mood. Camera choices: {CAMERAS}. Never invent unsupported historical details. Narration: {narration}\nVisual bible: {json.dumps(vb,ensure_ascii=False)}'''
    raw=qwen.generate_json(prompt,max_new_tokens=5000); scenes=raw.get('scenes',raw) if isinstance(raw,dict) else raw
    if not isinstance(scenes,list) or not scenes: raise ValueError('Scene planner did not return scenes')
    out=[]
    for i,s in enumerate(scenes,1):
        focus=str(s.get('visual_focus','historical scene')).strip(); image_prompt=f"{focus}, {vb.get('period','')}, {vb.get('region','')}, {vb.get('style','')}, {vb.get('lighting','dramatic natural light')}"
        out.append({'scene_id':i,'narration':str(s.get('narration','')).strip(),'visual_focus':focus,'camera':s.get('camera','slow_push') if s.get('camera') in CAMERAS else 'slow_push','transition':s.get('transition','hard_cut'),'historical_role':s.get('historical_role',''),'image_prompt':image_prompt})
    write_json_atomic(paths.scenes(job_id),out); return out
