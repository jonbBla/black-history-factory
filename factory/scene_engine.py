from __future__ import annotations
import json,os,re
from .utils import write_json_atomic

def count_words(text): return len(re.findall(r"\b[\w’'-]+\b",text or ""))
def normalize(text): return re.findall(r"\b[\w’'-]+\b",text or "")
def _prompt(narration,research,fact_check,config):
    style=getattr(config,"art_style_text","cinematic 3D historical reconstruction"); min_s=int(config.scene_count_min); max_s=int(config.scene_count_max)
    return f'''You are the cinematic scene director for a historical documentary.

NARRATION (must be preserved exactly across scenes):
{narration}

RESEARCH:
{json.dumps(research,ensure_ascii=False,indent=2)}

FACT CHECK:
{json.dumps(fact_check,ensure_ascii=False,indent=2)}

Create {min_s}-{max_s} intelligent visual scenes. Decide boundaries from meaning, natural speech, action, location, time, reveals and important details. Do NOT split the narration mechanically into equal chunks.

Every word of the narration must appear exactly once, in original order. Do not rewrite, add, remove, or repeat narration words.

Each scene must have scene_id, narration, visual_description, and camera. camera must be one of slow_push, zoom_in, zoom_out, pan_left, pan_right, slow_pull.

Make visual_description detailed enough to send directly to an image generator. When relevant specify main subject/action, historically appropriate people and attire, textiles, architecture, objects/tools, materials, environment, period/region, composition, camera perspective, depth, scale, lighting and atmosphere. Use only details supported by the research/fact check. Never use generic African stereotypes, modern objects, invented costumes, text, logos or watermarks.

ART STYLE: {style}

Return ONLY valid JSON with root shape:
{{"scenes":[{{"scene_id":1,"narration":"...","visual_description":"...","camera":"slow_push"}}]}}
Do not return a scene_count wrapper.'''
def _extract_json(text):
    text=(text or "").strip(); text=re.sub(r"^```(?:json)?\s*","",text,flags=re.I); text=re.sub(r"\s*```$","",text)
    try:return json.loads(text)
    except Exception:pass
    start=text.find('{')
    if start<0: raise ValueError("No JSON object found.")
    depth=0; ins=False; esc=False
    for i in range(start,len(text)):
        c=text[i]
        if ins:
            if esc: esc=False
            elif c=='\\': esc=True
            elif c=='"': ins=False
        else:
            if c=='"': ins=True
            elif c=='{': depth+=1
            elif c=='}':
                depth-=1
                if depth==0:return json.loads(text[start:i+1])
    raise ValueError("Incomplete JSON object.")
def _validate(data,narration,min_s,max_s):
    if not isinstance(data,dict) or not isinstance(data.get('scenes'),list): raise ValueError("Missing scenes list.")
    scenes=data['scenes']
    if not min_s<=len(scenes)<=max_s: raise ValueError(f"Scene count {len(scenes)} outside {min_s}-{max_s}.")
    combined=[]
    for i,s in enumerate(scenes,1):
        if not isinstance(s,dict) or int(s.get('scene_id',-1))!=i: raise ValueError(f"Scene {i} invalid or wrong scene_id.")
        for k in ('narration','visual_description','camera'):
            if not str(s.get(k,'')).strip(): raise ValueError(f"Scene {i} missing {k}.")
        if s['camera'] not in {'slow_push','zoom_in','zoom_out','pan_left','pan_right','slow_pull'}: raise ValueError(f"Scene {i} invalid camera.")
        combined.extend(normalize(s['narration']))
    if [x.lower() for x in combined]!=[x.lower() for x in normalize(narration)]: raise ValueError("Scene narration does not exactly preserve the original narration order.")
    return scenes
def run(paths,job_id,narration,research,fact_check,config,qwen):
    min_s,max_s=int(config.scene_count_min),int(config.scene_count_max); os.makedirs(os.path.dirname(paths.scenes(job_id)),exist_ok=True)
    for attempt in range(1,5):
        print(f"[SCENES] ATTEMPT {attempt}/4 | Qwen scene director")
        try:
            data=qwen.generate_json(_prompt(narration,research,fact_check,config),max_new_tokens=5000,retries=1); scenes=_validate(data,narration,min_s,max_s); final=[]
            for s in scenes:
                sid=int(s['scene_id']); visual=str(s['visual_description']).strip()
                final.append({'scene_id':sid,'scene_number':sid,'narration':str(s['narration']).strip(),'word_count':count_words(s['narration']),'visual_description':visual,'image_prompt':visual,'image_description':visual,'camera':str(s['camera']).strip()})
            write_json_atomic(paths.scenes(job_id),final); print(f"[SCENES] COMPLETE | {len(final)} scenes"); return final
        except Exception as e: print(f"[SCENES] FAILED | {e}")
    raise ValueError("Scene generation failed after 4 attempts.")
