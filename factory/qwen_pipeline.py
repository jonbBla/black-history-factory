from __future__ import annotations
import os, traceback
from factory import research_engine, fact_checker, script_engine, scene_engine, status, topic_engine
from .utils import read_json

def _update_status(paths, job_id, stage, message, state="running"):
    try: status.set_processor(paths,"qwen",state,job_id=job_id,stage=stage,detail=message)
    except Exception as e: print(f"[STATUS] WARNING | {e}")

def _run_or_load(path,label,job_id,fn):
    existing=read_json(path,None)
    if existing is not None:
        print(f"[QWEN] {job_id} | {label} | existing output found -> skipping"); return existing
    return fn()

def run_one(paths,topic,job_id,config,qwen):
    try:
        print(f"[QWEN] {job_id} | START | {topic.title}")
        _update_status(paths,job_id,"research","Source search + Qwen research")
        research=_run_or_load(paths.research(job_id),"RESEARCH",job_id,lambda: research_engine.run(paths=paths,job_id=job_id,topic=topic,config=config,qwen=qwen))
        _update_status(paths,job_id,"fact_check","Verifying research against discovered sources")
        fact_check=_run_or_load(paths.verified(job_id),"FACT_CHECK",job_id,lambda: fact_checker.run(paths=paths,job_id=job_id,topic=topic,research=research,config=config,qwen=qwen))
        _update_status(paths,job_id,"narration","Writing and validating narration")
        narration_path=paths.narration(job_id)
        if os.path.exists(narration_path):
            with open(narration_path,encoding="utf-8") as f: narration=f.read().strip()
            print(f"[QWEN] {job_id} | NARRATION | existing output found -> skipping")
        else:
            narration=script_engine.run(paths=paths,job_id=job_id,topic=topic,research=research,fact_check=fact_check,config=config,qwen=qwen)
        if not narration: raise ValueError("Narration is empty.")
        print(f"[QWEN] {job_id} | NARRATION WORDS: {script_engine.count_words(narration)}")
        _update_status(paths,job_id,"scene_planning","Qwen creating intelligent detailed scenes")
        scenes_path=paths.scenes(job_id)
        if os.path.exists(scenes_path):
            scenes=read_json(scenes_path,None); print(f"[QWEN] {job_id} | SCENES | existing output found -> skipping")
        else:
            scenes=scene_engine.run(paths=paths,job_id=job_id,narration=narration,research=research,fact_check=fact_check,config=config,qwen=qwen)
        if not isinstance(scenes,list): raise ValueError("scenes.json must contain a JSON list of scene objects.")
        min_s,max_s=int(config.scene_count_min),int(config.scene_count_max)
        if not min_s<=len(scenes)<=max_s: raise ValueError(f"Invalid scene count: {len(scenes)}; required {min_s}-{max_s}.")
        _update_status(paths,job_id,"complete",f"Qwen complete | {len(scenes)} scenes",state="idle")
        topic_engine.update_status(paths,job_id,"QWEN_READY",topic_id=topic.id,title=topic.title,scene_count=len(scenes),narration_words=script_engine.count_words(narration))
        topic_engine.mark_used(paths,topic)
        print(f"[QWEN] {job_id} | COMPLETE | {len(scenes)} scenes prepared")
        return True
    except Exception as e:
        print(f"[QWEN] ERROR {job_id} | {e}"); traceback.print_exc(); _update_status(paths,job_id,"error",str(e),state="error")
        try: topic_engine.update_status(paths,job_id,"QWEN_ERROR",error=str(e))
        except Exception: pass
        return False
