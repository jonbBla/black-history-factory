"""Transactional orchestration for the Qwen processor."""
from .utils import read_json, write_json_atomic, now_iso
from . import topic_engine, research_engine, fact_checker, script_engine, scene_engine

def _has_file(path):
    import os
    return os.path.isfile(path) and os.path.getsize(path) > 0

def run_one(paths, topic, job_id, config, qwen, visual_bible_runner, status):
    try:
        status.set_processor(paths,"qwen","RESEARCHING",job_id)
        research = read_json(paths.research(job_id),{}) if _has_file(paths.research(job_id)) else research_engine.run(paths,job_id,topic,qwen)

        status.set_processor(paths,"qwen","FACT_CHECKING",job_id)
        verified = fact_checker.run(paths,job_id,research,qwen)

        status.set_processor(paths,"qwen","VISUAL_BIBLE",job_id)
        vb = visual_bible_runner.run(paths,job_id,topic,verified,config,qwen)

        status.set_processor(paths,"qwen","SCRIPTING",job_id)
        narration = script_engine.run(paths,job_id,topic,verified,config,qwen)

        status.set_processor(paths,"qwen","SCENE_PLANNING",job_id)
        scenes = scene_engine.run(paths,job_id,narration,vb,config,qwen)

        if not scenes: raise ValueError("No scenes generated")

        write_json_atomic(paths.state(job_id,"qwen"),{
            "status":"QWEN_READY","processor":"qwen","job_id":job_id,"updated_at":now_iso()
        })
        topic_engine.mark_used(paths,topic)
        status.set_processor(paths,"qwen","QWEN_READY",job_id,f"{len(scenes)} scenes")
        return True

    except Exception as exc:
        write_json_atomic(paths.state(job_id,"qwen"),{
            "status":"FAILED","processor":"qwen","job_id":job_id,
            "error":str(exc),"updated_at":now_iso()
        })
        status.set_processor(paths,"qwen","FAILED",job_id,str(exc))
        raise
