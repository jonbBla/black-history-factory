from __future__ import annotations
import os
from . import topic_engine, research_engine, fact_checker, visual_bible, script_engine, scene_engine, status
from .utils import read_json, write_json_atomic, now_iso

def _exists(path): return os.path.isfile(path) and os.path.getsize(path) > 0

def _stage(paths, job_id, stage, detail):
    status.set_processor(paths, "qwen", "running", job_id, stage, detail)
    print(f"[QWEN] {job_id} | STAGE: {stage.upper()} | {detail}")

def run_one(paths, topic, job_id, config, qwen):
    try:
        _stage(paths, job_id, "research", "researching topic")
        research = read_json(paths.research(job_id), {}) if _exists(paths.research(job_id)) else research_engine.run(paths, job_id, topic, qwen)
        _stage(paths, job_id, "fact_check", "reviewing research classifications")
        verified = read_json(paths.verified(job_id), {}) if _exists(paths.verified(job_id)) else fact_checker.run(paths, job_id, qwen)
        _stage(paths, job_id, "visual_bible", "building consistent visual rules")
        if _exists(paths.visual_bible(job_id)):
            vb = read_json(paths.visual_bible(job_id), {})
        else:
            vb = visual_bible.run(topic, verified, config, qwen); write_json_atomic(paths.visual_bible(job_id), vb)
        _stage(paths, job_id, "narration", "writing and validating narration")
        narration = open(paths.narration(job_id), encoding="utf8").read() if _exists(paths.narration(job_id)) else script_engine.run(paths, job_id, topic, verified, config, qwen)
        _stage(paths, job_id, "scene_planning", "creating unique visual beats")
        scenes = read_json(paths.scenes(job_id), []) if _exists(paths.scenes(job_id)) else scene_engine.run(paths, job_id, narration, vb, config, qwen)
        if not scenes: raise ValueError("No scenes generated")
        topic_engine.update_status(paths, job_id, "QWEN_READY", scene_count=len(scenes), ready_at=now_iso())
        topic_engine.mark_used(paths, topic)
        status.set_processor(paths, "qwen", "idle", job_id, "ready", f"{len(scenes)} scenes")
        print(f"[QWEN] COMPLETE {job_id} | {topic.title} | {len(scenes)} scenes")
        return True
    except Exception as exc:
        topic_engine.update_status(paths, job_id, "FAILED", error=str(exc))
        status.set_processor(paths, "qwen", "error", job_id, "failed", str(exc))
        print(f"[QWEN] ERROR {job_id} | {exc}")
        return False
