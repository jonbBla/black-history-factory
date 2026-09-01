from . import topic_engine, research_engine, fact_checker
from . import visual_bible, script_engine, scene_engine, status
from .utils import read_json, write_json_atomic, now_iso
import os


def _valid_json_file(path):
    if not os.path.isfile(path):
        return False
    try:
        data = read_json(path, None)
        return isinstance(data, (dict, list)) and bool(data)
    except Exception:
        return False


def run_one(paths, topic, job_id, config, qwen):
    try:
        # ---------------------------------------------------------
        # RESEARCH
        # ---------------------------------------------------------
        if _valid_json_file(paths.research(job_id)):
            research = read_json(paths.research(job_id), {})
            print(f"[QWEN] {job_id} | RESEARCH | existing output found -> skipping")
        else:
            print(f"[QWEN] {job_id} | STAGE: RESEARCH | researching topic")
            topic_engine.update_status(paths, job_id, "QWEN_RESEARCHING")
            research = research_engine.run(paths, job_id, topic, qwen)

        # ---------------------------------------------------------
        # FACT CHECK
        # ---------------------------------------------------------
        if _valid_json_file(paths.verified(job_id)):
            verified = read_json(paths.verified(job_id), {})
            print(f"[QWEN] {job_id} | FACT_CHECK | existing output found -> skipping")
        else:
            print(f"[QWEN] {job_id} | STAGE: FACT_CHECK | reviewing research classifications")
            topic_engine.update_status(paths, job_id, "QWEN_FACT_CHECK")
            verified = fact_checker.run(paths, job_id, qwen)

            if verified.get("verdict") == "REJECT":
                topic_engine.update_status(
                    paths, job_id, "REJECTED",
                    reason=verified.get("issues", [])
                )
                print(f"[QWEN] {job_id} | REJECTED | fact check failed")
                return False

        # ---------------------------------------------------------
        # VISUAL BIBLE
        # ---------------------------------------------------------
        if _valid_json_file(paths.visual_bible(job_id)):
            vb = read_json(paths.visual_bible(job_id), {})
            print(f"[QWEN] {job_id} | VISUAL_BIBLE | existing output found -> skipping")
        else:
            print(f"[QWEN] {job_id} | STAGE: VISUAL_BIBLE | building consistent visual rules")
            topic_engine.update_status(paths, job_id, "QWEN_VISUAL_BIBLE")

            # visual_bible.run() in this repository takes EXACTLY 4 arguments.
            vb = visual_bible.run(topic, research, config, qwen)

            write_json_atomic(
                paths.visual_bible(job_id),
                vb
            )

        # ---------------------------------------------------------
        # NARRATION
        # ---------------------------------------------------------
        if os.path.isfile(paths.narration(job_id)):
            with open(paths.narration(job_id), "r", encoding="utf-8") as f:
                narration = f.read().strip()
            if narration:
                print(f"[QWEN] {job_id} | NARRATION | existing output found -> skipping")
            else:
                narration = None
        else:
            narration = None

        if not narration:
            print(f"[QWEN] {job_id} | STAGE: NARRATION | writing and validating narration")
            topic_engine.update_status(paths, job_id, "QWEN_NARRATION")
            narration = script_engine.run(
                paths, job_id, topic, verified, config, qwen
            )

        if not narration or len(narration.split()) < 100:
            raise ValueError(
                f"Narration is too short: {len(narration.split()) if narration else 0} words"
            )

        # ---------------------------------------------------------
        # SCENE PLANNING
        # ---------------------------------------------------------
        if _valid_json_file(paths.scenes(job_id)):
            scenes = read_json(paths.scenes(job_id), [])
            if isinstance(scenes, dict):
                scenes = scenes.get("scenes", [])
            if not isinstance(scenes, list) or not scenes:
                scenes = None
            else:
                print(f"[QWEN] {job_id} | SCENE_PLANNING | existing output found -> skipping")
        else:
            scenes = None

        if scenes is None:
            print(f"[QWEN] {job_id} | STAGE: SCENE_PLANNING | creating unique visual beats")
            topic_engine.update_status(paths, job_id, "QWEN_SCENE_PLANNING")
            scenes = scene_engine.run(
                paths, job_id, narration, vb, config, qwen
            )

        if not scenes:
            raise ValueError("No scenes generated")

        # ---------------------------------------------------------
        # COMPLETE
        # ---------------------------------------------------------
        topic_engine.update_status(
            paths,
            job_id,
            "QWEN_READY",
            scene_count=len(scenes),
            ready_at=now_iso()
        )

        topic_engine.mark_used(paths, topic)
        status.set_processor(paths, "qwen", "idle", job_id, "ready")

        print(
            f"[QWEN] {job_id} | COMPLETE | {len(scenes)} scenes | QWEN_READY"
        )
        return True

    except Exception as exc:
        topic_engine.update_status(
            paths, job_id, "FAILED", error=str(exc)
        )

        try:
            status.set_processor(
                paths, "qwen", "error", job_id, "failed", str(exc)
            )
        except Exception:
            pass

        print(f"[QWEN] ERROR {job_id} | {exc}")
        return False
