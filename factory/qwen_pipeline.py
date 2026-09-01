from . import topic_engine, research_engine, fact_checker
from . import visual_bible, script_engine, scene_engine, status
from .utils import read_json, now_iso


def run_one(paths, topic, job_id, config, qwen):

    try:
        # ---------------------------------------------------------
        # RESEARCH
        # ---------------------------------------------------------
        print(
            f"[QWEN] {job_id} | STAGE: RESEARCH | researching topic"
        )

        topic_engine.update_status(
            paths,
            job_id,
            "QWEN_RESEARCHING"
        )

        research = research_engine.run(
            paths,
            job_id,
            topic,
            qwen
        )

        # ---------------------------------------------------------
        # FACT CHECK
        # ---------------------------------------------------------
        print(
            f"[QWEN] {job_id} | STAGE: FACT_CHECK | "
            f"reviewing research classifications"
        )

        topic_engine.update_status(
            paths,
            job_id,
            "QWEN_FACT_CHECK"
        )

        verified = fact_checker.run(
            paths,
            job_id,
            qwen
        )

        if verified.get("verdict") == "REJECT":

            topic_engine.update_status(
                paths,
                job_id,
                "REJECTED",
                reason=verified.get("issues", [])
            )

            print(
                f"[QWEN] {job_id} | REJECTED | "
                f"fact check failed"
            )

            return False

        # ---------------------------------------------------------
        # VISUAL BIBLE
        # ---------------------------------------------------------
        print(
            f"[QWEN] {job_id} | STAGE: VISUAL_BIBLE | "
            f"building consistent visual rules"
        )

        topic_engine.update_status(
            paths,
            job_id,
            "QWEN_VISUAL_BIBLE"
        )

        vb = visual_bible.run(
            paths,
            job_id,
            topic,
            research,
            config,
            qwen
        )

        # ---------------------------------------------------------
        # NARRATION
        # ---------------------------------------------------------
        print(
            f"[QWEN] {job_id} | STAGE: NARRATION | "
            f"writing and validating narration"
        )

        topic_engine.update_status(
            paths,
            job_id,
            "QWEN_NARRATION"
        )

        narration = script_engine.run(
            paths,
            job_id,
            topic,
            verified,
            config,
            qwen
        )

        if not narration or len(narration.split()) < 100:
            raise ValueError(
                f"Narration is too short: "
                f"{len(narration.split()) if narration else 0} words"
            )

        # ---------------------------------------------------------
        # SCENE PLANNING
        # ---------------------------------------------------------
        print(
            f"[QWEN] {job_id} | STAGE: SCENE_PLANNING | "
            f"creating unique visual beats"
        )

        topic_engine.update_status(
            paths,
            job_id,
            "QWEN_SCENE_PLANNING"
        )

        scenes = scene_engine.run(
            paths,
            job_id,
            narration,
            vb,
            config,
            qwen
        )

        if not scenes:
            raise ValueError(
                "No scenes generated"
            )

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

        topic_engine.mark_used(
            paths,
            topic
        )

        status.set_processor(
            paths,
            "qwen",
            "idle",
            job_id,
            "ready"
        )

        print(
            f"[QWEN] {job_id} | COMPLETE | "
            f"{len(scenes)} scenes | QWEN_READY"
        )

        return True

    except Exception as exc:

        topic_engine.update_status(
            paths,
            job_id,
            "FAILED",
            error=str(exc)
        )

        try:
            status.set_processor(
                paths,
                "qwen",
                "error",
                job_id,
                "failed",
                str(exc)
            )
        except Exception:
            pass

        print(
            f"[QWEN] ERROR {job_id} | {exc}"
        )

        return False
