import os
import json
import traceback

from factory import (
    research_engine,
    fact_checker,
    visual_bible,
    script_engine,
    scene_engine,
    status,
)


# ============================================================
# HELPERS
# ============================================================

def _read_json(path):
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp = path + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(tmp, path)


def _update_status(paths, job_id, stage, message):
    """
    Status updates are best-effort so a dashboard problem
    does not kill the actual processing job.
    """

    try:
        status.update_status(
            paths,
            job_id,
            stage,
            message,
        )
    except Exception as e:
        print(
            f"[STATUS] WARNING | "
            f"Could not update status: {e}"
        )


# ============================================================
# MAIN QWEN PIPELINE
# ============================================================

def run_one(
    paths,
    topic,
    job_id,
    config,
    qwen,
):
    """
    Process one topic through the complete Qwen pipeline.

    PIPELINE:

        TOPIC
          ↓
        SOURCE SEARCH / RESEARCH
          ↓
        RESEARCH MATERIAL
          ↓
        QWEN
          ↓
        EVIDENCE / FACT DOSSIER
          ↓
        FACT-CHECK / VALIDATE
          ↓
        QWEN
          ↓
        NARRATION
          ↓
        QWEN
          ↓
        INTELLIGENT SCENE PLAN
          ↓
        QWEN
          ↓
        IMAGE DESCRIPTIONS
          ↓
        SDXL

    This function only orchestrates stages.
    Individual processors remain responsible for their
    own generation and validation.
    """

    try:

        # ====================================================
        # JOB DIRECTORY
        # ====================================================

        job_dir = paths.job(job_id)

        os.makedirs(job_dir, exist_ok=True)

        # ====================================================
        # FILE PATHS
        # ====================================================

        research_path = os.path.join(
            job_dir,
            "research.json",
        )

        fact_check_path = os.path.join(
            job_dir,
            "fact_check.json",
        )

        visual_bible_path = os.path.join(
            job_dir,
            "visual_bible.json",
        )

        narration_path = os.path.join(
            job_dir,
            "narration.txt",
        )

        scenes_path = os.path.join(
            job_dir,
            "scenes.json",
        )

        # ====================================================
        # START
        # ====================================================

        print(
            f"[QWEN] {job_id} | "
            f"START | {topic.title}"
        )

        # ====================================================
        # STAGE 1 — RESEARCH
        # ====================================================

        existing_research = _read_json(
            research_path
        )

        if existing_research is not None:

            research = existing_research

            print(
                f"[QWEN] {job_id} | "
                "RESEARCH | existing output found -> skipping"
            )

        else:

            print(
                f"[QWEN] {job_id} | "
                "STAGE: RESEARCH | researching topic"
            )

            _update_status(
                paths,
                job_id,
                "research",
                "Researching topic",
            )

            research = research_engine.run(
                topic=topic,
                config=config,
                qwen=qwen,
            )

            _write_json(
                research_path,
                research,
            )

            print(
                f"[QWEN] {job_id} | "
                "RESEARCH | complete"
            )

        # ====================================================
        # STAGE 2 — FACT CHECK
        # ====================================================

        existing_fact_check = _read_json(
            fact_check_path
        )

        if existing_fact_check is not None:

            fact_check = existing_fact_check

            print(
                f"[QWEN] {job_id} | "
                "FACT_CHECK | existing output found -> skipping"
            )

        else:

            print(
                f"[QWEN] {job_id} | "
                "STAGE: FACT_CHECK | reviewing research classifications"
            )

            _update_status(
                paths,
                job_id,
                "fact_check",
                "Validating research and evidence",
            )

            fact_check = fact_checker.run(
                topic=topic,
                research=research,
                config=config,
                qwen=qwen,
            )

            _write_json(
                fact_check_path,
                fact_check,
            )

            print(
                f"[QWEN] {job_id} | "
                "FACT_CHECK | complete"
            )

        # ====================================================
        # STAGE 3 — VISUAL BIBLE
        # ====================================================

        existing_visual_bible = _read_json(
            visual_bible_path
        )

        if existing_visual_bible is not None:

            visual_bible_data = existing_visual_bible

            print(
                f"[QWEN] {job_id} | "
                "VISUAL_BIBLE | existing output found -> skipping"
            )

        else:

            print(
                f"[QWEN] {job_id} | "
                "STAGE: VISUAL_BIBLE | building consistent visual rules"
            )

            _update_status(
                paths,
                job_id,
                "visual_bible",
                "Building visual bible",
            )

            visual_bible_data = visual_bible.run(
                topic=topic,
                research=research,
                config=config,
                qwen=qwen,
            )

            _write_json(
                visual_bible_path,
                visual_bible_data,
            )

            print(
                f"[QWEN] {job_id} | "
                "VISUAL_BIBLE | complete"
            )

        # ====================================================
        # STAGE 4 — NARRATION
        # ====================================================

        if os.path.exists(narration_path):

            with open(
                narration_path,
                "r",
                encoding="utf-8",
            ) as f:
                narration = f.read().strip()

            print(
                f"[QWEN] {job_id} | "
                "NARRATION | existing output found -> skipping"
            )

        else:

            print(
                f"[QWEN] {job_id} | "
                "STAGE: NARRATION | writing and validating narration"
            )

            _update_status(
                paths,
                job_id,
                "narration",
                "Writing and validating narration",
            )

            narration = script_engine.run(
                paths=paths,
                job_id=job_id,
                topic=topic,
                research=research,
                fact_check=fact_check,
                visual_bible=visual_bible_data,
                config=config,
                qwen=qwen,
            )

            narration = narration.strip()

            with open(
                narration_path,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(narration)

            print(
                f"[QWEN] {job_id} | "
                "NARRATION | complete"
            )

        # ====================================================
        # IMPORTANT NARRATION VALIDATION
        # ====================================================

        if not narration:

            raise ValueError(
                "Narration is empty."
            )

        print(
            f"[QWEN] {job_id} | "
            f"NARRATION WORDS: {len(narration.split())}"
        )

        # ====================================================
        # STAGE 5 — INTELLIGENT SCENE PLANNING
        # ====================================================

        if os.path.exists(scenes_path):

            with open(
                scenes_path,
                "r",
                encoding="utf-8",
            ) as f:
                scenes = json.load(f)

            print(
                f"[QWEN] {job_id} | "
                "SCENE_PLANNING | existing output found -> skipping"
            )

        else:

            print(
                f"[QWEN] {job_id} | "
                "STAGE: SCENE_PLANNING | "
                "Qwen intelligently planning scenes"
            )

            _update_status(
                paths,
                job_id,
                "scene_planning",
                "Planning cinematic scenes",
            )

            scenes = scene_engine.run(
                paths=paths,
                job_id=job_id,
                narration=narration,
                visual_bible=visual_bible_data,
                config=config,
                qwen=qwen,
            )

            _write_json(
                scenes_path,
                scenes,
            )

            print(
                f"[QWEN] {job_id} | "
                "SCENE_PLANNING | complete"
            )

        # ====================================================
        # FINAL QWEN VALIDATION
        # ====================================================

        if not isinstance(scenes, dict):

            raise ValueError(
                "Scene output must be a dictionary."
            )

        scene_list = scenes.get(
            "scenes"
        )

        if not isinstance(
            scene_list,
            list,
        ):

            raise ValueError(
                "Scene output does not contain "
                "a valid 'scenes' list."
            )

        if not (
            18
            <= len(scene_list)
            <= 22
        ):

            raise ValueError(
                f"Invalid scene count: "
                f"{len(scene_list)}"
            )

        # ====================================================
        # COMPLETE
        # ====================================================

        _update_status(
            paths,
            job_id,
            "qwen_complete",
            "Qwen processing complete",
        )

        print(
            f"[QWEN] {job_id} | "
            f"COMPLETE | "
            f"{len(scene_list)} scenes prepared"
        )

        return True

    except Exception as e:

        print(
            f"[QWEN] ERROR {job_id} | {e}"
        )

        traceback.print_exc()

        _update_status(
            paths,
            job_id,
            "error",
            str(e),
        )

        return False
