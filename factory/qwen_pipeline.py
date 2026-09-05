from __future__ import annotations

import os
import traceback

from factory import (
    research_engine,
    fact_checker,
    script_engine,
    scene_engine,
    status,
    topic_engine,
)

from .utils import read_json


def _update_status(
    paths,
    job_id,
    stage,
    message,
    state="running",
):
    """
    Update the Qwen processor status.

    This uses status.set_processor() only.
    Do NOT use topic_engine.update_status() because
    that function does not exist.
    """
    try:
        status.set_processor(
            paths,
            "qwen",
            state,
            job_id=job_id,
            stage=stage,
            detail=message,
        )
    except Exception as e:
        print(f"[STATUS] WARNING | {e}")


def _run_or_load(
    path,
    label,
    job_id,
    fn,
):
    """
    Load an existing JSON output if it already exists.
    Otherwise run the supplied function.
    """

    existing = read_json(path, None)

    if existing is not None:
        print(
            f"[QWEN] {job_id} | {label} | "
            f"existing output found -> skipping"
        )
        return existing

    return fn()


def run_one(
    paths,
    topic,
    job_id,
    config,
    qwen,
):
    """
    Process one topic through the complete Qwen pipeline.

    Workflow:

        RANDOM TOPIC
              ↓
        SOURCE SEARCH
              ↓
        QWEN RESEARCH
              ↓
        SOURCE-BASED VERIFICATION
              ↓
        QWEN NARRATION
              ↓
        QWEN DETAILED INTELLIGENT SCENES
              ↓
        scenes.json
    """

    try:
        print(
            f"[QWEN] {job_id} | START | {topic.title}"
        )

        # ---------------------------------------------------------
        # 1. RESEARCH
        # ---------------------------------------------------------

        _update_status(
            paths,
            job_id,
            "research",
            "Source search + Qwen research",
        )

        research = _run_or_load(
            paths.research(job_id),
            "RESEARCH",
            job_id,
            lambda: research_engine.run(
                paths=paths,
                job_id=job_id,
                topic=topic,
                config=config,
                qwen=qwen,
            ),
        )

        # ---------------------------------------------------------
        # 2. FACT CHECK
        # ---------------------------------------------------------

        _update_status(
            paths,
            job_id,
            "fact_check",
            "Verifying research against discovered sources",
        )

        fact_check = _run_or_load(
            paths.verified(job_id),
            "FACT_CHECK",
            job_id,
            lambda: fact_checker.run(
                paths=paths,
                job_id=job_id,
                topic=topic,
                research=research,
                config=config,
                qwen=qwen,
            ),
        )

        # ---------------------------------------------------------
        # 3. NARRATION
        # ---------------------------------------------------------

        _update_status(
            paths,
            job_id,
            "narration",
            "Writing narration from verified research",
        )

        narration_path = paths.narration(job_id)

        if os.path.exists(narration_path):
            with open(
                narration_path,
                encoding="utf-8",
            ) as f:
                narration = f.read().strip()

            print(
                f"[QWEN] {job_id} | NARRATION | "
                f"existing output found -> skipping"
            )

        else:
            narration = script_engine.run(
                paths=paths,
                job_id=job_id,
                topic=topic,
                research=research,
                fact_check=fact_check,
                config=config,
                qwen=qwen,
            )

        if not narration:
            raise ValueError(
                "Narration is empty."
            )

        narration_words = script_engine.count_words(
            narration
        )

        print(
            f"[QWEN] {job_id} | NARRATION WORDS: "
            f"{narration_words}"
        )

        # ---------------------------------------------------------
        # 4. INTELLIGENT SCENE PLANNING
        # ---------------------------------------------------------

        _update_status(
            paths,
            job_id,
            "scene_planning",
            "Qwen creating intelligent detailed scenes",
        )

        scenes_path = paths.scenes(job_id)

        if os.path.exists(scenes_path):

            scenes = read_json(
                scenes_path,
                None,
            )

            print(
                f"[QWEN] {job_id} | SCENES | "
                f"existing output found -> skipping"
            )

        else:

            scenes = scene_engine.run(
                paths=paths,
                job_id=job_id,
                narration=narration,
                research=research,
                fact_check=fact_check,
                config=config,
                qwen=qwen,
            )

        # ---------------------------------------------------------
        # 5. BASIC SCENE VALIDATION
        # ---------------------------------------------------------
        #
        # IMPORTANT:
        # There is intentionally NO scene-count restriction here.
        #
        # Qwen decides how many scenes are appropriate.
        #
        # We only verify that scene_engine returned a list.
        # scene_engine itself validates sentence coverage,
        # ordering, camera values, visual descriptions, etc.
        # ---------------------------------------------------------

        if not isinstance(scenes, list):
            raise ValueError(
                "scenes.json must contain a JSON list "
                "of scene objects."
            )

        scene_count = len(scenes)

        if scene_count == 0:
            raise ValueError(
                "scenes.json contains no scenes."
            )

        print(
            f"[QWEN] {job_id} | SCENES: "
            f"{scene_count}"
        )

        # ---------------------------------------------------------
        # 6. COMPLETE
        # ---------------------------------------------------------

        _update_status(
            paths,
            job_id,
            "complete",
            (
                f"Qwen complete | "
                f"{scene_count} scenes | "
                f"{narration_words} narration words"
            ),
            state="idle",
        )

        # Mark the topic as successfully used.
        #
        # IMPORTANT:
        # topic_engine.update_status() does NOT exist.
        # Do not call it here.
        topic_engine.mark_used(
            paths,
            topic,
        )

        print(
            f"[QWEN] {job_id} | COMPLETE | "
            f"{scene_count} scenes prepared"
        )

        return True

    # -------------------------------------------------------------
    # ERROR HANDLING
    # -------------------------------------------------------------

    except Exception as e:

        print(
            f"[QWEN] ERROR {job_id} | {e}"
        )

        traceback.print_exc()

        # Update the processor status only.
        # There is deliberately NO topic_engine.update_status()
        # call here.
        _update_status(
            paths,
            job_id,
            "error",
            str(e),
            state="error",
        )

        return False
