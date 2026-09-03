from __future__ import annotations

import os
import traceback

from factory import (
    research_engine,
    fact_checker,
    visual_bible,
    script_engine,
    scene_engine,
    status,
    topic_engine,
)
from .utils import read_json, write_json_atomic


def _update_status(paths, job_id, stage, message, state="running"):
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
        print(f"[STATUS] WARNING | Could not update status: {e}")


def _run_or_load(path, label, job_id, fn):
    existing = read_json(path, None)
    if existing is not None:
        print(f"[QWEN] {job_id} | {label} | existing output found -> skipping")
        return existing
    return fn()


def run_one(paths, topic, job_id, config, qwen):
    try:
        print(f"[QWEN] {job_id} | START | {topic.title}")

        # ----------------------------------------------------
        # 1. SOURCE SEARCH + EVIDENCE DOSSIER
        # ----------------------------------------------------
        _update_status(paths, job_id, "research", "Source search and evidence dossier")
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

        # ----------------------------------------------------
        # 2. FACT CHECK / VALIDATE
        # ----------------------------------------------------
        _update_status(paths, job_id, "fact_check", "Fact-checking and validating dossier")
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

        # ----------------------------------------------------
        # 3. VISUAL BIBLE
        # Kept as a consistency layer for image generation.
        # ----------------------------------------------------
        _update_status(paths, job_id, "visual_bible", "Building visual consistency rules")
        visual_path = paths.visual_bible(job_id)
        visual_data = read_json(visual_path, None)

        if visual_data is None:
            visual_data = visual_bible.run(
                topic=topic,
                research=fact_check,
                config=config,
                qwen=qwen,
            )
            write_json_atomic(visual_path, visual_data)
        else:
            print(f"[QWEN] {job_id} | VISUAL_BIBLE | existing output found -> skipping")

        # ----------------------------------------------------
        # 4. NARRATION
        # ----------------------------------------------------
        _update_status(paths, job_id, "narration", "Writing and validating narration")
        narration_path = paths.narration(job_id)

        if os.path.exists(narration_path):
            with open(narration_path, "r", encoding="utf8") as f:
                narration = f.read().strip()
            print(f"[QWEN] {job_id} | NARRATION | existing output found -> skipping")
        else:
            narration = script_engine.run(
                paths=paths,
                job_id=job_id,
                topic=topic,
                research=research,
                fact_check=fact_check,
                visual_bible=visual_data,
                config=config,
                qwen=qwen,
            )

        if not narration:
            raise ValueError("Narration is empty.")

        print(
            f"[QWEN] {job_id} | NARRATION WORDS: "
            f"{script_engine.count_words(narration)}"
        )

        # ----------------------------------------------------
        # 5. INTELLIGENT SCENE PLAN + IMAGE DESCRIPTIONS
        # ----------------------------------------------------
        _update_status(paths, job_id, "scene_planning", "Qwen intelligently planning scenes")
        scenes_path = paths.scenes(job_id)

        if os.path.exists(scenes_path):
            scenes = read_json(scenes_path, None)
            print(f"[QWEN] {job_id} | SCENE_PLANNING | existing output found -> skipping")
        else:
            scenes = scene_engine.run(
                paths=paths,
                job_id=job_id,
                narration=narration,
                visual_bible=visual_data,
                config=config,
                qwen=qwen,
            )

        if not isinstance(scenes, dict) or not isinstance(scenes.get("scenes"), list):
            raise ValueError("Scene output must contain a 'scenes' list.")

        scene_list = scenes["scenes"]
        if not 18 <= len(scene_list) <= 22:
            raise ValueError(f"Invalid scene count: {len(scene_list)}")

        # ----------------------------------------------------
        # COMPLETE QWEN PROCESSOR
        # ----------------------------------------------------
        _update_status(
            paths,
            job_id,
            "complete",
            f"Qwen complete | {len(scene_list)} scenes prepared",
            state="idle",
        )

        # Mark topic used only after all Qwen artifacts exist.
        topic_engine.update_status(
            paths,
            job_id,
            "QWEN_COMPLETE",
            topic_id=topic.id,
            title=topic.title,
            scene_count=len(scene_list),
        )
        topic_engine.mark_used(paths, topic)

        print(
            f"[QWEN] {job_id} | COMPLETE | "
            f"{len(scene_list)} scenes prepared"
        )
        return True

    except Exception as e:
        print(f"[QWEN] ERROR {job_id} | {e}")
        traceback.print_exc()
        _update_status(paths, job_id, "error", str(e), state="error")
        try:
            topic_engine.update_status(paths, job_id, "QWEN_ERROR", error=str(e))
        except Exception:
            pass
        return False
