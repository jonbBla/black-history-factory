"""The job loop. This is what Colab notebook Cell 6 calls.

Design: every stage follows the same pattern --
    if stage_output_exists(...): skip, advance checkpoint, continue
    else: run the stage, then advance checkpoint
so stopping Colab at any point and re-running this function picks up
exactly where it left off.
"""

from __future__ import annotations
import os

from . import topic_engine, research_engine, fact_checker, script_engine
from . import scene_engine, visual_bible, image_engine, audio_engine
from . import video_engine, thumbnail_engine, status, github
from .checkpoint import Checkpoint, find_in_progress_job, stage_output_exists, all_job_ids
from .config import Config
from .drive import DrivePaths, mount_drive
from .utils import read_json, write_json_atomic, next_job_id, now_iso


class NoTopicsAvailable(RuntimeError):
    """No unused topics remain and no job is in progress. Distinct from a
    mid-job failure so main.run() knows it's safe to mark status idle."""


def _get_or_start_job(paths) -> tuple:
    """Resume the in-progress job if one exists, otherwise pick a new topic
    and start a fresh job. Job ids and topic ids are separate namespaces --
    the checkpoint stores which topic_id a job belongs to, so resuming
    never has to guess the mapping between them."""
    in_progress_id = find_in_progress_job(paths)
    if in_progress_id:
        cp = Checkpoint.load(paths, in_progress_id)
        topics = topic_engine.load_topics(paths)
        topic = next((t for t in topics if t.id == cp.topic_id), None)
        return cp, topic

    topic = topic_engine.select_next_topic(paths)
    if topic is None:
        raise NoTopicsAvailable(
            "No unused topics remain in 01_TOPICS/topics.json. "
            "Add more, or wire in Phase C's AI topic generation."
        )
    job_id = next_job_id(all_job_ids(paths))
    cp = Checkpoint.load_or_create(paths, job_id, title=topic.title, topic_id=topic.id)
    return cp, topic


def run_one_job(paths, config, gh_token: str = "", models: dict = None) -> Checkpoint:
    models = models or {}
    qwen = models.get("qwen")
    cp, topic = _get_or_start_job(paths)
    if topic is None:
        raise RuntimeError("Could not resolve the topic for the in-progress job.")

    def _push():
        current = status.update_current(paths, cp)
        history = read_json(paths.status_history, default=[])
        github.push_status(config, current, history, gh_token)

    try:
        return _run_job_stages(paths, config, cp, topic, qwen, models, _push)
    except Exception as e:
        # Any stage failure marks the checkpoint as an error (visible on
        # the dashboard) instead of silently dying or leaving current.json
        # showing a stale "running" state. The topic is NOT marked used,
        # so a later run will retry this exact job from wherever it
        # stopped.
        cp.fail(paths, str(e))
        _push()
        raise


def _run_job_stages(paths, config, cp, topic, qwen, models, _push) -> Checkpoint:
    _push()

    # If a previous job in this same Colab session offloaded Qwen to CPU
    # before its image-generation stage, restore it now -- this job's
    # research/narration/visual-bible/scene stages need it back on GPU.
    # No-op if Qwen is already on GPU, was never offloaded, or is
    # quantized (see QwenClient.restore_to_gpu's docstring).
    if qwen is not None and hasattr(qwen, "restore_to_gpu"):
        qwen.restore_to_gpu()

    # --- research ---
    if not stage_output_exists(paths, cp.job_id, "research"):
        cp.advance(paths, stage="research", status="running"); _push()
        research_engine.run(paths, cp.job_id, topic, qwen=qwen)
    # --- fact check ---
    if not stage_output_exists(paths, cp.job_id, "fact_check"):
        cp.advance(paths, stage="fact_check"); _push()
        fact_checker.run(paths, cp.job_id, qwen=qwen)
    research = read_json(paths.research_verified(cp.job_id), default={})

    # --- narration ---
    if not stage_output_exists(paths, cp.job_id, "narration"):
        cp.advance(paths, stage="narration"); _push()
        script_engine.run(paths, cp.job_id, topic.title, research, config, qwen=qwen)
    with open(paths.narration_txt(cp.job_id), "r", encoding="utf-8") as f:
        narration_text = f.read()

    # --- visual bible ---
    if not stage_output_exists(paths, cp.job_id, "visual_bible"):
        cp.advance(paths, stage="visual_bible"); _push()
        vb = visual_bible.run(topic, research, config, qwen=qwen)
        write_json_atomic(paths.visual_bible_json(cp.job_id), vb)
    else:
        vb = read_json(paths.visual_bible_json(cp.job_id), default={})

    # --- scene planning ---
    if not stage_output_exists(paths, cp.job_id, "scene_planning"):
        cp.advance(paths, stage="scene_planning"); _push()
        scenes = scene_engine.run(paths, cp.job_id, narration_text, vb, config, qwen=qwen)
    else:
        scenes = read_json(paths.scenes_json(cp.job_id), default=[])
    total_scenes = len(scenes)

    # Qwen isn't needed again until the NEXT job's research stage (audio,
    # rendering, and thumbnailing don't use it either) -- freeing its GPU
    # memory here gives image generation real headroom on a tight-VRAM GPU
    # instead of Qwen sitting resident and unused for the rest of this job.
    if qwen is not None and hasattr(qwen, "offload_to_cpu"):
        qwen.offload_to_cpu()

    # --- images ---
    if not stage_output_exists(paths, cp.job_id, "image_generation", total_scenes):
        cp.advance(paths, stage="image_generation", total=total_scenes, completed=0); _push()

        def _image_progress(n):
            cp.advance(paths, completed=n); _push()

        image_files = image_engine.run(paths, cp.job_id, scenes, config,
                                        flux=models.get("flux"), on_progress=_image_progress,
                                        upscaler=models.get("upscaler"))
    else:
        image_files = sorted(
            os.path.join(paths.images_dir(cp.job_id), f)
            for f in os.listdir(paths.images_dir(cp.job_id)) if f.endswith(".png")
        )

    # --- audio ---
    if not stage_output_exists(paths, cp.job_id, "audio_generation", total_scenes):
        cp.advance(paths, stage="audio_generation", total=total_scenes, completed=0); _push()

        def _audio_progress(n):
            cp.advance(paths, completed=n); _push()

        audio_files = audio_engine.run(paths, cp.job_id, scenes,
                                        piper=models.get("piper"), on_progress=_audio_progress)
    else:
        audio_files = sorted(
            os.path.join(paths.audio_dir(cp.job_id), f)
            for f in os.listdir(paths.audio_dir(cp.job_id)) if f.endswith(".wav")
        )

    # --- render ---
    if not stage_output_exists(paths, cp.job_id, "rendering"):
        cp.advance(paths, stage="rendering", total=1, completed=0); _push()
        video_path = video_engine.run(paths, cp.job_id, scenes, image_files, audio_files, config)
        cp.advance(paths, completed=1); _push()
    else:
        video_path = paths.video_completed(cp.job_id)

    # --- thumbnail ---
    if not stage_output_exists(paths, cp.job_id, "thumbnail"):
        cp.advance(paths, stage="thumbnail"); _push()
        thumb_path = image_files[0] if image_files else ""
        thumbnail_engine.run(paths, cp.job_id, thumb_path, title=cp.title)

    # --- done ---
    cp.advance(paths, stage="completed", status="completed"); _push()
    topic_engine.mark_used(paths, topic.id)
    status.append_history(paths, job_id=cp.job_id, title=cp.title,
                           video_path=video_path, thumbnail_path=paths.thumbnail(cp.job_id))
    _archive_job_manifest(paths, cp, topic, research, total_scenes, video_path)
    _push()
    return cp


def _archive_job_manifest(paths, cp, topic, research: dict, total_scenes: int, video_path: str) -> None:
    """A durable, human-readable record of a completed job -- which topic,
    how long each stage took (from checkpoint.stage_history), how many
    scenes, what sources the research cited, and where the final outputs
    live. Written once per completed job to 09_LOGS/, separate from (and
    outliving) the checkpoint file."""
    manifest = {
        "job_id": cp.job_id,
        "topic_id": topic.id,
        "title": cp.title,
        "category": getattr(topic, "category", ""),
        "region": getattr(topic, "region", ""),
        "period": getattr(topic, "period", ""),
        "scene_count": total_scenes,
        "sources": (research or {}).get("sources", []),
        "video_path": video_path,
        "thumbnail_path": paths.thumbnail(cp.job_id),
        "stage_history": cp.stage_history,
        "completed_at": now_iso(),
    }
    write_json_atomic(paths.job_manifest(cp.job_id), manifest)


def run(max_jobs: int = 1, gh_token: str = "", models: dict = None) -> None:
    """Entry point called from the Colab notebook.
    max_jobs=1 processes one topic and stops (safest default for testing);
    raise it once Phase H's multi-video testing has passed.
    models: dict of loaded model clients, e.g. {"qwen": QwenClient.load()},
    built once in Colab Cell 4 and passed straight through.
    """
    drive_root = mount_drive()
    paths = DrivePaths(root=os.path.join(drive_root, "BLACK_HISTORY_FACTORY"))
    paths.ensure_tree()
    config = Config.load(paths.root)

    for _ in range(max_jobs):
        try:
            run_one_job(paths, config, gh_token=gh_token, models=models)
        except NoTopicsAvailable as e:
            print(f"[factory] stopping: {e}")
            status.mark_idle(paths)
            break
        except Exception as e:
            print(f"[factory] job failed, stopping: {e}")
            break
