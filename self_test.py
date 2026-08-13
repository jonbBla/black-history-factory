"""Phase H tooling: a local, no-GPU sanity check for the whole pipeline.

Runs main.run_one_job() against a mock Qwen client (so no model download or
GPU is needed) but through the REAL video_engine/thumbnail_engine, which
only need ffmpeg/PIL. This catches integration breakage -- wrong function
signatures, bad file paths, checkpoint/resume regressions, ffmpeg command
errors -- before you spend Colab GPU time on a real run.

It does NOT validate research quality, narration quality, or image/audio
fidelity -- only that the pipeline runs, checkpoints, resumes, and produces
non-empty output files at every stage.

Usage:
    cd black-history-factory
    python3 tests/self_test.py

Exits non-zero on any failed assertion.
"""

from __future__ import annotations
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factory.drive import DrivePaths
from factory.config import Config
from factory import main, topic_engine
from factory.utils import read_json, write_json_atomic
from factory.checkpoint import Checkpoint, all_job_ids, find_in_progress_job


NARRATION = (
    "For centuries a hidden feat of engineering lay buried beneath the sand. "
    "This is the story of an ancient kingdom's mastery over water itself."
)

RESEARCH = {
    "topic": "x", "overview": "Test overview.",
    "timeline": [{"event": "Dam construction", "classification": "archaeological_evidence"}],
    "people": [], "architecture": [{"item": "reservoirs", "classification": "established_fact"}],
    "technology": [], "daily_life": [], "religion": [],
    "mythology": [{"claim": "River spirit legend", "classification": "mythology"}],
    "trade": [], "art": [], "lesser_known_facts": [], "archaeological_evidence": [],
    "scholarly_debates": [], "sources": ["Example Museum catalog"],
}

VISUAL_BIBLE = {
    "architecture": "sandstone reservoir walls", "clothing": "linen wraps",
    "materials": "sandstone, clay", "environment": "riverside plain",
    "lighting": "harsh midday sun",
}

SCENES = [
    {"scene_id": 1, "duration": 1, "narration": "Opening hook line.", "location": "riverside",
     "period": "4th century", "characters": [], "objects": [], "camera": "zoom_in", "transition": "crossfade"},
    {"scene_id": 2, "duration": 1, "narration": "The discovery.", "location": "ruins",
     "period": "modern", "characters": [], "objects": [], "camera": "pan_left", "transition": "crossfade"},
]


class MockQwen:
    def __init__(self):
        self.calls = {"generate": 0, "generate_json": 0}

    def generate(self, prompt, max_new_tokens=2048, temperature=0.7):
        self.calls["generate"] += 1
        return NARRATION

    def generate_json(self, prompt, max_new_tokens=2048, retries=2):
        self.calls["generate_json"] += 1
        if "Review the research package" in prompt:
            return RESEARCH
        if "Establish the shared visual rules" in prompt:
            return VISUAL_BIBLE
        if "Break the narration" in prompt:
            return SCENES
        return RESEARCH


def _check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        raise AssertionError(label)


def _reset_topics(paths):
    topics = read_json(paths.topics_json, default=[])
    for t in topics:
        t["used"] = False
    write_json_atomic(paths.topics_json, topics)
    write_json_atomic(paths.used_topics_json, [])
    write_json_atomic(paths.rejected_topics_json, [])


def main_test():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory(prefix="bhf_selftest_") as tmp_root:
        # Work on a throwaway copy so this never touches real 00_CONFIG/01_TOPICS.
        paths = DrivePaths(root=tmp_root)
        paths.ensure_tree()
        shutil.copy(os.path.join(project_root, "00_CONFIG", "config.json"),
                    paths.config_json)
        shutil.copy(os.path.join(project_root, "01_TOPICS", "topics.json"),
                    paths.topics_json)
        write_json_atomic(paths.used_topics_json, [])
        write_json_atomic(paths.rejected_topics_json, [])
        config = Config.load(paths.root)
        # This suite cares about integration correctness (right files in the
        # right places, checkpointing, resume, error handling) -- not render
        # fidelity, which is already verified separately at full resolution
        # during development. Keep renders cheap so this runs in seconds.
        config.values["enable_subtitles"] = False
        config.values["video_width"] = 320
        config.values["video_height"] = 240
        config.values["video_fps"] = 8

        print("1. Full run with a mock model (research through rendered video)")
        qwen = MockQwen()
        cp = main.run_one_job(paths, config, models={"qwen": qwen})
        _check("job completed", cp.status == "completed")
        _check("video file exists and non-empty",
                os.path.exists(paths.video_completed(cp.job_id)) and
                os.path.getsize(paths.video_completed(cp.job_id)) > 0)
        _check("thumbnail exists and non-empty",
                os.path.exists(paths.thumbnail(cp.job_id)) and
                os.path.getsize(paths.thumbnail(cp.job_id)) > 0)
        _check("manifest archived", os.path.exists(paths.job_manifest(cp.job_id)))
        _check("history.json has an entry", len(read_json(paths.status_history, default=[])) == 1)

        print("2. A second job gets a distinct id and distinct topic")
        cp2 = main.run_one_job(paths, config, models={"qwen": MockQwen()})
        _check("distinct job id", cp2.job_id != cp.job_id)
        _check("distinct topic", cp2.topic_id != cp.topic_id)

        print("3. Simulated crash mid-job resumes the SAME job, not a new one")
        from factory import research_engine, fact_checker, script_engine, status as status_mod
        from factory.utils import next_job_id
        topic = topic_engine.select_next_topic(paths)
        job_id = next_job_id(all_job_ids(paths))
        crash_cp = Checkpoint.load_or_create(paths, job_id, title=topic.title, topic_id=topic.id)
        crash_qwen = MockQwen()
        research_engine.run(paths, job_id, topic, qwen=crash_qwen)
        fact_checker.run(paths, job_id, qwen=crash_qwen)
        script_engine.run(paths, job_id, topic.title, read_json(paths.research_verified(job_id)), config, qwen=crash_qwen)
        crash_cp.advance(paths, stage="visual_bible", status="paused")
        status_mod.update_current(paths, crash_cp, colab_status="paused")
        resumed = main.run_one_job(paths, config, models={"qwen": MockQwen()})
        _check("resumed the crashed job, not a new one", resumed.job_id == job_id)
        _check("resumed job completed", resumed.status == "completed")

        print("4. A hard model failure marks status=error and does not consume the topic")
        _reset_topics(paths)

        class BrokenQwen:
            def generate_json(self, prompt, max_new_tokens=2048, retries=2):
                raise ValueError("simulated unparseable model output")

        try:
            main.run_one_job(paths, config, models={"qwen": BrokenQwen()})
            _check("broken model call raised", False)
        except RuntimeError:
            pass
        current = read_json(paths.status_current)
        _check("status shows error", current["status"] == "error")
        _check("topic not marked used after failure", read_json(paths.used_topics_json) == [])

        print("5. Retrying after a failure resumes the same failed job")
        stuck_job = find_in_progress_job(paths)
        recovered = main.run_one_job(paths, config, models={"qwen": MockQwen()})
        _check("retry resumed the failed job", recovered.job_id == stuck_job)
        _check("retry completed", recovered.status == "completed")

        print("6. Pipeline still runs with zero models loaded (pure placeholder path)")
        _reset_topics(paths)
        no_model_cp = main.run_one_job(paths, config)
        _check("no-model job completed", no_model_cp.status == "completed")

        print("7. Locked art style is never overridden by a model-supplied value")
        _reset_topics(paths)

        class StyleLeakQwen(MockQwen):
            def generate_json(self, prompt, max_new_tokens=2048, retries=2):
                if "Establish the shared visual rules" in prompt:
                    d = dict(VISUAL_BIBLE)
                    d["style"] = "SHOULD NEVER APPEAR photoreal 3D render"
                    return d
                return super().generate_json(prompt, max_new_tokens, retries)

        leak_cp = main.run_one_job(paths, config, models={"qwen": StyleLeakQwen()})
        vb = read_json(paths.visual_bible_json(leak_cp.job_id))
        _check("locked style used, model value ignored",
                "SHOULD NEVER APPEAR" not in vb["style"] and vb["style"] == config.art_style)

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    try:
        main_test()
    except AssertionError as e:
        print(f"\nSELF-TEST FAILED: {e}")
        sys.exit(1)
