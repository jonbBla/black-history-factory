"""Real audio-generation implementation.

Contract:
  input:  scenes list (each with its own "narration" text) + a loaded
          Piper voice
  output: paths.audio_dir(job_id)/scene_{NNN}.wav -- one file per scene, so
          video_engine.py can match each image's on-screen duration to its
          own narration clip length instead of one long audio track.

Same per-file skip + on_progress(completed_count) checkpoint pattern as
image_engine.py.
"""

from __future__ import annotations
import os
import wave
import struct


def load_piper_voice(model_path: str = None, device: str = "cuda"):
    """Call once in Colab Cell 4, e.g.:
        from factory.audio_engine import load_piper_voice
        models["piper"] = load_piper_voice("/content/en_US-lessac-medium.onnx")
    Piper voice models (.onnx + .onnx.json) are downloaded separately --
    see https://github.com/rhasspy/piper/blob/master/VOICES.md for the
    catalog.
    """
    from piper import PiperVoice
    return PiperVoice.load(model_path, use_cuda=(device == "cuda"))


def _write_silence_wav(path: str, seconds: float = 3.0, rate: int = 22050) -> None:
    """Fallback used when no Piper voice is loaded, so downstream stages
    (duration matching, FFmpeg assembly) still have a real WAV to read."""
    n_frames = max(1, int(seconds * rate))
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<" + "h" * n_frames, *([0] * n_frames)))


def _synthesize(voice, text: str, path: str) -> None:
    import wave as wave_mod
    with wave_mod.open(path, "wb") as wav_file:
        voice.synthesize(text, wav_file)


def run(paths, job_id: str, scenes: list, piper=None, on_progress=None) -> list:
    out_dir = paths.audio_dir(job_id)
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for scene in scenes:
        fname = os.path.join(out_dir, f"scene_{scene['scene_id']:03d}.wav")
        if os.path.exists(fname):
            written.append(fname)
            if on_progress:
                on_progress(len(written))
            continue

        text = (scene.get("narration") or "").strip()
        if piper is None or not text:
            _write_silence_wav(fname, seconds=scene.get("duration", 8))
        else:
            try:
                _synthesize(piper, text, fname)
            except Exception as e:
                raise RuntimeError(
                    f"Audio generation failed for {job_id} scene {scene['scene_id']}: {e}"
                ) from e

        written.append(fname)
        if on_progress:
            on_progress(len(written))

    return written


def wav_duration_seconds(path: str) -> float:
    """Used by video_engine.py to size each scene's clip to its actual
    narration length rather than the (approximate) planned duration."""
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())
