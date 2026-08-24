"""Real video-assembly implementation, driven entirely through ffmpeg via
subprocess.

Contract:
  input:  scenes list, image files, audio files, config (video_width,
          video_height, video_fps, enable_music, enable_subtitles)
  output: renders to paths.video_rendering(job_id) first, then on success
          moves to paths.video_completed(job_id).

Pipeline:
  1. Per scene: still image -> a short video clip sized to that scene's
     ACTUAL narration audio duration, with a Ken Burns zoom/pan matching
     the scene's requested camera move, muxed with its narration audio.
  2. All scene clips concatenated with crossfade transitions.
  3. Optional subtitle burn-in, generated from each scene's narration text.
  4. Optional background music mixed in under the narration at low volume,
     if a music file is present in 05_AUDIO/music/.
"""

from __future__ import annotations
import glob
import os
import shutil
import subprocess
import tempfile

from .audio_engine import wav_duration_seconds

CAMERA_ZOOMPAN = {
    "zoom_in":   dict(z="min(zoom+0.0020,1.4)", x="iw/2-(iw/zoom/2)", y="ih/2-(ih/zoom/2)"),
    "zoom_out":  dict(z="if(eq(on,0),1.4,max(zoom-0.0020,1.0))", x="iw/2-(iw/zoom/2)", y="ih/2-(ih/zoom/2)"),
    "slow_push": dict(z="min(zoom+0.0010,1.2)", x="iw/2-(iw/zoom/2)", y="ih/2-(ih/zoom/2)"),
    "slow_pull": dict(z="if(eq(on,0),1.2,max(zoom-0.0010,1.0))", x="iw/2-(iw/zoom/2)", y="ih/2-(ih/zoom/2)"),
    "pan_left":  dict(z="1.15", x="max(iw-iw/zoom-(iw-iw/zoom)*(on/{d}),0)", y="ih/2-(ih/zoom/2)"),
    "pan_right": dict(z="1.15", x="(iw-iw/zoom)*(on/{d})",                  y="ih/2-(ih/zoom/2)"),
    "pan_up":    dict(z="1.15", x="iw/2-(iw/zoom/2)", y="max(ih-ih/zoom-(ih-ih/zoom)*(on/{d}),0)"),
    "pan_down":  dict(z="1.15", x="iw/2-(iw/zoom/2)", y="(ih-ih/zoom)*(on/{d})"),
}
DEFAULT_CAMERA = "slow_push"


def _run(cmd: list) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg command failed:\n"
            + " ".join(cmd)
            + "\n--- stderr (tail) ---\n"
            + result.stderr.decode("utf-8", errors="replace")[-2000:]
        )


def _build_scene_clip(image_path, audio_path, duration, camera, width, height, fps, out_path):
    frames = max(1, int(round(duration * fps)))
    move_template = CAMERA_ZOOMPAN.get(camera, CAMERA_ZOOMPAN[DEFAULT_CAMERA])
    move = {k: v.format(d=frames) for k, v in move_template.items()}
    upscale = f"{width * 2}:{height * 2}"
    zoompan = (
        f"scale={upscale}:force_original_aspect_ratio=increase,"
        f"crop={width * 2}:{height * 2},"
        f"zoompan=z='{move['z']}':x='{move['x']}':y='{move['y']}':"
        f"d={frames}:s={width}x{height}:fps={fps}"
    )
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-vf", zoompan,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-t", f"{duration:.3f}",
        "-shortest",
        out_path,
    ]
    _run(cmd)


def _concat_with_crossfade(clip_paths, durations, transition_sec, width, height, fps, out_path):
    n = len(clip_paths)
    if n == 1:
        shutil.copy(clip_paths[0], out_path)
        return

    safe_transition = min(transition_sec, min(durations) * 0.4)
    inputs = []
    for p in clip_paths:
        inputs += ["-i", p]

    filter_parts = []
    v_label = "0:v"
    a_label = "0:a"
    cumulative = durations[0]
    for i in range(1, n):
        offset = max(cumulative - safe_transition, 0)
        next_v, next_a = f"v{i}", f"a{i}"
        filter_parts.append(
            f"[{v_label}][{i}:v]xfade=transition=fade:duration={safe_transition:.3f}:"
            f"offset={offset:.3f}[{next_v}]"
        )
        filter_parts.append(
            f"[{a_label}][{i}:a]acrossfade=d={safe_transition:.3f}[{next_a}]"
        )
        v_label, a_label = next_v, next_a
        cumulative = cumulative - safe_transition + durations[i]

    filter_complex = ";".join(filter_parts)
    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{v_label}]", "-map", f"[{a_label}]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        out_path,
    ]
    _run(cmd)


def _write_srt(scenes, durations, out_path):
    def _fmt(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    lines = []
    t = 0.0
    for i, (scene, dur) in enumerate(zip(scenes, durations), start=1):
        start, end = t, t + dur
        text = (scene.get("narration") or "").strip() or " "
        lines.append(f"{i}\n{_fmt(start)} --> {_fmt(end)}\n{text}\n")
        t = end
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _burn_subtitles(in_path, srt_path, out_path):
    cmd = ["ffmpeg", "-y", "-i", in_path, "-vf", f"subtitles={srt_path}",
           "-c:a", "copy", out_path]
    _run(cmd)


def _mix_music(in_path, music_path, out_path, music_volume=0.12):
    cmd = [
        "ffmpeg", "-y", "-i", in_path, "-i", music_path,
        "-filter_complex",
        f"[1:a]volume={music_volume},aloop=loop=-1:size=2e9[music];"
        f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac",
        out_path,
    ]
    _run(cmd)


def _find_music_file(paths):
    music_dir = paths("05_AUDIO", "music")
    if not os.path.isdir(music_dir):
        return None
    for ext in ("*.mp3", "*.wav", "*.m4a"):
        matches = sorted(glob.glob(os.path.join(music_dir, ext)))
        if matches:
            return matches[0]
    return None


def run(paths, job_id: str, scenes: list, image_files: list, audio_files: list, config) -> str:
    width = getattr(config, "video_width", 1080)
    height = getattr(config, "video_height", 1920)
    fps = getattr(config, "video_fps", 25)
    enable_music = getattr(config, "enable_music", True)
    enable_subtitles = getattr(config, "enable_subtitles", True)

    if not scenes or not image_files or not audio_files:
        raise RuntimeError(f"Cannot render {job_id}: missing scenes, images, or audio.")

    image_by_scene = {int(os.path.basename(p).split("_")[1].split(".")[0]): p for p in image_files}
    audio_by_scene = {int(os.path.basename(p).split("_")[1].split(".")[0]): p for p in audio_files}

    with tempfile.TemporaryDirectory(prefix=f"bhf_render_{job_id}_") as tmp:
        clip_paths, durations = [], []
        for scene in scenes:
            sid = scene["scene_id"]
            img = image_by_scene.get(sid)
            aud = audio_by_scene.get(sid)
            if not img or not aud:
                raise RuntimeError(f"Missing image or audio for {job_id} scene {sid}")
            duration = max(1.0, wav_duration_seconds(aud))
            clip_path = os.path.join(tmp, f"clip_{sid:03d}.mp4")
            _build_scene_clip(img, aud, duration, scene.get("camera", DEFAULT_CAMERA),
                               width, height, fps, clip_path)
            clip_paths.append(clip_path)
            durations.append(duration)

        concatenated = os.path.join(tmp, "concatenated.mp4")
        _concat_with_crossfade(clip_paths, durations, transition_sec=0.6,
                                width=width, height=height, fps=fps, out_path=concatenated)

        current = concatenated
        if enable_subtitles:
            srt_path = os.path.join(tmp, "subs.srt")
            _write_srt(scenes, durations, srt_path)
            subbed = os.path.join(tmp, "subbed.mp4")
            _burn_subtitles(current, srt_path, subbed)
            current = subbed

        if enable_music:
            music_file = _find_music_file(paths)
            if music_file:
                mixed = os.path.join(tmp, "mixed.mp4")
                _mix_music(current, music_file, mixed)
                current = mixed

        rendering_path = paths.video_rendering(job_id)
        os.makedirs(os.path.dirname(rendering_path), exist_ok=True)
        shutil.copy(current, rendering_path)

    completed_path = paths.video_completed(job_id)
    os.makedirs(os.path.dirname(completed_path), exist_ok=True)
    shutil.move(rendering_path, completed_path)
    return completed_path
