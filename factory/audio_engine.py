from __future__ import annotations
import os, subprocess, wave
from .utils import write_json_atomic

def load_piper_voice(model_path):
    from piper import PiperVoice
    return PiperVoice.load(model_path, use_cuda=True)

def synthesize(voice, text, out):
    with wave.open(out, "wb") as f: voice.synthesize(text, f)

def duration(path):
    return float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nk=1:nw=1",path], text=True).strip())

def run(paths, job_id, scenes, voice, config, progress=None):
    out = paths.audio_dir(job_id); os.makedirs(out, exist_ok=True); files=[]
    total=len(scenes)
    for i, scene in enumerate(scenes, 1):
        p=os.path.join(out,f"scene_{int(scene['scene_id']):03d}.wav")
        if not os.path.exists(p):
            print(f"[AUDIO] {job_id} | STAGE: NARRATION | scene {i}/{total} | generating")
            synthesize(voice, scene["narration"], p)
        else: print(f"[AUDIO] {job_id} | scene {i}/{total} | exists, skip")
        files.append(p)
        if progress: progress(i,total)
    concat=os.path.join(out,"narration_concat.txt"); final=paths.audio_final(job_id)
    with open(concat,"w",encoding="utf8") as f:
        for p in files: f.write("file '"+os.path.abspath(p).replace("'","'\\''")+"'\n")
    if not os.path.exists(final):
        subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",concat,"-c","copy",final],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    seconds=duration(final)
    content_min=float(config.min_video_seconds)-float(config.source_card_seconds)
    content_max=float(config.max_video_seconds)-float(config.source_card_seconds)
    ready=content_min <= seconds <= content_max
    write_json_atomic(paths.state(job_id,"audio"),{"status":"AUDIO_READY" if ready else "AUDIO_REVIEW_REQUIRED","content_seconds":seconds,"final_with_source_card_seconds":seconds+float(config.source_card_seconds),"scene_count":len(files)})
    return files, seconds
