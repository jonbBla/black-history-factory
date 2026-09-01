from __future__ import annotations
import os, subprocess, shutil, tempfile
from .source_card import make_source_card

def ffprobe_duration(path):
    return float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nk=1:nw=1",path],text=True).strip())

def run(paths,job_id,scenes,research,config,progress=None):
    outdir=paths.video_dir(job_id); clips_dir=paths.video_clips_dir(job_id); os.makedirs(outdir,exist_ok=True); os.makedirs(clips_dir,exist_ok=True); tmp=tempfile.mkdtemp(prefix="bhf_")
    try:
        clips=[]; total=len(scenes)
        for i,s in enumerate(scenes,1):
            sid=int(s["scene_id"]); img=os.path.join(paths.images_dir(job_id),f"scene_{sid:03d}.png"); aud=os.path.join(paths.audio_dir(job_id),f"scene_{sid:03d}.wav"); clip=os.path.join(clips_dir,f"scene_{sid:03d}.mp4")
            if not os.path.exists(img) or not os.path.exists(aud): raise RuntimeError(f"Missing image/audio for scene {sid}")
            if os.path.exists(clip):
                clips.append(clip); print(f"[VIDEO] {job_id} | STAGE: RENDERING | scene {i}/{total} | exists, skip");
                if progress: progress(i,total)
                continue
            dur=max(0.5,ffprobe_duration(aud)); frames=max(1,int(dur*config.video_fps)); cam=s.get("camera","slow_push")
            if cam=="zoom_in": z="min(zoom+0.002,1.28)"; x="iw/2-(iw/zoom/2)"; y="ih/2-(ih/zoom/2)"
            elif cam=="zoom_out": z="if(eq(on,0),1.22,max(zoom-0.001,1.0))"; x="iw/2-(iw/zoom/2)"; y="ih/2-(ih/zoom/2)"
            elif cam=="pan_left": z="1.15"; x=f"(iw-iw/zoom)*(1-on/{frames})"; y="ih/2-(ih/zoom/2)"
            elif cam=="pan_right": z="1.15"; x=f"(iw-iw/zoom)*(on/{frames})"; y="ih/2-(ih/zoom/2)"
            elif cam=="slow_pull": z="if(eq(on,0),1.16,max(zoom-0.0007,1.0))"; x="iw/2-(iw/zoom/2)"; y="ih/2-(ih/zoom/2)"
            else: z="min(zoom+0.0009,1.16)"; x="iw/2-(iw/zoom/2)"; y="ih/2-(ih/zoom/2)"
            vf=f"scale={config.video_width*2}:{config.video_height*2}:force_original_aspect_ratio=increase,crop={config.video_width*2}:{config.video_height*2},zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={config.video_width}x{config.video_height}:fps={config.video_fps}"
            subprocess.run(["ffmpeg","-y","-loop","1","-i",img,"-i",aud,"-vf",vf,"-t",f"{dur:.3f}","-shortest","-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-movflags","+faststart",clip],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            clips.append(clip); print(f"[VIDEO] {job_id} | STAGE: RENDERING | scene {i}/{total} | rendered")
            if progress: progress(i,total)
        concat=os.path.join(tmp,"concat.txt")
        with open(concat,"w",encoding="utf8") as f:
            for c in clips: f.write("file '"+os.path.abspath(c).replace("'","'\\''")+"'\n")
        current=os.path.join(tmp,"base.mp4")
        subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",concat,"-c","copy","-movflags","+faststart",current],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if config.enable_subtitles:
            srt=os.path.join(tmp,"subs.srt"); t=0
            def ts(x):
                h=int(x//3600); m=int(x%3600//60); sec=x%60; return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".",",")
            with open(srt,"w",encoding="utf8") as f:
                for i,s in enumerate(scenes,1):
                    d=ffprobe_duration(os.path.join(paths.audio_dir(job_id),f"scene_{int(s['scene_id']):03d}.wav")); f.write(f"{i}\n{ts(t)} --> {ts(t+d)}\n{s['narration']}\n\n"); t+=d
            sub=os.path.join(tmp,"sub.mp4")
            subprocess.run(["ffmpeg","-y","-i",current,"-vf",f"subtitles='{srt}'","-c:a","copy","-movflags","+faststart",sub],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); current=sub
        musicdir=paths("04_AUDIO_LIBRARY","music")
        music=next((os.path.join(musicdir,f) for f in os.listdir(musicdir) if f.lower().endswith((".mp3",".wav",".m4a",".ogg"))),None) if os.path.isdir(musicdir) else None
        if config.enable_music and music:
            mixed=os.path.join(tmp,"mixed.mp4")
            subprocess.run(["ffmpeg","-y","-i",current,"-stream_loop","-1","-i",music,"-filter_complex",f"[1:a]volume={config.music_volume}[m];[0:a][m]amix=inputs=2:duration=first:dropout_transition=2[a]","-map","0:v","-map","[a]","-c:v","copy","-c:a","aac","-movflags","+faststart",mixed],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); current=mixed
        card=make_source_card(paths,job_id,research,config.video_width,config.video_height) if config.source_card_enabled else None
        final=os.path.join(tmp,"final.mp4")
        if card:
            cd=float(config.source_card_seconds)
            fc=f"[1:v]trim=duration={cd},setpts=PTS-STARTPTS[cardv];anullsrc=r=48000:cl=stereo,atrim=duration={cd},asetpts=PTS-STARTPTS[carda];[0:v][0:a][cardv][carda]concat=n=2:v=1:a=1[v][a]"
            subprocess.run(["ffmpeg","-y","-i",current,"-loop","1","-i",card,"-filter_complex",fc,"-map","[v]","-map","[a]","-c:v","libx264","-c:a","aac","-pix_fmt","yuv420p","-movflags","+faststart",final],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        else: shutil.copy2(current,final)
        final_dur=ffprobe_duration(final)
        if not float(config.min_video_seconds) <= final_dur <= float(config.max_video_seconds):
            raise RuntimeError(f"Final video duration {final_dur:.1f}s is outside configured {config.min_video_seconds}-{config.max_video_seconds}s")
        shutil.copy2(final,paths.video_render(job_id)); shutil.copy2(final,paths.video_final(job_id)); shutil.copy2(final,paths.output_video(job_id)); return paths.output_video(job_id),final_dur
    finally: shutil.rmtree(tmp,ignore_errors=True)
