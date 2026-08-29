import os,subprocess
from .utils import write_json_atomic

def load_piper_voice(model_path):
    from piper import PiperVoice
    return PiperVoice.load(model_path,use_cuda=True)
def synthesize(voice,text,out):
    import wave
    with wave.open(out,'wb') as f: voice.synthesize(text,f)
def run(paths,job_id,scenes,voice,config):
    out=paths.audio_dir(job_id); os.makedirs(out,exist_ok=True); narr=[]
    for s in scenes:
        p=os.path.join(out,f"scene_{int(s['scene_id']):03d}.wav")
        if not os.path.exists(p): synthesize(voice,s['narration'],p)
        narr.append(p)
    concat=os.path.join(out,'narration_concat.txt'); final=os.path.join(out,'final_mix.wav')
    if not os.path.exists(final):
        with open(concat,'w',encoding='utf8') as f:
            for p in narr: f.write("file '"+p.replace("'","'\\''")+"'\n")
        subprocess.run(['ffmpeg','-y','-f','concat','-safe','0','-i',concat,'-c','copy',final],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    write_json_atomic(paths.state(job_id,'audio'),{'status':'complete','scene_count':len(narr)}); return final
