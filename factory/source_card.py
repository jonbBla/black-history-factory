from __future__ import annotations
import os, textwrap
from PIL import Image, ImageDraw, ImageFont

def make_source_card(paths, job_id, research, width, height):
    out=os.path.join(paths.video_dir(job_id),"source_card.png"); os.makedirs(os.path.dirname(out),exist_ok=True)
    img=Image.new("RGB",(width,height),(0,0,0)); d=ImageDraw.Draw(img)
    try:
        title=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",48)
        font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",27)
    except Exception: title=font=ImageFont.load_default()
    lines=["SOURCES","","Documented evidence is separated from tradition, interpretation and uncertainty.",""]
    sources=research.get("sources",[]) if isinstance(research,dict) else []
    if sources:
        for i,s in enumerate(sources[:8],1):
            if isinstance(s,dict): lines.append(f"{i}. {s.get('title') or s.get('name') or s.get('citation') or 'Unnamed source'}" + (f" — {s.get('url')}" if s.get('url') else ""))
            else: lines.append(f"{i}. {s}")
    else: lines.append("No specific source was identified in the research package.")
    y=180
    for i,line in enumerate(lines):
        for wrapped in textwrap.wrap(line,width=52) or [""]:
            d.text((70,y),wrapped,fill="white",font=title if i==0 else font); y += 65 if i==0 else 44
    d.text((70,height-100),"Black History Factory",fill=(150,150,150),font=font); img.save(out); return out
