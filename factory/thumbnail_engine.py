from __future__ import annotations
import os, shutil, textwrap

def run(paths,job_id,source_image,title=""):
    out=paths.thumbnail(job_id); os.makedirs(os.path.dirname(out),exist_ok=True)
    if not source_image or not os.path.exists(source_image): raise FileNotFoundError("Thumbnail source image missing")
    try:
        from PIL import Image,ImageDraw,ImageFont
        img=Image.open(source_image).convert("RGB"); w,h=img.size; draw=ImageDraw.Draw(img)
        try: font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",max(24,w//20))
        except Exception: font=ImageFont.load_default()
        y=h-int(h*.18)
        for line in textwrap.wrap(title or "",width=22): draw.text((int(w*.06),y),line,font=font,fill="white"); y+=getattr(font,"size",30)+8
        img.save(out,"PNG"); return out
    except Exception: shutil.copy2(source_image,out); return out
