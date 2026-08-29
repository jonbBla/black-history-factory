import os
from PIL import Image,ImageDraw,ImageFont

def make_source_card(paths,job_id,research,width,height):
    out=paths('02_JOBS',job_id,'06_video','source_card.png'); os.makedirs(os.path.dirname(out),exist_ok=True)
    img=Image.new('RGB',(width,height),(5,5,5)); d=ImageDraw.Draw(img)
    try: title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',42); font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',26)
    except: title=font=None
    sources=research.get('sources',[]) if isinstance(research,dict) else []
    lines=['SOURCES','','Documented evidence is separated from','tradition, interpretation and uncertainty.','']
    if sources:
        for i,s in enumerate(sources[:8],1):
            if isinstance(s,dict):
                label=s.get('title') or s.get('name') or s.get('citation') or 'Unnamed source'; url=s.get('url',''); lines.append(f'{i}. {label}'+(f' — {url}' if url else ''))
            else: lines.append(f'{i}. {s}')
    else: lines += ['No specific surviving written source was identified','in the research package.']
    y=260
    for i,line in enumerate(lines): d.text((80,y),line,fill='white',font=title if i==0 else font); y += 65 if i==0 else 48
    d.text((80,height-120),'Black History Factory',fill=(150,150,150),font=font); img.save(out); return out
