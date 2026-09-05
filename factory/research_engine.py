from __future__ import annotations
import re
from urllib.parse import quote_plus
import requests
from .utils import write_json_atomic, now_iso

def _request_json(url,params=None,timeout=15):
    r=requests.get(url,params=params,timeout=timeout,headers={"User-Agent":"BlackHistoryFactory/1.0"}); r.raise_for_status(); return r.json()

def _wiki_search(topic,limit=5):
    data=_request_json("https://en.wikipedia.org/w/api.php",{"action":"query","list":"search","srsearch":topic.title,"srlimit":limit,"format":"json","utf8":1})
    out=[]
    for item in data.get("query",{}).get("search",[]):
        title=item.get("title","").strip()
        if title: out.append({"title":title,"url":"https://en.wikipedia.org/wiki/"+quote_plus(title.replace(" ","_")),"snippet":re.sub(r"<[^>]+>","",item.get("snippet","")).strip()})
    return out

def _wiki_extract(title,chars=5000):
    data=_request_json("https://en.wikipedia.org/w/api.php",{"action":"query","prop":"extracts","explaintext":1,"exchars":chars,"titles":title,"format":"json","utf8":1})
    for page in data.get("query",{}).get("pages",{}).values():
        if page.get("extract"): return page["extract"][:chars]
    return ""

def _ddg_search(topic,limit=5):
    try:
        r=requests.get("https://html.duckduckgo.com/html/",params={"q":topic.title},timeout=15,headers={"User-Agent":"Mozilla/5.0 (Black History Factory research)"}); r.raise_for_status()
    except Exception: return []
    out=[]
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',r.text,re.I|re.S):
        title=re.sub(r"<[^>]+>","",m.group(2)); title=re.sub(r"\s+"," ",title).strip(); href=m.group(1).strip()
        if title and href: out.append({"title":title,"url":href})
        if len(out)>=limit: break
    return out

def source_search(topic):
    sources=[]; seen=set()
    try:
        for item in _wiki_search(topic):
            if item["url"] in seen: continue
            seen.add(item["url"]); sources.append({**item,"source_type":"Wikipedia lead","material":_wiki_extract(item["title"])})
    except Exception as e: print(f"[RESEARCH] Wikipedia warning: {e}")
    try:
        for item in _ddg_search(topic):
            if item["url"] in seen: continue
            seen.add(item["url"]); sources.append({**item,"source_type":"Web search lead","material":item.get("snippet","")})
    except Exception as e: print(f"[RESEARCH] Web search warning: {e}")
    return sources[:10]

def _research_prompt(topic,source_material):
    return f'''You are the research historian for a short documentary about Black/African history, culture, technology, art, architecture, or mythology.

TOPIC
Title: {topic.title}
Category: {topic.category}
Region: {topic.region}
Period: {topic.period}
Angle: {topic.description}

SOURCE MATERIAL DISCOVERED BEFORE THIS REQUEST
{source_material}

Build a compact evidence dossier using information reasonably supported by the supplied source material and established historical knowledge. Do not invent citations. Clearly distinguish established historical facts, archaeological evidence, scholarly interpretation/debate, oral tradition, mythology/traditional accounts, and uncertainty.

Cover relevant details about attire/textiles, culture/customs, architecture/buildings, technology/tools, art, food/daily life, religion/belief, trade/economy, important people/places, environment, archaeology, dates, and lesser-known details when supported.

Return ONLY valid JSON:
{{
  "topic":"...","overview":"...","timeline":[],"people":[],"architecture":[],
  "attire_and_textiles":[],"culture_and_customs":[],"technology_and_tools":[],
  "daily_life_and_food":[],"religion_and_belief":[],"mythology_and_oral_tradition":[],
  "trade_and_economy":[],"art":[],"lesser_known_facts":[],"archaeological_evidence":[],
  "scholarly_debates":[],"source_supported_claims":[]
}}
Keep entries concise. Do not create a giant essay.'''

def run(paths,job_id,topic,config=None,qwen=None):
    if qwen is None: raise ValueError("Qwen client is required for research.")
    print(f"[RESEARCH] {job_id} | source search")
    sources=source_search(topic)
    if not sources: raise ValueError("No research sources were discovered.")
    material=[]
    for i,s in enumerate(sources,1): material.append(f"SOURCE {i}\nTITLE: {s.get('title','')}\nURL: {s.get('url','')}\nTYPE: {s.get('source_type','')}\nMATERIAL:\n{s.get('material','')[:5000]}")
    dossier=qwen.generate_json(_research_prompt(topic,"\n\n".join(material)),max_new_tokens=3000,retries=2)
    if not isinstance(dossier,dict): raise ValueError("Research dossier must be a JSON object.")
    dossier.update({"source_search":sources,"researched_at":now_iso(),"topic_id":topic.id})
    write_json_atomic(paths.sources(job_id),sources); write_json_atomic(paths.research(job_id),dossier)
    return dossier
