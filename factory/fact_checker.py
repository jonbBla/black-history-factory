from __future__ import annotations
import json,re
MAX_ATTEMPTS=4
ALLOWED={"verified","supported","uncertain","unsupported","false","misleading"}

def _extract_json(text):
    text=(text or "").strip(); text=re.sub(r"^```(?:json)?\s*","",text,flags=re.I); text=re.sub(r"\s*```$","",text)
    try:return json.loads(text)
    except Exception:pass
    for start,ch in enumerate(text):
        if ch not in "{[": continue
        stack=[]; ins=False; esc=False
        for i in range(start,len(text)):
            c=text[i]
            if ins:
                if esc: esc=False
                elif c=='\\': esc=True
                elif c=='"': ins=False
                continue
            if c=='"': ins=True
            elif c in "{[": stack.append(c)
            elif c in "}]":
                if not stack or (c=='}' and stack[-1]!='{') or (c==']' and stack[-1]!='['): break
                stack.pop()
                if not stack:
                    try:return json.loads(text[start:i+1])
                    except Exception: break
    raise ValueError("No valid JSON object found in model response.")

def _prompt(topic,research):
    sources=research.get("source_search",[]) if isinstance(research,dict) else []
    source_lines=[]
    for i,s in enumerate(sources,1): source_lines.append(f"SOURCE {i}: {s.get('title','')} | {s.get('url','')}\n{s.get('material','')[:3500]}")
    dossier=dict(research); dossier.pop("source_search",None); dossier.pop("researched_at",None); dossier.pop("topic_id",None)
    return f'''You are a strict historical fact checker.

TOPIC: {topic.title}

SOURCE MATERIAL
{chr(10).join(source_lines)}

EVIDENCE DOSSIER
{json.dumps(dossier,ensure_ascii=False,indent=2)}

Check the important claims in the dossier against the supplied source material. Do not invent sources. Traditional or mythological claims can be valid cultural evidence, but do not label them independently verified history without supporting evidence.

Return ONLY this compact JSON:
{{
  "overall_status":"PASS" or "REVIEW",
  "claims":[{{"claim":"short claim","classification":"verified|supported|uncertain|unsupported|false|misleading","evidence":"short reason","source":"SOURCE 1 or SOURCE 2 or NONE"}}],
  "summary":"short overall assessment"
}}
Check important claims, not every sentence. Keep the response compact.'''

def _validate(data):
    if not isinstance(data,dict) or not isinstance(data.get("claims"),list): raise ValueError("Fact-check result must contain a claims list.")
    if data.get("overall_status") not in {"PASS","REVIEW"}: raise ValueError("overall_status must be PASS or REVIEW.")
    for i,c in enumerate(data["claims"],1):
        if not isinstance(c,dict): raise ValueError(f"Claim {i} is not an object.")
        for k in ("claim","classification","evidence","source"):
            if not str(c.get(k,"")).strip(): raise ValueError(f"Claim {i} missing {k}.")
        if c["classification"] not in ALLOWED: raise ValueError(f"Claim {i} has invalid classification.")

def run(paths,job_id,topic,research=None,config=None,qwen=None):
    if qwen is None: raise ValueError("Qwen client is required for fact checking.")
    if not isinstance(research,dict): raise ValueError("Research dossier is missing or invalid.")
    prompt=_prompt(topic,research); last=None
    for attempt in range(1,MAX_ATTEMPTS+1):
        print(f"[FACT_CHECK] ATTEMPT {attempt}/{MAX_ATTEMPTS}")
        p=prompt if last is None else prompt+f"\n\nPrevious error: {last}\nRegenerate the COMPLETE compact JSON from scratch."
        raw=qwen.generate(p,max_new_tokens=1600,temperature=0.15)
        try:
            data=_extract_json(raw); _validate(data); data["checked_claim_count"]=len(data["claims"]); return data
        except Exception as e: last=str(e); print(f"[FACT_CHECK] FAILED | {last}")
    raise ValueError(f"Fact-check failed after {MAX_ATTEMPTS} attempts: {last}")
