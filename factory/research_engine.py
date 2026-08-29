import json
from .utils import write_json_atomic,now_iso
VALID={'established_fact','archaeological_evidence','scholarly_interpretation','oral_tradition','mythology','uncertain'}
def build_prompt(topic):
    return f'''Research this topic for a historically responsible short documentary. Topic: {topic.title}. Category: {topic.category}. Region: {topic.region}. Period: {topic.period}. Angle: {topic.description}.\n\nReturn JSON with overview, timeline, people, architecture, technology, daily_life, religion, mythology, trade, art, lesser_known_facts, archaeological_evidence, scholarly_debates, and sources. Every claim item must include classification from {sorted(VALID)}. Never invent sources. Clearly separate mythology/oral tradition from established history. If evidence is weak, say so.'''
def normalize(d,topic):
    d=d if isinstance(d,dict) else {}; keys=['topic','overview','timeline','people','architecture','technology','daily_life','religion','mythology','trade','art','lesser_known_facts','archaeological_evidence','scholarly_debates','sources']
    out={k:d.get(k,'' if k in ('topic','overview') else []) for k in keys}; out['topic']=topic.title
    for k,v in out.items():
        if isinstance(v,list):
            for item in v:
                if isinstance(item,dict) and item.get('classification') not in VALID: item['classification']='uncertain'
    return out
def run(paths,job_id,topic,qwen):
    out=normalize(qwen.generate_json(build_prompt(topic),max_new_tokens=3200),topic); out['_generated_at']=now_iso(); write_json_atomic(paths.research(job_id),out); write_json_atomic(paths.sources(job_id),out.get('sources',[])); return out
