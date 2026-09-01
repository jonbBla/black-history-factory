from __future__ import annotations
from dataclasses import dataclass,asdict
import os,re
from .utils import read_json,write_json_atomic,now_iso
@dataclass
class Topic:
    id:str; title:str; category:str; region:str; period:str=''; description:str=''; aliases:list=None; used:bool=False
    def __post_init__(self): self.aliases=self.aliases or []
    def to_dict(self): return asdict(self)
def load_topics(paths):
    out=[]
    for x in read_json(paths.topics_json,[]) or []:
        if isinstance(x,dict):
            try: out.append(Topic(**x))
            except TypeError: pass
    return out
def _next_job_id(paths):
    nums=[]
    for n in os.listdir(paths('02_JOBS')) if os.path.isdir(paths('02_JOBS')) else []:
        m=re.fullmatch(r'BH(\d+)',n)
        if m: nums.append(int(m.group(1)))
    return f'BH{max(nums,default=0)+1:06d}'
def find_resumable_job(paths):
    topics={t.id:t for t in load_topics(paths)}
    for jid in sorted(os.listdir(paths('02_JOBS'))) if os.path.isdir(paths('02_JOBS')) else []:
        if not jid.startswith('BH'): continue
        m=read_json(paths.manifest(jid),{}) or {}; st=(read_json(paths.state(jid,'qwen'),{}) or {}).get('status','')
        if m.get('claimed_by')=='qwen' and st not in {'QWEN_READY','COMPLETED','FAILED','REJECTED'} and m.get('topic_id') in topics:
            return topics[m['topic_id']],jid
    return None,None
def claim_next_topic(paths):
    claimed=set()
    for jid in os.listdir(paths('02_JOBS')) if os.path.isdir(paths('02_JOBS')) else []:
        m=read_json(paths.manifest(jid),{}) or {}; st=(read_json(paths.state(jid,'qwen'),{}) or {}).get('status','')
        if m.get('topic_id') and st not in {'FAILED','REJECTED'}: claimed.add(m['topic_id'])
    for topic in load_topics(paths):
        if topic.used or topic.id in claimed: continue
        jid=_next_job_id(paths); os.makedirs(paths.job(jid),exist_ok=True); os.makedirs(os.path.join(paths.job(jid),'state'),exist_ok=True)
        write_json_atomic(paths.manifest(jid),{"job_id":jid,"topic_id":topic.id,"title":topic.title,"category":topic.category,"region":topic.region,"period":topic.period,"claimed_by":"qwen","created_at":now_iso(),"status":"QWEN_RESEARCHING"})
        write_json_atomic(paths.state(jid,'qwen'),{"status":"CLAIMED","processor":"qwen","updated_at":now_iso()})
        return topic,jid
    return None,None
def update_status(paths,jid,status,**extra):
    m=read_json(paths.manifest(jid),{}) or {}; m.update(status=status,updated_at=now_iso(),**extra); write_json_atomic(paths.manifest(jid),m)
    write_json_atomic(paths.state(jid,'qwen'),{"status":status,"processor":"qwen","updated_at":now_iso(),**extra})
def mark_used(paths,topic):
    used=read_json(paths.used_topics_json,[]) or []
    if not any(x.get('id')==topic.id for x in used): used.append(topic.to_dict())
    write_json_atomic(paths.used_topics_json,used)
    topics=load_topics(paths)
    for t in topics:
        if t.id==topic.id:t.used=True
    write_json_atomic(paths.topics_json,[t.to_dict() for t in topics])
