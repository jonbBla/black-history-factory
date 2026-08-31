from dataclasses import dataclass,asdict
import os
from .utils import read_json,write_json_atomic,now_iso
@dataclass
class Topic:
 id:str; title:str; category:str; region:str; period:str=''; description:str=''; aliases:list=None; used:bool=False
 def __post_init__(self): self.aliases=self.aliases or []
 def to_dict(self): return asdict(self)
def load_topics(paths): return [Topic(**x) for x in read_json(paths.topics_json,[]) if isinstance(x,dict)]
def _next_job_id(paths):
 nums=[]
 for n in os.listdir(paths('02_JOBS')) if os.path.isdir(paths('02_JOBS')) else []:
  if n.startswith('BH'):
   try: nums.append(int(n[2:]))
   except: pass
 return f'BH{max(nums,default=0)+1:06d}'
def claim_next_topic(paths,processor='qwen'):
 topics=load_topics(paths); claimed=set()
 for n in os.listdir(paths('02_JOBS')) if os.path.isdir(paths('02_JOBS')) else []:
  d=read_json(paths.manifest(n),{})
  if d and d.get('topic_id'): claimed.add(d['topic_id'])
 for t in topics:
  if t.used or t.id in claimed: continue
  job_id=_next_job_id(paths); os.makedirs(paths.job(job_id,'state'),exist_ok=True)
  write_json_atomic(paths.manifest(job_id),{'job_id':job_id,'topic_id':t.id,'title':t.title,'created_at':now_iso(),'claimed_by':processor,'status':'QWEN_RESEARCHING'})
  write_json_atomic(paths.state(job_id,'qwen'),{'status':'claimed','updated_at':now_iso(),'processor':processor})
  return t,job_id
 return None,None
def mark_used(paths,topic):
 used=read_json(paths.used_topics_json,[]) or []
 if not any(x.get('id')==topic.id for x in used): used.append(topic.to_dict())
 write_json_atomic(paths.used_topics_json,used)
 topics=load_topics(paths); write_json_atomic(paths.topics_json,[t.to_dict() for t in topics])
