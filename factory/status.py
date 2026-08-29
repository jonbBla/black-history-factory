from .utils import read_json,write_json_atomic,now_iso
def set_processor(paths,name,status,job_id='',detail=''):
 data=read_json(paths.status_current,{}) or {}; data[name]={'status':status,'job_id':job_id,'detail':detail,'updated_at':now_iso()}; write_json_atomic(paths.status_current,data)
