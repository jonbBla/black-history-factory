from __future__ import annotations
import base64,json,requests

def _put(repo,path,content,token,message):
    url=f"https://api.github.com/repos/{repo}/contents/{path}"
    headers={"Authorization":f"Bearer {token}","Accept":"application/vnd.github+json"}
    r=requests.get(url,headers=headers,timeout=15)
    if r.status_code not in (200,404): r.raise_for_status()
    body={"message":message,"content":base64.b64encode(json.dumps(content,indent=2).encode()).decode()}
    if r.status_code==200: body["sha"]=r.json()["sha"]
    requests.put(url,headers=headers,json=body,timeout=15).raise_for_status()

def push_status(config,current,history,token):
    if not config.github_repo or not token:return
    base=config.github_dashboard_path.rstrip('/')
    _put(config.github_repo,f"{base}/current.json",current,token,"dashboard: update current")
    _put(config.github_repo,f"{base}/history.json",history,token,"dashboard: update history")
