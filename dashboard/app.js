const CURRENT_URL = "data/current.json";
const HISTORY_URL = "data/history.json";
const POLL_MS = 15000;
const STALE_MS = 5 * 60 * 1000;

async function fetchJson(url) {
  const res = await fetch(url + "?t=" + Date.now(), {cache:"no-store"});
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

function statusPillHtml(status) {
  const known=["running","paused","error","idle","completed","warning","stalled"];
  const cls=known.includes(status)?status:"idle";
  return `<span class="status-pill status-${cls}">${status || "unknown"}</span>`;
}

function processorRows(data) {
  const processors=data?.processors || {};
  return Object.entries(processors).map(([name,p]) => {
    const stale=p.status==="running" && p.updated_at && Date.now()-new Date(p.updated_at).getTime()>STALE_MS;
    const status=stale?"stalled":p.status;
    return `<div class="stage-row"><span><strong>${name}</strong> — ${p.stage || "-"} ${statusPillHtml(status)}</span><span>${p.detail || ""} ${p.total ? `${p.completed}/${p.total}` : ""}</span></div>`;
  }).join("");
}

function renderCurrent(data) {
  const el=document.getElementById("current-content");
  if(!data){el.innerHTML='<p class="muted">No status data.</p>';return;}
  el.innerHTML=`<div class="stage-row"><span>Factory status</span><span>${data.updated_at ? new Date(data.updated_at).toLocaleString() : "-"}</span></div>${processorRows(data) || '<p class="muted">No processors reporting yet.</p>'}`;
}

function renderLatest(history) {
  const el=document.getElementById("latest-content");
  if(!history?.length){el.innerHTML='<p class="muted">No completed videos yet.</p>';return;}
  const latest=history[0];
  el.innerHTML=`${latest.thumbnail ? `<img class="thumb" src="${latest.thumbnail}" alt="${latest.title}">` : ""}<p><strong>${latest.title || latest.job_id}</strong></p><p class="muted">${latest.date || ""}${latest.seconds ? ` · ${latest.seconds.toFixed(1)}s` : ""}</p>${latest.video ? `<a class="btn" href="${latest.video}" target="_blank" rel="noopener">▶ Watch</a> <a class="btn" href="${latest.video}" download>↓ Download</a>` : ""}`;
}

function renderHistory(history) {
  const el=document.getElementById("history-list");
  if(!history || history.length<=1){el.innerHTML='<li class="muted">Nothing else yet.</li>';return;}
  el.innerHTML=history.slice(1).map(h=>`<li><span>${h.title || h.job_id}</span><span class="muted">${h.date || ""}</span></li>`).join("");
}

async function refresh(){
  try{renderCurrent(await fetchJson(CURRENT_URL));}catch(e){document.getElementById("current-content").innerHTML='<p class="muted">Couldn’t load status yet.</p>';}
  try{const h=await fetchJson(HISTORY_URL);renderLatest(h);renderHistory(h);}catch(e){document.getElementById("latest-content").innerHTML='<p class="muted">Couldn’t load history yet.</p>';}
}
refresh(); setInterval(refresh,POLL_MS);
