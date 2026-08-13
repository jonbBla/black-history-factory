// The dashboard only ever reads these two JSON files — it never talks to
// Colab directly. factory/github.py pushes fresh copies here on every
// checkpoint advance.
const CURRENT_URL = "data/current.json";
const HISTORY_URL = "data/history.json";
const POLL_MS = 15000;

async function fetchJson(url) {
  const res = await fetch(url + "?t=" + Date.now(), { cache: "no-store" });
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

function statusPillHtml(status) {
  const known = ["running", "paused", "error", "idle", "completed", "stalled"];
  const cls = known.includes(status) ? status : "idle";
  const label = status === "stalled" ? "possibly disconnected" : (status || "unknown");
  return `<span class="status-pill status-${cls}">${label}</span>`;
}

// If the last status push claims "running" but it's far older than the
// factory's own push cadence (every checkpoint advance -- at minimum once
// per image/audio file, well under a minute apart in practice), the most
// likely explanation is that Colab's session ended mid-job rather than the
// job being unusually slow. current.json has no way to know this on its
// own since nothing pushes a "the session ended" event -- so the dashboard
// infers it from silence instead, matching the spec's
// "🟡 Paused / Reason: Colab session ended" state.
const STALE_MS = 3 * 60 * 1000;

function isStale(data) {
  if (!data || !data.last_update) return false;
  if (data.status !== "running") return false;
  const age = Date.now() - new Date(data.last_update).getTime();
  return age > STALE_MS;
}

function renderCurrent(data) {
  const el = document.getElementById("current-content");
  if (!data || data.status === "idle" || !data.job_id) {
    el.innerHTML = `<p class="muted">No job running right now. ${statusPillHtml("idle")}</p>`;
    return;
  }
  const stalled = isStale(data);
  const displayStatus = stalled ? "stalled" : data.status;
  el.innerHTML = `
    <p><strong>${data.title || data.job_id}</strong> ${statusPillHtml(displayStatus)}</p>
    <div class="progress-bar-track">
      <div class="progress-bar-fill" style="width:${data.percent || 0}%"></div>
    </div>
    <div class="stage-row">
      <span>Stage: ${data.stage || "-"}</span>
      <span>${data.current ?? 0}/${data.total ?? 0} (${data.percent ?? 0}%)</span>
    </div>
    ${data.error ? `<p class="muted" style="color:var(--red)">Error: ${data.error}</p>` : ""}
    ${stalled ? `<p class="muted" style="color:var(--yellow)">
        Last checkpoint: ${data.stage || "-"}${data.current ? ` (scene ${data.current}/${data.total})` : ""}
        — no update since ${new Date(data.last_update).toLocaleString()}.
        Reason: Colab session likely ended.
      </p>` : ""}
    <div class="stage-row">
      <span>Colab: ${stalled ? "paused" : (data.colab_status || "-")}</span>
      <span>Updated: ${data.last_update ? new Date(data.last_update).toLocaleString() : "-"}</span>
    </div>
  `;
}

function renderLatest(history) {
  const el = document.getElementById("latest-content");
  if (!history || history.length === 0) {
    el.innerHTML = `<p class="muted">No completed videos yet.</p>`;
    return;
  }
  const latest = history[0];
  el.innerHTML = `
    ${latest.thumbnail ? `<img class="thumb" src="${latest.thumbnail}" alt="${latest.title}">` : ""}
    <p><strong>${latest.title}</strong></p>
    <p class="muted">${latest.date || ""}</p>
    ${latest.video ? `<a class="btn" href="${latest.video}" target="_blank" rel="noopener">▶ Watch</a>
    <a class="btn" href="${latest.video}" download>↓ Download</a>` : ""}
  `;
}

function renderHistory(history) {
  const el = document.getElementById("history-list");
  if (!history || history.length <= 1) {
    el.innerHTML = `<li class="muted">Nothing else yet.</li>`;
    return;
  }
  el.innerHTML = history.slice(1).map(h => `
    <li>
      <span>${h.title}</span>
      <span class="muted">${h.date || ""}</span>
    </li>
  `).join("");
}

async function refresh() {
  try {
    const current = await fetchJson(CURRENT_URL);
    renderCurrent(current);
  } catch (e) {
    document.getElementById("current-content").innerHTML =
      `<p class="muted">Couldn't load status yet.</p>`;
  }
  try {
    const history = await fetchJson(HISTORY_URL);
    renderLatest(history);
    renderHistory(history);
  } catch (e) {
    document.getElementById("latest-content").innerHTML =
      `<p class="muted">Couldn't load history yet.</p>`;
  }
}

refresh();
setInterval(refresh, POLL_MS);
