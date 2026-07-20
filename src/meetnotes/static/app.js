/* MeetNotes SPA — hash routing, live transcript over WebSocket. */
"use strict";

const app = document.getElementById("app");
const banner = document.getElementById("banner");

const state = {
  route: null,
  liveMeetingId: null, // meeting currently shown in the live view
  captureState: "idle",
  startedAt: null,
  timerHandle: null,
  autoScroll: true,
  bannerSticky: false,
  levelTesting: false,
  noteLines: [],      // [{ts_ms, text}] — per-line timestamps for the notepad
  noteDirty: false,
  noteSaveHandle: null,
  copilotBusy: false,
};

function md2html(md) {
  // Tiny escaping-first markdown renderer: headings, bold/italic, code, lists.
  const out = [];
  let list = null; // "ul" | "ol" | null
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  const inline = (s) =>
    esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
  for (const raw of String(md || "").split("\n")) {
    const line = raw.trimEnd();
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    const ul = line.match(/^\s*[-*]\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (h) {
      closeList();
      out.push(`<h${h[1].length + 1}>${inline(h[2])}</h${h[1].length + 1}>`);
    } else if (ul || ol) {
      const kind = ul ? "ul" : "ol";
      if (list !== kind) { closeList(); out.push(`<${kind}>`); list = kind; }
      out.push(`<li>${inline((ul || ol)[1])}</li>`);
    } else if (line.trim() === "") {
      closeList();
    } else {
      closeList();
      out.push(`<p>${inline(line)}</p>`);
    }
  }
  closeList();
  return out.join("\n");
}

// ---- helpers ---------------------------------------------------------------

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : (body.detail?.message || detail);
      const err = new Error(detail);
      err.code = body.detail?.code || null;
      throw err;
    } catch (e) {
      if (e instanceof Error && e.code !== undefined) throw e;
      throw new Error(detail);
    }
  }
  return res.json();
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function mmss(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function fmtDate(epochMs) {
  return new Date(epochMs).toLocaleString([], {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

function showBanner(text, isError = false, sticky = false) {
  banner.textContent = text;
  banner.classList.toggle("error", isError);
  banner.classList.remove("hidden");
  state.bannerSticky = sticky;
}

function hideBanner() {
  banner.classList.add("hidden");
  state.bannerSticky = false;
}

// ---- websocket -------------------------------------------------------------

let ws = null;

function connectWS() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    handleEvent(msg);
  };
  ws.onclose = () => setTimeout(connectWS, 1000);
}

function handleEvent(msg) {
  const d = msg.data || {};
  if (msg.type === "capture.status") {
    onCaptureStatus(d);
  } else if (msg.type === "transcript.interim") {
    if (d.meeting_id === state.liveMeetingId) renderInterim(d);
  } else if (msg.type === "transcript.final") {
    if (d.meeting_id === state.liveMeetingId) appendFinal(d);
  } else if (msg.type === "meeting.ended") {
    if (d.meeting_id === state.liveMeetingId) location.hash = `#/meeting/${d.meeting_id}`;
  } else if (msg.type === "copilot.card") {
    if (d.meeting_id === state.liveMeetingId) addCopilotCard(d);
  } else if (msg.type === "copilot.status") {
    if (d.meeting_id === state.liveMeetingId) onCopilotStatus(d);
  } else if (msg.type === "note.status") {
    const noteSlot = document.getElementById("note-section");
    if (noteSlot && noteSlot.dataset.meetingId === d.meeting_id) {
      renderNoteSection(d.meeting_id);
    }
  } else if (msg.type === "audio.level") {
    const bar = document.getElementById("level-bar");
    if (bar) {
      bar.style.width = `${Math.round(d.level * 100)}%`;
      bar.classList.toggle("hot", d.level > 0.02);
    }
  }
}

function onCaptureStatus(d) {
  state.captureState = d.state;
  if (d.state === "reconnecting") {
    const buffered = d.buffered_s != null ? ` Buffered: ${d.buffered_s}s of audio.` : "";
    showBanner(`Reconnecting to transcription… ${d.message || ""}${buffered}`, false, true);
  } else if (d.state === "error") {
    showBanner(d.message || "Something went wrong.", true, true);
    if (d.code === "mic_silent") {
      const help = document.getElementById("mic-help");
      if (help) help.classList.remove("hidden");
    }
  } else if (d.code === "buffer_warning") {
    showBanner(d.message);
  } else {
    hideBanner();
  }
  const dot = document.getElementById("rec-dot");
  if (dot && (d.state === "recording" || d.state === "paused")) {
    dot.classList.toggle("paused", d.state === "paused");
    dot.lastChild.textContent = d.state === "paused" ? " Paused" : " Recording";
    const pauseBtn = document.getElementById("btn-pause");
    if (pauseBtn) pauseBtn.textContent = d.state === "paused" ? "Resume" : "Pause";
  }
}

// ---- views -----------------------------------------------------------------

async function renderHome() {
  stopTimer();
  state.liveMeetingId = null;
  const meetings = await api("/meetings");
  const items = meetings.map((m) => {
    const dur = m.ended_at && m.started_at ? mmss(m.ended_at - m.started_at) : "—";
    const live = m.status !== "ended" ? " · live" : "";
    const target = m.status !== "ended" ? `#/live/${m.id}` : `#/meeting/${m.id}`;
    const trash = m.status === "ended"
      ? `<button class="trash" data-id="${esc(m.id)}" data-title="${esc(m.title || "Untitled")}" title="Delete meeting">🗑</button>`
      : "";
    return `<li><a href="${target}">
      <span>${esc(m.title || "Untitled")}${live}</span>
      <span class="meta">${m.started_at ? fmtDate(m.started_at) : ""} · ${dur} · ${m.segment_count} segments</span>
    </a>${trash}</li>`;
  }).join("");
  app.innerHTML = `
    <div class="page-head">
      <h1>Meetings</h1>
      <button class="primary" id="btn-new">New meeting</button>
    </div>
    ${items ? `<ul class="meeting-list">${items}</ul>` : `<p class="empty">No meetings yet. Start one!</p>`}
  `;
  document.getElementById("btn-new").onclick = startMeeting;
  for (const btn of document.querySelectorAll(".trash")) {
    btn.onclick = (e) => {
      e.preventDefault();
      deleteMeeting(btn.dataset.id, btn.dataset.title, renderHome);
    };
  }
}

async function renameMeeting(meetingId, currentTitle, after) {
  const title = prompt("Meeting title:", currentTitle || "");
  if (title === null || !title.trim() || title.trim() === currentTitle) return;
  try {
    await api(`/meetings/${meetingId}/title`, {
      method: "PUT",
      body: JSON.stringify({ title: title.trim() }),
    });
    if (after) after();
  } catch (e) {
    showBanner(e.message, true);
  }
}

async function deleteMeeting(meetingId, title, after) {
  if (!confirm(`Delete "${title}"?\n\nThis permanently removes its transcript and notes.`)) return;
  try {
    await api(`/meetings/${meetingId}`, { method: "DELETE" });
    if (after) after();
  } catch (e) {
    showBanner(e.message, true);
  }
}

async function startMeeting() {
  try {
    const meeting = await api("/meetings", { method: "POST", body: JSON.stringify({}) });
    location.hash = `#/live/${meeting.id}`;
  } catch (e) {
    if (e.code === "no_key") {
      location.hash = "#/settings";
      showBanner("Add your Deepgram API key first, then start a meeting.");
    } else if (e.code === "mic_error") {
      renderMicHelp(e.message);
    } else {
      showBanner(e.message, true);
    }
  }
}

function micHelpHTML(message) {
  return `
    <div class="help-card" id="mic-help">
      <h2>Microphone access needed</h2>
      ${message ? `<p>${esc(message)}</p>` : ""}
      <p>macOS blocks microphone access until you allow it:</p>
      <ol>
        <li>Open <strong>System Settings → Privacy &amp; Security → Microphone</strong></li>
        <li>Enable access for your terminal (e.g. Terminal, iTerm) or Python</li>
        <li>Restart MeetNotes and try again</li>
      </ol>
    </div>`;
}

function renderMicHelp(message) {
  app.innerHTML = micHelpHTML(message) + `<a href="#/">Back to meetings</a>`;
}

async function renderLive(meetingId) {
  const { meeting, segments } = await api(`/meetings/${meetingId}/transcript`);
  if (meeting.status === "ended") {
    location.hash = `#/meeting/${meetingId}`;
    return;
  }
  state.liveMeetingId = meetingId;
  state.startedAt = meeting.started_at;
  state.autoScroll = true;
  const paused = meeting.status === "paused";
  app.innerHTML = `
    <div class="live-controls">
      <span class="live-title" id="live-title">${esc(meeting.title || "Untitled")}</span>
      <button class="trash" id="btn-rename-live" title="Rename meeting">✏️</button>
      <span class="rec-dot ${paused ? "paused" : ""}" id="rec-dot"><span>${paused ? " Paused" : " Recording"}</span></span>
      <span class="timer" id="timer">00:00</span>
      <button id="btn-pause">${paused ? "Resume" : "Pause"}</button>
      <button class="danger" id="btn-end">End meeting</button>
    </div>
    <div class="help-card hidden" id="mic-help-slot"></div>
    <div class="live-grid ${localStorage.getItem("copilotCollapsed") === "1" ? "copilot-hidden" : ""}" id="live-grid">
      <div class="transcript" id="transcript">
        <div id="finals"></div>
        <div class="line interim" id="interim"></div>
        <button class="jump-live hidden" id="jump-live">Jump to live ↓</button>
      </div>
      <div class="notepad-wrap">
        <label for="notepad">Your notes <span class="note" id="note-save-state"></span></label>
        <textarea id="notepad" class="notepad"
          placeholder="Jot rough notes here — they'll be expanded into a full note when the meeting ends."></textarea>
      </div>
      <div class="copilot-wrap" id="copilot-wrap">
        <div class="copilot-head">
          <label>Copilot</label>
          <span>
            <button id="btn-copilot-help" title="⌘K">Help me here</button>
            <button id="btn-copilot-toggle" title="Collapse sidebar">⇥</button>
          </span>
        </div>
        <div class="copilot-cards" id="copilot-cards">
          <p class="note" id="copilot-empty">Context from your knowledge base will appear
            here as the conversation touches on it.</p>
        </div>
      </div>
      <button class="copilot-expand ${localStorage.getItem("copilotCollapsed") === "1" ? "" : "hidden"}"
        id="btn-copilot-expand" title="Show copilot">✨</button>
    </div>
  `;
  document.getElementById("btn-copilot-help").onclick = copilotHelp;
  document.getElementById("btn-copilot-toggle").onclick = () => toggleCopilot(true);
  document.getElementById("btn-copilot-expand").onclick = () => toggleCopilot(false);
  document.getElementById("btn-rename-live").onclick = () =>
    renameMeeting(meetingId, meeting.title || "", async () => {
      const fresh = await api(`/meetings/${meetingId}/transcript`);
      meeting.title = fresh.meeting.title;
      const el = document.getElementById("live-title");
      if (el) el.textContent = fresh.meeting.title || "Untitled";
    });
  await setupNotepad(meetingId);
  document.getElementById("mic-help-slot").outerHTML = micHelpHTML("").replace(
    'class="help-card"', 'class="help-card hidden"');
  const finals = document.getElementById("finals");
  for (const s of segments) finals.insertAdjacentHTML("beforeend", finalLineHTML(s));

  const pane = document.getElementById("transcript");
  pane.scrollTop = pane.scrollHeight;
  pane.addEventListener("scroll", () => {
    const nearBottom = pane.scrollTop + pane.clientHeight >= pane.scrollHeight - 60;
    state.autoScroll = nearBottom;
    document.getElementById("jump-live").classList.toggle("hidden", nearBottom);
  });
  document.getElementById("jump-live").onclick = () => {
    pane.scrollTop = pane.scrollHeight;
    state.autoScroll = true;
  };
  document.getElementById("btn-pause").onclick = async () => {
    const isPaused = document.getElementById("rec-dot").classList.contains("paused");
    await api(`/meetings/${meetingId}/${isPaused ? "resume" : "pause"}`, { method: "POST" });
  };
  document.getElementById("btn-end").onclick = async () => {
    document.getElementById("btn-end").disabled = true;
    try {
      state.noteDirty = true;
      await saveNotepad(meetingId); // flush the notepad before generation kicks off
      stopNotepad();
      await api(`/meetings/${meetingId}/end`, { method: "POST" });
      location.hash = `#/meeting/${meetingId}`;
    } catch (e) {
      showBanner(e.message, true);
      document.getElementById("btn-end").disabled = false;
    }
  };
  startTimer();
}

function finalLineHTML(s) {
  const who = s.channel === 1 ? "them" : "me";
  const chip = `<span class="chan ${who}">${who === "them" ? "Them" : "Me"}</span>`;
  const spk = s.speaker != null ? `<span class="spk">[speaker ${s.speaker}]</span>` : "";
  return `<p class="line"><span class="ts">${mmss(s.start_ms)}</span>${chip}${spk}${esc(s.text)}</p>`;
}

function appendFinal(d) {
  const finals = document.getElementById("finals");
  if (!finals) return;
  finals.insertAdjacentHTML("beforeend", finalLineHTML(d));
  const interim = document.getElementById("interim");
  if (interim) interim.textContent = "";
  maybeScroll();
}

function renderInterim(d) {
  const interim = document.getElementById("interim");
  if (!interim) return;
  interim.innerHTML = `<span class="ts">${mmss(d.start_ms)}</span>${esc(d.text)}`;
  maybeScroll();
}

function maybeScroll() {
  if (!state.autoScroll) return;
  const pane = document.getElementById("transcript");
  if (pane) pane.scrollTop = pane.scrollHeight;
}

// ---- copilot sidebar -------------------------------------------------------

function toggleCopilot(collapse) {
  localStorage.setItem("copilotCollapsed", collapse ? "1" : "0");
  const grid = document.getElementById("live-grid");
  const expand = document.getElementById("btn-copilot-expand");
  if (grid) grid.classList.toggle("copilot-hidden", collapse);
  if (expand) expand.classList.toggle("hidden", !collapse);
}

async function copilotHelp() {
  const btn = document.getElementById("btn-copilot-help");
  try {
    if (btn) { btn.disabled = true; btn.textContent = "Thinking…"; }
    await api("/copilot/trigger", { method: "POST" });
  } catch (e) {
    showBanner(e.message, true);
    if (btn) { btn.disabled = false; btn.textContent = "Help me here"; }
  }
}

function onCopilotStatus(d) {
  const btn = document.getElementById("btn-copilot-help");
  if (!btn) return;
  if (d.state === "thinking") {
    btn.disabled = true;
    btn.textContent = "Thinking…";
  } else {
    btn.disabled = false;
    btn.textContent = "Help me here";
    if (d.state === "error" && d.message) showBanner(d.message, true);
  }
}

function addCopilotCard(d) {
  const cardsEl = document.getElementById("copilot-cards");
  if (!cardsEl) return;
  const empty = document.getElementById("copilot-empty");
  if (empty) empty.remove();
  const badge = d.source === "meeting" ? "past meeting" : "doc";
  const card = document.createElement("div");
  card.className = "copilot-card";
  card.dataset.cardId = d.id;
  card.innerHTML = `
    <div class="copilot-card-head">
      <strong>${esc(d.doc_title)}</strong>
      <span class="chan">${esc(badge)}</span>
    </div>
    ${d.heading ? `<div class="meta">${esc(d.heading)}</div>` : ""}
    <p class="copilot-snippet">${esc(d.snippet)}${d.page ? ` <span class="meta">(p.${d.page})</span>` : ""}</p>
    <div class="form-line">
      <button class="primary" data-act="insert">Insert into notes</button>
      <button data-act="dismiss">Dismiss</button>
    </div>`;
  card.querySelector('[data-act="insert"]').onclick = async () => {
    try {
      const res = await api(`/copilot/cards/${d.id}/insert`, { method: "POST" });
      appendNoteLine(res.line, res.ts_ms);
      card.remove();
    } catch (e) { showBanner(e.message, true); }
  };
  card.querySelector('[data-act="dismiss"]').onclick = async () => {
    card.remove();
    try { await api(`/copilot/cards/${d.id}/dismiss`, { method: "POST" }); } catch {}
  };
  cardsEl.prepend(card);
}

function appendNoteLine(line, tsMs) {
  const pad = document.getElementById("notepad");
  if (!pad) return;
  const needsNewline = pad.value.length > 0 && !pad.value.endsWith("\n");
  pad.value += (needsNewline ? "\n" : "") + line;
  // Keep the per-line timestamp map aligned: the server already saved this
  // line, so mirror it locally with the server's timestamp.
  state.noteLines = pad.value.split("\n").map((text, i) => {
    const prev = state.noteLines[i];
    if (prev && prev.text === text) return prev;
    return { ts_ms: tsMs, text };
  });
  pad.scrollTop = pad.scrollHeight;
}

// ---- notepad (autosave every 2s, per-line timestamps) ----------------------

async function setupNotepad(meetingId) {
  const pad = document.getElementById("notepad");
  if (!pad) return;
  try {
    const { lines } = await api(`/meetings/${meetingId}/rough-notes`);
    state.noteLines = lines;
    pad.value = lines.map((l) => l.text).join("\n");
  } catch { state.noteLines = []; }
  state.noteDirty = false;
  pad.addEventListener("input", () => { state.noteDirty = true; });
  if (state.noteSaveHandle) clearInterval(state.noteSaveHandle);
  state.noteSaveHandle = setInterval(() => saveNotepad(meetingId), 2000);
}

function currentNoteLines() {
  const pad = document.getElementById("notepad");
  if (!pad) return state.noteLines;
  const elapsed = state.startedAt ? Date.now() - state.startedAt : 0;
  const texts = pad.value.split("\n");
  // A line keeps its original timestamp while its text at that index is
  // unchanged; new or edited lines get stamped with the current elapsed time.
  return texts.map((text, i) => {
    const prev = state.noteLines[i];
    if (prev && prev.text === text) return prev;
    return { ts_ms: Math.max(0, elapsed), text };
  });
}

async function saveNotepad(meetingId) {
  if (!state.noteDirty) return;
  const lines = currentNoteLines();
  state.noteDirty = false;
  try {
    await api(`/meetings/${meetingId}/rough-notes`, {
      method: "PUT",
      body: JSON.stringify({ lines }),
    });
    state.noteLines = lines;
    const el = document.getElementById("note-save-state");
    if (el) el.textContent = "· saved";
  } catch {
    state.noteDirty = true; // retry on the next tick
  }
}

function stopNotepad() {
  if (state.noteSaveHandle) clearInterval(state.noteSaveHandle);
  state.noteSaveHandle = null;
}

function startTimer() {
  stopTimer();
  const tick = () => {
    const el = document.getElementById("timer");
    if (el && state.startedAt) el.textContent = mmss(Date.now() - state.startedAt);
  };
  tick();
  state.timerHandle = setInterval(tick, 1000);
}

function stopTimer() {
  if (state.timerHandle) clearInterval(state.timerHandle);
  state.timerHandle = null;
}

async function renderMeeting(meetingId) {
  stopTimer();
  state.liveMeetingId = null;
  const { meeting, segments } = await api(`/meetings/${meetingId}/transcript`);
  const dur = meeting.ended_at && meeting.started_at ? mmss(meeting.ended_at - meeting.started_at) : "—";
  const lines = segments.map(finalLineHTML).join("") ||
    `<p class="empty">No transcript was captured.</p>`;
  app.innerHTML = `
    <div class="page-head">
      <h1>${esc(meeting.title || "Untitled")}
        <button class="trash" id="btn-rename-meeting" title="Rename meeting">✏️</button>
      </h1>
      <span class="meta">${meeting.started_at ? fmtDate(meeting.started_at) : ""} · ${dur}
        <button class="trash" id="btn-delete-meeting" title="Delete meeting">🗑</button>
      </span>
    </div>
    <div id="note-section" data-meeting-id="${esc(meetingId)}"></div>
    <h2 class="section-title">Transcript</h2>
    <div class="transcript readonly">${lines}</div>
  `;
  document.getElementById("btn-rename-meeting").onclick = () =>
    renameMeeting(meetingId, meeting.title || "", () => renderMeeting(meetingId));
  document.getElementById("btn-delete-meeting").onclick = () =>
    deleteMeeting(meetingId, meeting.title || "Untitled", () => { location.hash = "#/"; });
  await renderNoteSection(meetingId);
}

async function renderNoteSection(meetingId) {
  const slot = document.getElementById("note-section");
  if (!slot) return;
  let note;
  try { note = await api(`/meetings/${meetingId}/note`); } catch { return; }
  const templates = note.templates || ["general"];
  const templateSelect = `
    <select id="note-template">
      ${templates.map((t) => `<option value="${esc(t)}" ${t === (note.template || "general") ? "selected" : ""}>${esc(t.replace(/_/g, " "))}</option>`).join("")}
    </select>`;

  if (note.status === "generating") {
    slot.innerHTML = `
      <div class="note-card">
        <h2 class="section-title">Meeting note</h2>
        <p class="note generating">✨ Generating your note from the transcript and your rough notes…</p>
      </div>`;
    return;
  }
  if (note.status === "error") {
    slot.innerHTML = `
      <div class="note-card">
        <h2 class="section-title">Meeting note</h2>
        <p class="note">Generation failed: ${esc(note.error || "unknown error")}</p>
        <div class="form-line">${templateSelect}<button class="primary" id="btn-regen">Try again</button></div>
      </div>`;
    wireRegenerate(meetingId);
    return;
  }
  if (note.status !== "done") {
    const status = await api("/settings/status").catch(() => ({}));
    const hint = status.anthropic_configured
      ? ""
      : `<p class="note">Add your Anthropic API key in <a href="#/settings">Settings</a> to generate notes.</p>`;
    slot.innerHTML = `
      <div class="note-card">
        <h2 class="section-title">Meeting note</h2>
        ${hint}
        <div class="form-line">${templateSelect}<button class="primary" id="btn-regen" ${status.anthropic_configured ? "" : "disabled"}>Generate note</button></div>
      </div>`;
    wireRegenerate(meetingId);
    return;
  }

  const items = (note.action_items || []).map((a) => {
    const bits = [esc(a.task)];
    if (a.owner) bits.push(`<span class="chan">${esc(a.owner)}</span>`);
    if (a.due) bits.push(`<span class="meta">due ${esc(a.due)}</span>`);
    if (a.source_ts) bits.push(`<span class="ts">${esc(a.source_ts)}</span>`);
    return `<li>${bits.join(" ")}</li>`;
  }).join("");
  const decisions = (note.decisions || []).map((d) => `<li>${esc(d)}</li>`).join("");
  slot.innerHTML = `
    <div class="note-card">
      <div class="page-head">
        <h2 class="section-title">Meeting note</h2>
        <span>
          <button id="btn-note-edit">Edit</button>
          <a class="button-link" href="/meetings/${esc(meetingId)}/note.md" download><button>Export .md</button></a>
        </span>
      </div>
      <div class="note-body" id="note-body">${md2html(note.markdown)}</div>
      <div class="hidden" id="note-editor-wrap">
        <textarea id="note-editor" class="notepad tall"></textarea>
        <div class="form-line">
          <button class="primary" id="btn-note-save">Save</button>
          <button id="btn-note-cancel">Cancel</button>
        </div>
      </div>
      ${items ? `<h3>Action items</h3><ul class="action-items">${items}</ul>` : ""}
      ${decisions ? `<h3>Decisions</h3><ul>${decisions}</ul>` : ""}
      <div class="form-line">${templateSelect}<button id="btn-regen">Regenerate</button></div>
    </div>`;
  wireRegenerate(meetingId);
  document.getElementById("btn-note-edit").onclick = () => {
    document.getElementById("note-editor").value = note.markdown || "";
    document.getElementById("note-body").classList.add("hidden");
    document.getElementById("note-editor-wrap").classList.remove("hidden");
  };
  document.getElementById("btn-note-cancel").onclick = () => renderNoteSection(meetingId);
  document.getElementById("btn-note-save").onclick = async () => {
    try {
      await api(`/meetings/${meetingId}/note`, {
        method: "PUT",
        body: JSON.stringify({ markdown: document.getElementById("note-editor").value }),
      });
      renderNoteSection(meetingId);
    } catch (e) { showBanner(e.message, true); }
  };
}

function wireRegenerate(meetingId) {
  const btn = document.getElementById("btn-regen");
  if (!btn) return;
  btn.onclick = async () => {
    const select = document.getElementById("note-template");
    try {
      await api(`/meetings/${meetingId}/note/regenerate`, {
        method: "POST",
        body: JSON.stringify({ template: select ? select.value : null }),
      });
      renderNoteSection(meetingId);
    } catch (e) {
      if (e.code === "no_anthropic_key") {
        location.hash = "#/settings";
        showBanner("Add your Anthropic API key first, then generate notes.");
      } else {
        showBanner(e.message, true);
      }
    }
  };
}

async function renderSettings() {
  stopTimer();
  state.liveMeetingId = null;
  const status = await api("/settings/status");
  let deviceInfo = { devices: [], blackhole_detected: false };
  let deviceError = null;
  try {
    deviceInfo = await api("/settings/devices");
  } catch (e) {
    deviceError = e.message;
  }
  app.innerHTML = `
    <div class="page-head"><h1>Settings</h1></div>
    <div class="settings-row">
      <label for="dg-key">Deepgram API key</label>
      <span class="status-pill ${status.deepgram_configured ? "ok" : ""}">
        ${status.deepgram_configured ? "Configured" : "Not configured"}
      </span>
      <div class="form-line">
        <input type="password" id="dg-key" placeholder="Paste your Deepgram API key" autocomplete="off" />
        <button class="primary" id="btn-save-key">Save</button>
      </div>
      <p class="note">Stored in the macOS Keychain — never written to disk. Get a free key at
        <a href="https://console.deepgram.com" target="_blank" rel="noreferrer">console.deepgram.com</a>.</p>
    </div>
    <div class="settings-row">
      <label for="an-key">Anthropic API key — for note generation</label>
      <span class="status-pill ${status.anthropic_configured ? "ok" : ""}">
        ${status.anthropic_configured ? "Configured" : "Not configured"}
      </span>
      <div class="form-line">
        <input type="password" id="an-key" placeholder="Paste your Anthropic API key" autocomplete="off" />
        <button class="primary" id="btn-save-an-key">Save</button>
      </div>
      <p class="note">Used on meeting end to expand your rough notes into a structured note.
        Get a key at <a href="https://console.anthropic.com" target="_blank" rel="noreferrer">console.anthropic.com</a>.</p>
    </div>
    <div class="settings-row">
      <label>System audio — the other side of the call ("Them")</label>
      ${systemAudioHTML(status, deviceInfo, deviceError)}
    </div>
    <div class="settings-row">
      <label>Knowledge base — meeting copilot</label>
      <div id="kb-section"><p class="note">Loading…</p></div>
    </div>
  `;
  document.getElementById("btn-save-key").onclick = async () => {
    const key = document.getElementById("dg-key").value.trim();
    if (!key) return;
    try {
      await api("/settings/deepgram-key", { method: "PUT", body: JSON.stringify({ key }) });
      stopLevelTest();
      renderSettings();
      hideBanner();
    } catch (e) {
      showBanner(e.message, true);
    }
  };
  document.getElementById("btn-save-an-key").onclick = async () => {
    const key = document.getElementById("an-key").value.trim();
    if (!key) return;
    try {
      await api("/settings/anthropic-key", { method: "PUT", body: JSON.stringify({ key }) });
      stopLevelTest();
      renderSettings();
      hideBanner();
    } catch (e) {
      showBanner(e.message, true);
    }
  };
  wireSystemAudio(status, deviceInfo);
  renderKbSection();
}

async function renderKbSection() {
  const slot = document.getElementById("kb-section");
  if (!slot) return;
  let kb;
  try { kb = await api("/kb/docs"); } catch (e) {
    slot.innerHTML = `<p class="note">Could not load the knowledge base: ${esc(e.message)}</p>`;
    return;
  }
  const rows = kb.docs.map((d) => `
    <li>
      <span class="kb-doc">
        <span>${esc(d.title || d.path)}</span>
        <span class="meta">${d.chunks} chunks · ${d.ingested_at ? fmtDate(d.ingested_at) : ""}</span>
      </span>
      <button data-act="reingest" data-id="${esc(d.id)}" title="Re-ingest">↻</button>
      <button class="trash" data-act="remove" data-id="${esc(d.id)}" title="Remove from KB">🗑</button>
    </li>`).join("");
  slot.innerHTML = `
    <p class="note">The copilot watches the live transcript and surfaces snippets from
      these documents. <label class="inline"><input type="checkbox" id="copilot-enabled"
      ${kb.copilot_enabled ? "checked" : ""}/> Copilot enabled</label>
      · relevance floor <input type="number" id="copilot-floor" class="floor-input"
        min="0" max="1" step="0.05" value="${kb.relevance_floor}"/>
      ${kb.vec_available ? "" : " · <strong>keyword-only (sqlite-vec unavailable)</strong>"}</p>
    ${rows ? `<ul class="kb-list">${rows}</ul>` : `<p class="note">No documents ingested yet.</p>`}
    <div class="form-line">
      <input type="text" id="kb-path" placeholder="/path/to/file-or-folder (pdf, docx, md, txt, html)" />
      <button class="primary" id="btn-kb-ingest">Ingest</button>
      <button id="btn-kb-meetings">Ingest past meetings</button>
    </div>
    <p class="note">Also available from the terminal: <code>uv run meetnotes ingest &lt;path&gt;</code></p>
  `;
  document.getElementById("copilot-enabled").onchange = async (e) => {
    await api("/settings/copilot", { method: "PUT", body: JSON.stringify({ enabled: e.target.checked }) });
  };
  document.getElementById("copilot-floor").onchange = async (e) => {
    const value = parseFloat(e.target.value);
    if (!isNaN(value)) {
      await api("/settings/copilot", { method: "PUT", body: JSON.stringify({ relevance_floor: value }) });
    }
  };
  document.getElementById("btn-kb-ingest").onclick = async () => {
    const path = document.getElementById("kb-path").value.trim();
    if (!path) return;
    await runKbIngest({ path });
  };
  document.getElementById("btn-kb-meetings").onclick = () => runKbIngest({ include_meetings: true });
  for (const btn of slot.querySelectorAll("[data-act]")) {
    const id = btn.dataset.id;
    if (!id) continue;
    if (btn.dataset.act === "remove") {
      btn.onclick = async () => {
        await api(`/kb/docs/${id}`, { method: "DELETE" }).catch((e) => showBanner(e.message, true));
        renderKbSection();
      };
    } else if (btn.dataset.act === "reingest") {
      btn.onclick = async () => {
        btn.disabled = true;
        try { await api(`/kb/docs/${id}/reingest`, { method: "POST" }); }
        catch (e) { showBanner(e.message, true); }
        renderKbSection();
      };
    }
  }
}

async function runKbIngest(body) {
  const btn = document.getElementById("btn-kb-ingest");
  if (btn) { btn.disabled = true; btn.textContent = "Ingesting…"; }
  try {
    const res = await api("/kb/ingest", { method: "POST", body: JSON.stringify(body) });
    if (res.errors.length) showBanner(`Ingest finished with errors: ${res.errors.join("; ")}`, true);
    else hideBanner();
  } catch (e) {
    showBanner(e.message, true);
  }
  renderKbSection();
}

function systemAudioHTML(status, deviceInfo, deviceError) {
  if (deviceError) {
    return `<p class="note">Could not list audio devices: ${esc(deviceError)}</p>`;
  }
  if (!deviceInfo.blackhole_detected && !status.system_device) {
    return `
      <span class="status-pill">BlackHole not detected</span>
      <div class="setup-steps">
        <p>To transcribe the other side of a call, install the free
          <a href="https://github.com/ExistentialAudio/BlackHole" target="_blank" rel="noreferrer">BlackHole</a>
          loopback driver and route your call audio through it:</p>
        <ol>
          <li><code>brew install blackhole-2ch</code> (then restart this app)</li>
          <li>Open <strong>Audio MIDI Setup</strong> → “+” → <strong>Create Multi-Output Device</strong>;
              check both your speakers and BlackHole 2ch</li>
          <li>Select that Multi-Output Device as your Mac's sound output during calls</li>
        </ol>
        <button id="btn-rescan">Re-scan devices</button>
      </div>`;
  }
  const options = deviceInfo.devices.map((d) => {
    const selected = status.system_device === d.name ? "selected" : (!status.system_device && d.is_blackhole ? "selected" : "");
    return `<option value="${esc(d.name)}" data-index="${d.index}" ${selected}>${esc(d.name)}${d.is_blackhole ? " ✓ loopback" : ""}</option>`;
  }).join("");
  const configured = status.system_device
    ? `<span class="status-pill ok">Using: ${esc(status.system_device)}</span>`
    : `<span class="status-pill">Not configured — mic only</span>`;
  return `
    ${configured}
    <div class="form-line">
      <select id="sys-device">${options}</select>
      <button id="btn-level">Test signal</button>
      <button class="primary" id="btn-save-device">Use this device</button>
      ${status.system_device ? `<button id="btn-clear-device">Disable</button>` : ""}
    </div>
    <div class="level-meter"><div id="level-bar" class="level-bar"></div></div>
    <p class="note">Play some audio (e.g. a video) routed through your Multi-Output Device —
      the meter should move. During meetings this device is recorded as channel 1 ("Them").</p>`;
}

function wireSystemAudio(status, deviceInfo) {
  const rescan = document.getElementById("btn-rescan");
  if (rescan) rescan.onclick = () => renderSettings();
  const select = document.getElementById("sys-device");
  if (!select) return;
  const selectedIndex = () => {
    const opt = select.selectedOptions[0];
    return opt ? Number(opt.dataset.index) : null;
  };
  const levelBtn = document.getElementById("btn-level");
  levelBtn.onclick = async () => {
    if (state.levelTesting) {
      stopLevelTest();
      levelBtn.textContent = "Test signal";
      return;
    }
    try {
      await api("/settings/level-test", {
        method: "POST",
        body: JSON.stringify({ device: selectedIndex() }),
      });
      state.levelTesting = true;
      levelBtn.textContent = "Stop test";
    } catch (e) {
      showBanner(e.message, true);
    }
  };
  document.getElementById("btn-save-device").onclick = async () => {
    try {
      await api("/settings/system-device", {
        method: "PUT",
        body: JSON.stringify({ device_name: select.value }),
      });
      stopLevelTest();
      renderSettings();
    } catch (e) {
      showBanner(e.message, true);
    }
  };
  const clearBtn = document.getElementById("btn-clear-device");
  if (clearBtn) {
    clearBtn.onclick = async () => {
      await api("/settings/system-device", {
        method: "PUT",
        body: JSON.stringify({ device_name: null }),
      });
      stopLevelTest();
      renderSettings();
    };
  }
}

function stopLevelTest() {
  if (!state.levelTesting) return;
  state.levelTesting = false;
  api("/settings/level-test", { method: "DELETE" }).catch(() => {});
}

// ---- router ----------------------------------------------------------------

async function route() {
  if (!state.bannerSticky) hideBanner();
  stopLevelTest();
  stopNotepad();
  const hash = location.hash || "#/";
  const liveMatch = hash.match(/^#\/live\/([\w-]+)$/);
  const meetingMatch = hash.match(/^#\/meeting\/([\w-]+)$/);
  try {
    if (liveMatch) await renderLive(liveMatch[1]);
    else if (meetingMatch) await renderMeeting(meetingMatch[1]);
    else if (hash === "#/settings") await renderSettings();
    else await renderHome();
  } catch (e) {
    app.innerHTML = `<p class="empty">${esc(e.message)}</p>`;
  }
}

window.addEventListener("hashchange", route);

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k" && state.liveMeetingId) {
    e.preventDefault();
    copilotHelp();
  }
});

(async function init() {
  connectWS();
  await route();
  // First-run: steer to Settings until a Deepgram key exists.
  try {
    const status = await api("/settings/status");
    if (!status.deepgram_configured && location.hash !== "#/settings") {
      location.hash = "#/settings";
      showBanner("Welcome! Add your Deepgram API key to get started.");
    } else if (status.active_meeting_id && !location.hash.startsWith("#/live/")) {
      location.hash = `#/live/${status.active_meeting_id}`;
    }
  } catch { /* server not ready; ignore */ }
})();
