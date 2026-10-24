"use strict";

let ownerToken = "";
let authGeneration = 0;
let previousJobStates = null;
let jobPollTimer = null;
let jobPollInFlight = false;

const TERMINAL_JOB_STATES = new Set(["succeeded", "failed", "cancelled", "needs_reconciliation"]);

const byId = (id) => document.getElementById(id);
const authForm = byId("auth-form");
const tokenInput = byId("owner-token");
const refreshButton = byId("refresh-button");
const disconnectButton = byId("disconnect-button");

function setStatus(message, isError = false) {
  const node = byId("status-message");
  node.textContent = message;
  node.classList.toggle("error", isError);
}

function setConnected(connected) {
  const node = byId("connection-state");
  node.textContent = connected ? "Connected" : "Disconnected";
  node.classList.toggle("connected", connected);
  refreshButton.disabled = !connected;
  disconnectButton.disabled = !connected;
}

async function api(path, options = {}) {
  if (!ownerToken) throw new Error("Owner is disconnected.");
  const headers = new Headers(options.headers || {});
  headers.set("Authorization", `Bearer ${ownerToken}`);
  const response = await fetch(path, { ...options, headers, cache: "no-store", credentials: "same-origin" });
  if (!response.ok) {
    let message = `Request failed (HTTP ${response.status}).`;
    try {
      const body = await response.json();
      if (typeof body?.error?.message === "string") message = body.error.message;
    } catch (_error) {
      // Keep generic status when response is not a contract error.
    }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return response.status === 204 ? null : response.json();
}

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined && text !== null) element.textContent = String(text);
  if (className) element.className = className;
  return element;
}

function appendTable(target, headers, rows) {
  target.replaceChildren();
  if (!rows.length) {
    target.append(node("p", "No records yet.", "empty-state"));
    return;
  }
  const table = node("table");
  const thead = node("thead");
  const headerRow = node("tr");
  for (const label of headers) headerRow.append(node("th", label));
  thead.append(headerRow);
  table.append(thead);
  const tbody = node("tbody");
  for (const values of rows) {
    const row = node("tr");
    for (const value of values) {
      const cell = node("td");
      if (value instanceof Node) cell.append(value);
      else cell.textContent = value === null || value === undefined ? "—" : String(value);
      row.append(cell);
    }
    tbody.append(row);
  }
  table.append(tbody);
  target.append(table);
}

function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString();
}

function shortId(value) {
  return typeof value === "string" ? value.slice(0, 8) : "—";
}

function renderSessions(payload) {
  const sessions = payload.sessions || [];
  byId("session-count").textContent = String(sessions.length);
  appendTable(byId("sessions-content"), ["Session", "State", "Opened"], sessions.map((session) => [
    shortId(session.session_id),
    session.state,
    formatTime(session.opened_at),
  ]));
  const policy = byId("session-policy");
  const latest = sessions[0];
  policy.textContent = latest
    ? `Latest session policy: idle close ${latest.idle_close_seconds}s; maximum ${latest.max_session_seconds ?? "unset"}s.`
    : "No session policy recorded yet.";
}

function renderJobs(payload) {
  const jobs = payload.jobs || [];
  byId("job-count").textContent = String(jobs.length);
  const target = byId("jobs-content");
  target.replaceChildren();
  if (!jobs.length) {
    target.append(node("p", "No jobs yet.", "empty-state"));
    return;
  }
  for (const job of jobs) {
    const card = node("article", undefined, "job-card");
    const head = node("div", undefined, "job-head");
    head.append(node("span", `Job ${shortId(job.job_id)}`, "job-id"));
    head.append(node("span", job.status, `badge ${job.status}`));
    card.append(head);
    card.append(node("p", job.summary, "job-summary"));
    const artifacts = job.artifact_refs || [];
    if (artifacts.length) {
      const artifactList = node("div", undefined, "artifact-list");
      for (const artifact of artifacts) {
        const button = node("button", artifact.label, "button secondary");
        button.type = "button";
        button.addEventListener("click", () => downloadArtifact(artifact));
        artifactList.append(button);
      }
      card.append(artifactList);
    }
    const sources = job.sources || [];
    if (sources.length) {
      const sourceList = node("div", undefined, "source-list");
      for (const source of sources) {
        try {
          const url = new URL(source.url);
          if (!new Set(["https:", "http:"]).has(url.protocol)) continue;
          const link = node("a", source.title || url.hostname);
          link.href = url.href;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          sourceList.append(link);
        } catch (_error) {
          // Ignore malformed source links from stale data.
        }
      }
      if (sourceList.childElementCount) card.append(sourceList);
    }
    target.append(card);
  }
}

function observeJobUpdates(payload) {
  const next = new Map();
  for (const job of payload.jobs || []) {
    if (typeof job.job_id !== "string") continue;
    const state = {
      status: job.status,
      contextRevision: Number(job.context_revision || 0),
    };
    const previous = previousJobStates?.get(job.job_id);
    const isTerminal = TERMINAL_JOB_STATES.has(state.status);
    const changedTerminal = previousJobStates !== null
      && isTerminal
      && (!previous
        || !TERMINAL_JOB_STATES.has(previous.status)
        || previous.status !== state.status
        || previous.contextRevision !== state.contextRevision);
    if (changedTerminal) showJobNotification(job);
    next.set(job.job_id, state);
  }
  previousJobStates = next;
}

function showJobNotification(job) {
  const target = byId("job-notifications");
  while (target.childElementCount >= 3) target.firstElementChild.remove();
  const card = node("article", undefined, "job-notification");
  card.setAttribute("role", "status");
  const head = node("div", undefined, "job-notification-head");
  head.append(node("strong", `Job ${shortId(job.job_id)} ${job.status}.`));
  const dismiss = node("button", "Dismiss", "notification-dismiss");
  dismiss.type = "button";
  dismiss.addEventListener("click", () => card.remove());
  head.append(dismiss);
  card.append(head);
  const summary = typeof job.summary === "string" ? job.summary : "Result ready.";
  card.append(node("p", summary.length > 240 ? `${summary.slice(0, 237)}…` : summary));
  target.append(card);
}

function startJobPolling() {
  if (jobPollTimer !== null || !ownerToken) return;
  jobPollTimer = setInterval(pollJobs, 15_000);
}

function stopJobPolling() {
  if (jobPollTimer !== null) clearInterval(jobPollTimer);
  jobPollTimer = null;
}

async function pollJobs() {
  if (!ownerToken || jobPollInFlight || document.visibilityState === "hidden") return;
  const requestToken = ownerToken;
  const requestGeneration = authGeneration;
  jobPollInFlight = true;
  try {
    const jobs = await api("/v1/jobs?limit=100");
    if (requestGeneration !== authGeneration || requestToken !== ownerToken) return;
    renderJobs(jobs);
    observeJobUpdates(jobs);
    setStatus(`Checked jobs ${new Date().toLocaleTimeString()}.`);
  } catch (error) {
    if (requestGeneration !== authGeneration || requestToken !== ownerToken) return;
    if (error.status === 401 || error.status === 403) {
      ownerToken = "";
      authGeneration += 1;
      setConnected(false);
      clearDashboard();
    }
    setStatus(error.message || "Could not refresh jobs.", true);
  } finally {
    jobPollInFlight = false;
  }
}

async function downloadArtifact(artifact) {
  try {
    const response = await fetch(`/v1/artifacts/${encodeURIComponent(artifact.artifact_id)}`, {
      headers: { Authorization: `Bearer ${ownerToken}` },
      cache: "no-store",
      credentials: "same-origin",
    });
    if (!response.ok) throw new Error(`Download failed (HTTP ${response.status}).`);
    const url = URL.createObjectURL(await response.blob());
    const link = node("a");
    link.href = url;
    link.download = (artifact.label || "cube-artifact").replace(/[\\/:*?"<>|]/g, "_");
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1_000);
    setStatus(`Downloaded ${artifact.label}.`);
  } catch (error) {
    setStatus(error.message || "Artifact download failed.", true);
  }
}

function renderApprovals(payload) {
  const approvals = payload.approvals || [];
  byId("approval-count").textContent = String(approvals.filter((item) => item.status === "pending").length);
  const target = byId("approvals-content");
  target.replaceChildren();
  if (!approvals.length) {
    target.append(node("p", "No approvals.", "empty-state"));
    return;
  }
  for (const approval of approvals) {
    const card = node("article", undefined, "approval-card");
    const head = node("div", undefined, "approval-head");
    head.append(node("strong", approval.action_summary));
    head.append(node("span", approval.status, `badge ${approval.status}`));
    card.append(head);
    card.append(node("p", `Expires ${formatTime(approval.expires_at)} · revision ${approval.context_revision}`, "muted small"));
    if (approval.status === "pending") {
      const buttons = node("div", undefined, "button-row");
      buttons.append(decisionButton(approval, "approve"));
      buttons.append(decisionButton(approval, "deny"));
      card.append(buttons);
    }
    target.append(card);
  }
}

function decisionButton(approval, decision) {
  const button = node("button", decision === "approve" ? "Approve" : "Deny", `button ${decision === "approve" ? "primary" : "danger"}`);
  button.type = "button";
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await api(`/v1/approvals/${encodeURIComponent(approval.approval_id)}/decision`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Idempotency-Key": crypto.randomUUID(),
        },
        body: JSON.stringify({
          decision,
          payload_digest: approval.payload_digest,
          context_revision: approval.context_revision,
        }),
      });
      setStatus(`Approval ${decision}d.`);
      await refreshDashboard();
    } catch (error) {
      button.disabled = false;
      setStatus(error.message || "Approval decision failed.", true);
    }
  });
  return button;
}

function renderMemories(payload) {
  const memories = payload.memories || [];
  byId("memory-count").textContent = String(memories.length);
  const target = byId("memories-content");
  target.replaceChildren();
  if (!memories.length) {
    target.append(node("p", "No saved memories.", "empty-state"));
    return;
  }
  for (const memory of memories) {
    const card = node("article", undefined, "memory-card");
    const head = node("div", undefined, "approval-head");
    head.append(node("strong", memory.category));
    head.append(node("span", shortId(memory.memory_id), "job-id"));
    card.append(head);
    card.append(node("p", memory.content, "job-summary"));
    card.append(node("p", `Updated ${formatTime(memory.updated_at)}`, "muted small"));
    const buttons = node("div", undefined, "button-row");
    const remove = node("button", "Delete", "button danger");
    remove.type = "button";
    remove.addEventListener("click", () => deleteMemory(memory, remove));
    buttons.append(remove);
    card.append(buttons);
    target.append(card);
  }
}

async function deleteMemory(memory, button) {
  if (!window.confirm(`Delete saved ${memory.category} memory?`)) return;
  button.disabled = true;
  try {
    await api(`/v1/memories/${encodeURIComponent(memory.memory_id)}`, {
      method: "DELETE",
      headers: { "Idempotency-Key": crypto.randomUUID() },
    });
    setStatus("Memory deleted.");
    await refreshDashboard();
  } catch (error) {
    button.disabled = false;
    setStatus(error.message || "Memory deletion failed.", true);
  }
}

function renderUsage(payload) {
  const summary = payload.summary || {};
  byId("input-tokens").textContent = Number(summary.input_tokens || 0).toLocaleString();
  byId("output-tokens").textContent = Number(summary.output_tokens || 0).toLocaleString();
  byId("connected-seconds").textContent = Number(summary.connected_seconds || 0).toLocaleString(undefined, { maximumFractionDigits: 1 });
  byId("measured-cost").textContent = summary.measured_cost_usd === null || summary.measured_cost_usd === undefined
    ? "—" : `$${Number(summary.measured_cost_usd).toFixed(4)}`;
  byId("estimated-cost").textContent = summary.estimated_cost_usd === null || summary.estimated_cost_usd === undefined
    ? "—" : `$${Number(summary.estimated_cost_usd).toFixed(4)}`;
  byId("usage-period").textContent = `Since ${formatTime(summary.period_start)}`;
  const records = payload.measured_usage || [];
  appendTable(byId("usage-content"), ["Provider", "Observed", "Tokens", "Seconds", "USD"], records.slice(0, 20).map((item) => {
    const measurement = item.measurement || {};
    const tokens = [measurement.input_tokens, measurement.output_tokens].filter((value) => value !== null && value !== undefined).join(" / ") || "—";
    return [
      item.provider,
      formatTime(item.observed_at),
      tokens,
      measurement.connected_seconds ?? "—",
      measurement.cost_usd === null || measurement.cost_usd === undefined ? "—" : `$${measurement.cost_usd} (${measurement.cost_kind})`,
    ];
  }));
}

async function refreshDashboard() {
  if (!ownerToken) return;
  const requestToken = ownerToken;
  const requestGeneration = authGeneration;
  refreshButton.disabled = true;
  setStatus("Loading owner data…");
  try {
    const [sessions, jobs, approvals, memories, usage] = await Promise.all([
      api("/v1/sessions?limit=100"),
      api("/v1/jobs?limit=100"),
      api("/v1/approvals"),
      api("/v1/memories"),
      api("/v1/usage"),
    ]);
    if (requestGeneration !== authGeneration || requestToken !== ownerToken) return;
    renderSessions(sessions);
    renderJobs(jobs);
    observeJobUpdates(jobs);
    renderApprovals(approvals);
    renderMemories(memories);
    renderUsage(usage);
    setConnected(true);
    setStatus(`Updated ${new Date().toLocaleTimeString()}.`);
    startJobPolling();
  } catch (error) {
    if (requestGeneration !== authGeneration || requestToken !== ownerToken) return;
    if (error.status === 401 || error.status === 403) {
      ownerToken = "";
      authGeneration += 1;
      setConnected(false);
      clearDashboard();
    }
    setStatus(error.message || "Could not load dashboard.", true);
  } finally {
    refreshButton.disabled = !ownerToken;
  }
}

function clearDashboard() {
  stopJobPolling();
  previousJobStates = null;
  byId("job-notifications").replaceChildren();
  byId("session-count").textContent = "0";
  byId("job-count").textContent = "0";
  byId("approval-count").textContent = "0";
  byId("memory-count").textContent = "0";
  byId("sessions-content").replaceChildren(node("p", "Connect to load sessions.", "empty-state"));
  byId("jobs-content").replaceChildren(node("p", "Connect to load jobs.", "empty-state"));
  byId("approvals-content").replaceChildren(node("p", "Connect to load approvals.", "empty-state"));
  byId("memories-content").replaceChildren(node("p", "Connect to load memories.", "empty-state"));
  byId("usage-content").replaceChildren(node("p", "Connect to load usage.", "empty-state"));
  byId("session-policy").textContent = "";
  byId("usage-period").textContent = "";
  for (const id of ["input-tokens", "output-tokens", "connected-seconds", "measured-cost", "estimated-cost"]) {
    byId(id).textContent = "—";
  }
}

authForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  ownerToken = tokenInput.value.trim();
  authGeneration += 1;
  tokenInput.value = "";
  setConnected(false);
  clearDashboard();
  if (!ownerToken) {
    setStatus("Owner token required.", true);
    return;
  }
  await refreshDashboard();
});

refreshButton.addEventListener("click", refreshDashboard);
disconnectButton.addEventListener("click", () => {
  ownerToken = "";
  authGeneration += 1;
  setConnected(false);
  clearDashboard();
  setStatus("Disconnected. Token cleared from page memory.");
});
