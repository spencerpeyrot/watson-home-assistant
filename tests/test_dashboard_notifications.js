"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class FakeElement {
  constructor() {
    this.children = [];
    this.textContent = "";
    this.className = "";
    this.disabled = false;
    this.classList = { toggle() {} };
  }

  append(...children) {
    for (const child of children) {
      if (child instanceof FakeElement) child.parentElement = this;
      this.children.push(child);
    }
  }

  replaceChildren(...children) {
    this.children = [];
    this.append(...children);
  }

  addEventListener() {}
  setAttribute() {}

  get childElementCount() {
    return this.children.filter((child) => child instanceof FakeElement).length;
  }

  get firstElementChild() {
    return this.children.find((child) => child instanceof FakeElement) || null;
  }

  remove() {
    if (!this.parentElement) return;
    this.parentElement.children = this.parentElement.children.filter((item) => item !== this);
  }
}

function loadDashboard({
  fetchImpl = fetch,
  setIntervalImpl = setInterval,
  clearIntervalImpl = clearInterval,
} = {}) {
  const elements = new Map();
  const document = {
    visibilityState: "visible",
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new FakeElement());
      return elements.get(id);
    },
    createElement() {
      return new FakeElement();
    },
  };
  const context = {
    document,
    Node: FakeElement,
    Headers,
    fetch: fetchImpl,
    setInterval: setIntervalImpl,
    clearInterval: clearIntervalImpl,
  };
  const source = fs.readFileSync(path.join(__dirname, "../dashboard/app.js"), "utf8");
  vm.runInNewContext(
    `${source}
globalThis.observeJobUpdatesForTest = observeJobUpdates;
globalThis.startJobPollingForTest = startJobPolling;
globalThis.pollJobsForTest = pollJobs;
globalThis.stopJobPollingForTest = stopJobPolling;
globalThis.setAuthForTest = (token) => { ownerToken = token; authGeneration += 1; };`,
    context,
  );
  return {
    context,
    observe: context.observeJobUpdatesForTest,
    startPolling: context.startJobPollingForTest,
    poll: context.pollJobsForTest,
    stopPolling: context.stopJobPollingForTest,
    setAuth: context.setAuthForTest,
    notifications: document.getElementById("job-notifications"),
  };
}

function job(jobId, status, contextRevision = 0, summary = "Ready.") {
  return { job_id: jobId, status, context_revision: contextRevision, summary };
}

function testOnlyNewTerminalTransitionsNotify() {
  const { observe, notifications } = loadDashboard();
  observe({ jobs: [job("job-1", "succeeded")] });
  assert.equal(notifications.childElementCount, 0, "initial load must not alert for old results");
  observe({ jobs: [job("job-1", "running")] });
  observe({ jobs: [job("job-1", "succeeded", 0, "<script>literal text</script>")] });
  assert.equal(notifications.childElementCount, 1);
  assert.equal(notifications.children[0].children[1].textContent, "<script>literal text</script>");
  observe({ jobs: [job("job-1", "succeeded", 0)] });
  assert.equal(notifications.childElementCount, 1, "unchanged terminal jobs must not duplicate alerts");
  observe({ jobs: [job("job-1", "succeeded", 1, "Corrected result.")] });
  assert.equal(notifications.childElementCount, 2, "new correction revision must notify");
}

function testDashboardKeepsOnlyThreeRecentNotices() {
  const { observe, notifications } = loadDashboard();
  const jobs = ["one", "two", "three", "four"].map((id) => job(id, "running"));
  observe({ jobs });
  observe({ jobs: jobs.map((item) => ({ ...item, status: "succeeded" })) });
  assert.equal(notifications.childElementCount, 3);
  assert.match(notifications.firstElementChild.children[0].children[0].textContent, /Job two/);
}

async function testPollingUsesInMemoryOwnerTokenAndOnlyJobsRoute() {
  const requests = [];
  let intervalMilliseconds = 0;
  let clearedInterval = false;
  const { context, setAuth, startPolling, poll, stopPolling } = loadDashboard({
    fetchImpl: async (url, options) => {
      requests.push({ url, options });
      return { ok: true, status: 200, json: async () => ({ jobs: [job("job-1", "running")] }) };
    },
    setIntervalImpl: (_callback, delay) => {
      intervalMilliseconds = delay;
      return 7;
    },
    clearIntervalImpl: (id) => {
      clearedInterval = id === 7;
    },
  });
  setAuth("owner-token-in-memory");
  startPolling();
  assert.equal(intervalMilliseconds, 15_000);
  await poll();
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "/v1/jobs?limit=100");
  assert.equal(requests[0].options.headers.get("Authorization"), "Bearer owner-token-in-memory");
  assert.equal(requests[0].options.cache, "no-store");
  context.document.visibilityState = "hidden";
  await poll();
  assert.equal(requests.length, 1, "hidden dashboard must not poll");
  stopPolling();
  assert.equal(clearedInterval, true);
}

async function run() {
  testOnlyNewTerminalTransitionsNotify();
  testDashboardKeepsOnlyThreeRecentNotices();
  await testPollingUsesInMemoryOwnerTokenAndOnlyJobsRoute();
  process.stdout.write("Dashboard job notification checks passed.\n");
}

run().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
