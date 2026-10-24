# Assistant observability gap analysis and implementation status

Audit date: 24 October 2026 (America/New_York). Source baseline: `b0a0705`. Scope: Pi voice, gateway, Muse boundary, durable jobs, research, device tools, timers, storage, export, deployment and iteration feedback.

## What this repo does

Watson is a household voice assistant. A Raspberry Pi detects wake words, captures speech, sends turns to ElevenLabs/Jev/Muse, then plays replies. Gateway and worker services also run durable research, device actions and timer delivery. Assumed load: one household/device with concurrent voice, timers and occasional background jobs; reliability matters even at low traffic.

**Audit verdict:** useful voice traces existed, but could not explain every failed interaction. This PR closes several code gaps below. Production ingestion, independent telemetry-loss detection, dashboards and hardware behavior still need operational validation.

“No matter what fails” needs a measurable boundary. Power loss, a dead microphone and a broken exporter cannot reliably report through the failing process. The attainable contract is: every accepted operation has a start and terminal outcome, or an independent watcher detects an overdue/missing outcome; evidence survives a process restart; telemetry outages become visible outside Langfuse. Physical audibility still requires hardware verification.

## Evidence and limits

Second pass: error-preservation and liveness review added findings 19–20. These expand earlier categories with concrete failure cases. Findings are not a claim of exhaustive coverage; confidence must come from the acceptance drills and deployed evidence.

Finding evidence and line references describe baseline `b0a0705`; the implementation-status table describes this PR.

- Read tracing adapter, main session lifecycle, Muse turn stages, gateway admission/cleanup, worker execution and persistence, graph model/tool instrumentation, research invocation and usage callback, timer retry loop, device transports, deployment units, CI and evaluation reporters. Searched tests for instrumentation coverage. Inspected provider/tool boundary code selectively, rather than every repository line.
- Existing documentation describes canary releases and prior ingestion checks. These are historical evidence, not confirmation of current deployed configuration.
- Connected Langfuse read: `listObservations`, 17 October 00:00 UTC through 25 October 00:00 UTC, returned an empty collection. Connected dashboard, alert, score-config and evaluation-rule lists also returned empty collections. Earlier repo notes already suspect a project mismatch (`docs/langfuse-observability.md`). These results **do not establish that production has no traces or configuration**. Project/region identity and deployment access need verification first.
- No protected environment files, credentials, raw conversations or recordings were copied into this report. No production changes, synthetic device actions or paid provider experiments were run. Focused automated tests run for changed trace, provider, timer, audio, gateway, cleanup and research paths; hardware and real-provider drills remain pending. Existing local change in `service/muse_gadget.py` was preserved outside this PR worktree.
- Installed dependency lock is Langfuse Python **4.17.0**; dependency constraint is `>=4,<5`. PyPI reported **5.0.4** during this audit. Current documentation contains breaking v5 changes; examples must match the deployed SDK/server.

## What already works

Preserve these foundations:

- `voice/observability.py` centralizes optional, failure-isolated instrumentation, bounded content, credential redaction, session propagation and shutdown.
- Device owns deterministic conversation root; gateway and worker attach to that trace. Diagnostics are grouped to reduce audio-event noise.
- Speech recognition, route selection/fallback, Muse wait and synthesis have useful timing events. PCM levels, VAD speech duration, queue loss and first speaker frame help diagnose bad input/output without exporting PCM.
- `service/job_graph.py` records individual ordinary model calls as generations, tools as tool observations, and usage from model responses.
- Worker records failed execution, lease loss and publication failures; durable application state already provides useful recovery evidence.
- Content export is explicit opt-in. The existing docs correctly warn that conversation content can remain personal despite credential masking.

## Implementation status in this PR

This table separates repository changes from production work that needs deployed access, hardware or an independent monitoring service.

| Finding | PR status | Still needed |
|---|---|---|
| 1 Project identity/readback | Not implemented | Verify project, emit and read back canary from each deployed component |
| 2 Export loss | Partial | External readback/heartbeat alert and durable local failure evidence |
| 3 Wake/admission gaps | Partial | Added wake-attempt, HTTP request and WebSocket admission traces; verify canary and idle hardware externally |
| 4 Terminal outcomes | Partial | Shared status levels plus turn, graph, worker and device outcomes; audit remaining close/recovery paths |
| 5 Startup/task supervision | Partial | Added unexpected background-task logs and degraded health status; pre-runtime capture and independent unit watcher remain |
| 6 Redaction | Implemented in adapter | Test against deployment secrets and personal-data policy |
| 7 Trace unit | Partial | Added voice-turn span and independent job root when no live parent exists; migrate to per-turn roots and durable links |
| 8 Research internals | Implemented in code | Validate production trace shape and usage against real LangChain runs |
| 9 Muse internals | Not implemented | Upstream run/request identity and provider-side telemetry access |
| 10 Provider diagnostics | Partial | Added STT/TTS stage, model, cause, safe code/status/request ID, timeout and partial-output fields; deduplicated repeated partial-usage snapshots; provider readback remains |
| 11 Timer delivery | Partial | Added server delivery and Pi receipt/presentation/tool-result traces, retries, acknowledgements and deadline; physical audibility remains unverified |
| 12 Answer delivery | Partial | Added playback queue deadline and first-frame signal; full turn drain and physical audibility confirmation remain |
| 13 Queue/storage boundaries | Partial | Added artifact-cleanup outcomes; job queue age, claim/checkpoint timing and pool metrics remain |
| 14 Release provenance | Partial | Trace carries release/environment when configured; packaging must provide immutable release values |
| 15 Trace contract | Partial | Added focused redaction/error/status tests; full versioned exported-tree contract and migration remain |
| 16 Dashboards/alerts | Not implemented | Configure denominators, Langfuse alerts and independent missing-canary/unit alerts |
| 17 Evaluation linkage | Not implemented | Add trace-linked scores and deterministic incident regression gates |
| 18 SDK migration | Deferred | Keep pinned v4 until deployed server compatibility and real readback are confirmed |
| 19 Other cleanup errors | Partial | Gateway cleanup and artifact deletion now emit outcomes; ALSA stderr, SQLite/job notification diagnostics remain |
| 20 Hung stages | Partial | Added STT/TTS, timer, Muse-turn and playback-drain deadlines; independent overdue watcher and non-voice background-stage deadlines remain |

## Coverage map

“Partial” means some evidence exists, not that every failure is captured.

| Boundary | Existing evidence | Gap to close |
|---|---|---|
| Boot/config/dependency initialization | Startup observation, stderr/journal | Crashes before initialization, component readiness, external heartbeat |
| Idle capture/wake/session HTTP creation | Local logs; trace begins after session exists | Failed attempts and idle hardware health |
| Gateway authentication/session admission | Socket close reasons | Structured rejected-attempt record before gateway trace starts |
| STT and route decision | Timed stage spans, lengths/confidence/fallback | Provider call identity/usage, explicit degraded/failed decisions |
| Muse | Wait span and arriving text | Internal calls/tools opaque; upstream correlation absent from trace schema |
| TTS and speaker playback | Synthesis span, queues, first playback frame | Per-turn drain completion, cancellation and physical outcome boundary |
| Ordinary background graph | Generations/tools and aggregate result | Queue wait, claim failures, persistent outcomes and complete context |
| Sol research | Parent graph and usage totals | Each inner model/tool/retrieval invocation |
| Timers/device actions/notifications | Durable records, retry logs, some enclosing tool spans | Independent asynchronous delivery, acknowledgement and retry traces |
| Database/checkpoint/artifact operations | Parent errors, selective warnings | Named boundary timing, pool/lease/backlog metrics |
| Langfuse export | Startup/session flush logs | Ingestion proof, export loss signal, durable fallback |
| Quality/regressions | Local replay/report tools and CI tests | Correlated scores, failure dataset and telemetry contract gates |

## Must fix

### 1. Prove project identity and ingestion before trusting trace absence

**Evidence:** `docs/langfuse-observability.md` records connected-project mismatch concerns; current connected reads are empty. `voice/observability.py:458` emits startup observations but does not verify their readback.

**Failure case:** Pi exports to project A while operators inspect project B. An incident looks like missing instrumentation even if ingestion works.

**Patch:** Record a non-secret project identifier, normalized region/base host, component, environment and release in deployment evidence. Compare actual API/voice/worker configuration privately to the connected project. Emit a uniquely identified, content-free canary and verify its exact observation through the same project. Do not print keys or assume that a successful flush authenticates and persists data.

**Acceptance:** One canary from each process is readable by exact ID in the intended project and environment. Store timestamp, component, release and links in the runbook. Treat project access mismatch as a blocked deployment check, not an empty production result.

### 2. Exporter failures can look successful, with no independent loss signal

**Evidence:** `voice/observability.py:232,458` logs success when `flush()` returns; `_get_client()` latches `_initialized` before configuration/initialization succeeds. Export is batched in process memory, with no application-owned durable fallback. Langfuse documents that flush may log network failure without raising.

**Failure case:** Wrong credentials, network loss or process kill loses telemetry while “flush completed” appears in logs. A failed initialization leaves tracing disabled for that process until restart.

**Patch:** Rename flush logs to describe attempted/drained SDK work, not confirmed ingestion. Add counters for instrumentation exceptions/export failure signals and an independent canary readback monitor. Expose configured/initialized/last-readback health separately. For this household system, start with persistent structured journal records for lifecycle/failure events; add a bounded local outbox only where replay is required. If adding an outbox, specify byte cap, retention, permissions, dedup IDs, disk-full behavior and replay order; never block audio on export. Recover initialization through bounded retry or explicit restart supervision.

**Acceptance:** Reject export requests, then kill/restart a process. Operator sees lost/degraded telemetry and durable failure records; recovery readback succeeds. Do not infer that retrying `flush()` itself guarantees persistence.

### 3. Failed wake/session creation and rejected connections have no trace

**Evidence:** Device root starts inside `on_session_created` (`device/voice_runtime.py:420`). `service/live_gateway.py:90–133` authenticates, looks up and claims session before creating gateway observation. HTTP middleware (`service/durable_api.py:165`) assigns a UUID but creates no observation.

**Failure case:** Wake happens but session creation times out, database lookup fails or WebSocket admission rejects the device. No conversation tree exists to explain silence.

**Patch:** Create local `interaction_attempt_id` at wake, before network I/O. Persist safe attempt/start/failure records; carry attempt ID into session creation and later spans. Add one HTTP request boundary and one socket-admission boundary capturing route template, status, elapsed time and classified reason. For unauthenticated traffic, generate server request IDs and never trust caller-provided identity or record credentials. Link the attempt to the allocated session rather than assigning a fake session.

**Acceptance:** Session HTTP 401/503/timeout and socket lookup/claim failure each leave a searchable attempt with terminal cause and no user text.

### 4. Root outcomes hide child failures and cancellations

**Evidence:** `device/voice_runtime.py:823` ends the root as `closed`. `service/job_worker.py:73–82,147–175` records child failure/lease events but ends lifecycle without corresponding terminal root status. `service/job_graph.py:735` similarly ends its wrapper without outcome. `Observation.update()` maps only `failed`, `expired`, `abandoned` to error/warning; `operation()` treats all `BaseException`, including intentional cancellation, as failure. `service/muse_provider.py:722` rethrows cancellation before a terminal turn event.

**Failure case:** A failed task or transport failure appears under an ordinary closed conversation. Intentional cancellation inflates error counts while unexpected interruption lacks a turn outcome.

**Patch:** Define terminal outcomes once: succeeded, failed, cancelled, rejected, timed_out, abandoned; represent degraded/fallback separately. Propagate actual outcome to operation, turn and lifecycle roots. Keep session close reason distinct from task success. Preserve first/root failure and final cleanup failure independently. Set Langfuse level/status message consistently, using sanitized error code and failure stage.

**Acceptance:** Provider timeout, lease loss, publication failure, user stop and shutdown cancellation each produce exactly one correct terminal outcome; failed work cannot be counted as success because cleanup ran.

### 5. Failures before startup and silent background-task death are weakly observable

**Evidence:** `service/worker.py:190–193` opens store after tracing initialization but before its cleanup `try`; process-level handler prints only error type. `service/durable_api.py:125–143` launches background tasks and gathers exceptions at shutdown; no running-task supervision here. `device/voice_service.py:53` initializes only after composition has succeeded. Units restart failed processes, with finite start bursts.

**Failure case:** Bad config/audio setup prevents tracing initialization; timer/cleanup task dies while API remains alive; start-limit exhaustion leaves service down.

**Patch:** Configure safe structured logging before settings/hardware setup. Capture boot stage and fatal cause in journal. Supervise required background tasks: unexpected completion marks component degraded and triggers restart/failure policy. Add an external watcher for unit inactivity, restart loops, heartbeat age, disk pressure, capture progress and queue age. Health endpoint currently checks PostgreSQL only (`service/durable_api.py:620`); extend component readiness without making Langfuse availability block voice service.

**Acceptance:** Fail boot, kill required task and exhaust restart budget. Independent watcher identifies component and stage even when Langfuse receives nothing.

### 6. Redaction truncates before masking, and metadata bypasses content masking

**Evidence:** `_content()` (`voice/observability.py:155`) slices strings to 4,000 characters before replacing known environment secrets. `_metadata()` allowlists names and scalar lengths, but does not apply secret masking. `reason` accepts arbitrary strings; callers include exception-derived reasons.

**Failure case:** An arbitrary environment password crossing the truncation boundary becomes a prefix that no longer matches the complete known secret; generic patterns may not catch it. A short secret inside a permitted metadata string can export unchanged.

**Patch:** Redact full strings before truncating, then apply explicit truncation markers and byte limits. Apply common masking to metadata/status messages and at final export boundary where supported. Prefer bounded reason enums over exception messages. Preserve opt-in text policy, define retention/access/deletion policy and keep raw audio off by default. Do not treat regex credential masking as personal-data removal.

**Acceptance:** Adversarial cases cover secrets crossing boundaries, nested content, metadata reasons, unknown objects and content disabled. No literal or partial credential reaches captured exporter payloads.

## Should fix

### 7. Conversation-wide trace is a poor long-term unit for turn outcomes

**Evidence:** `start_session()` forces `session_id` as trace ID and attaches remote work to deterministic device root. `Observation.transcript()` retains only twenty role segments, each at most 4,000 characters; updates separate user and assistant arrays. Current Langfuse guidance recommends one self-contained turn/run per trace, grouped by session.

**Failure case:** Multiple turns and late background jobs share a root that already closed; root transcript truncation and mixed outcomes make per-turn scores and latency ambiguous. Device-root loss leaves remote children without their root.

**Patch:** First add explicit turn parent observations with input, output, outcome and turn ID. Then migrate to one trace per turn and independent durable job/delivery attempt, sharing conversation/session ID. Persist parent trace/span and originating turn with work; use span links/metadata for asynchronous work that outlives the turn. Make each service-created root useful even if Pi root never arrives. Version the trace schema and retain queries for old trees during migration.

**Acceptance:** Two questions, one ignored turn and a late job result each have unambiguous outcomes; session view groups conversation correctly. A missing device root does not erase server context.

### 8. Research agent's inner reasoning loop and tools are invisible

**Evidence:** `service/research_agent.py:357–375` invokes Deep Agent with only `_AgentUsageCallback`; callback records aggregate message usage. No Langfuse instrumentation appears in that module. Custom exporter only accepts names starting `cube.` (`voice/observability.py:224`).

**Failure case:** Research exhausts recursion/search limits or fails source validation. Parent graph fails but trace cannot show which model call or search led there.

**Patch:** Instrument each completed/failed model call and tool invocation in the existing callback/boundaries, or integrate official LangChain handler using the same tracer/context and explicit safe capture settings. If choosing integration, update exporter filtering deliberately; simply attaching callback will not bypass the current name filter. Preserve usage callback and avoid duplicate spans/billing totals. Capture source IDs/counts, search outcome, report validation reason and model identity; never request or store hidden chain of thought.

**Acceptance:** Research with two searches shows distinct generations, tools and validation result; failure at search two identifies that call. Summed generation usage agrees with provider measurements where available.

### 9. Muse is a measured wait, not an observable agent run

**Evidence:** `service/muse_provider.py:598–636` wraps Muse in wait span; response/session handling does not export a dedicated upstream request/run identifier through `_FIELDS`. Main voice path delegates ordinary conversation to external Muse, not the locally instrumented job graph.

**Failure case:** Muse answer is slow/wrong or performs an unexpected action. Watson sees elapsed wait and text but cannot explain upstream retrieval, tool decisions or model usage.

**Patch:** Add sanitized upstream session/run/request identifiers and separate send, first-text and completion boundaries. Propagate trace context if Muse supports it; otherwise link upstream run by identifier and collect its telemetry independently. Audit the actual upstream deployment/provider instrumentation. Describe unavailable internal visibility explicitly; do not label a wait span as a fully traced generation or invent usage.

**Acceptance:** One Watson turn can be followed to corresponding Muse run. If upstream access is unavailable, trace marks coverage as external/unknown and exposes request identity and timing.

### 10. Provider usage, retries and latency are incomplete across voice path

**Evidence:** `observed_await()` supports usage only on ordinary graph responses. STT/TTS are spans, route classifier is wrapped as `voice.route`, and Muse is a wait. No complete standard generation/model/usage export is present for all these boundaries; `Observation.usage()` suppresses update exceptions silently.

**Failure case:** Jev fallback, TTS regeneration or upstream retries increase bill/latency while model dashboard mostly reflects durable graph calls.

**Patch:** Use correct observation types: generations for actual LLM invocations; speech operations with audio seconds/characters and measured provider billing units as appropriate. Record provider, exact model, attempt, timeout, sanitized HTTP/provider code and safe request ID. Record time to first result/audio separately from total duration. Preserve input/output/cached token details when provider exposes them. Reconcile with existing application usage ledger; mark unknown explicitly and separate estimated prices from measured usage. Make usage-update failure observable without breaking requests.

**Acceptance:** Forced primary failure shows failed primary and fallback attempts; totals do not double-count retries or parent aggregate usage. Unknown Muse/STT costs never appear as zero.

#### ElevenLabs: preserve actual cause, not only exception class

This is a confirmed gap in both speech adapters, not merely a dashboard improvement.

- `service/elevenlabs_stt.py`: provider `error`/`rate_limited` events collapse to “ElevenLabs rejected transcription.” Connection exceptions become `ElevenLabsSTTError`; original exception type exists only inside a message and exception chaining is suppressed.
- `service/elevenlabs_tts.py`: provider error payload collapses to “ElevenLabs rejected speech generation.” Handshake, network, stream and protocol errors share `ElevenLabsError`. Original transport type is retained only in message text.
- `Observation.operation()` exports outer exception class only. `service/muse_provider.py` likewise reports `type(error).__name__` in failed-turn and acknowledgement events. Thus existing traces can show `ElevenLabsError` without the useful underlying cause. Failed-turn event also omits `turn_id` despite having it in scope.

**Required patch:** Keep current exception classes for caller compatibility; add safe structured attributes at the adapter boundary, then export those attributes through the shared observation error path. Do not recover diagnostic data by parsing human-readable exception strings. Add allowed metadata fields explicitly; otherwise `_metadata()` silently discards them.

| Field | Required meaning |
|---|---|
| `provider`, `operation`, `model` | ElevenLabs; STT or TTS; exact configured model |
| `failure_stage` | Connect/handshake, session start, send audio/text, wait transcript/first audio, stream, decode/validate or finalize |
| `error_category` | Authentication, permission, quota, rate limit, provider unavailable, timeout, network, premature close, invalid provider payload, invalid local input, cancelled or unknown |
| `provider_error_code` | Bounded provider machine code when present; use unknown when absent |
| `http_status`, `websocket_close_code` | Handshake response status or stream close code when actually available |
| `provider_request_id` | Request/correlation ID from explicitly allowlisted headers/events, if provided |
| `cause_type` | Original transport/protocol exception class, e.g. timeout versus connection failure |
| `retryable`, `retry_after_ms`, `attempt` | Explicit policy decision, provider retry delay when available and actual attempt number; unknown is distinct from false |
| `session_id`, `turn_id`, `elapsed_ms` | Correlation and duration on every failed stage and failed-turn event |
| `output_started`, `audio_bytes` | Whether partial speech was already emitted and how much; avoid unsafe full replay after partial delivery |

Category names above are **application taxonomy**, not claimed ElevenLabs event names. The PR maps available event fields and allowlisted WebSocket response status/request-ID headers; controlled provider readback must confirm what the deployed endpoint supplies. Preserve distinct raw machine code alongside category; never assume every 429 is the same quota/concurrency condition. If no code/request ID exists, report unknown explicitly.

Capture only bounded, scrubbed provider error detail if necessary. Exclude request headers, `xi-api-key`, TTS initialization payload (`xi_api_key`), arbitrary response bodies, input speech and generated text from error diagnostics. Full response logging is not an acceptable patch. Emit failure once per operation; correlate parent outcome without double-counting it as another provider attempt.

**Acceptance:** Focused fake tests cover provider rejection, timeout, sender failure, stalled TTS, malformed transcript/audio and redacted structured metadata. Broader handshake/status/close-code, partial-playback and cancellation drills plus controlled provider readback remain pending.

### 11. Timers and device-event delivery lack independent lifecycle telemetry

**Evidence:** `service/timers.py:51–117` claims, retries, heartbeats and acknowledges without observations. `deploy/cube-assistant-timer.service` does not load shared Langfuse environment; `device/timer_client.py` reports reconnects to stderr. Device-event broker and timer socket handlers have protocol/error handling but no tracing adapter.

**Failure case:** Timer accepted successfully, due later, then fails due to offline device, lease loss or speaker error. Original request trace cannot show eventual delivery failure.

**Patch:** Instrument timer due/claim/delivery/ack and local playback as separate operations, with timer/event ID, attempt and delay from due time. Cover local-tool delivery through same broker boundaries with safe action ID, verification outcome and idempotency/replay decision. Initialize/export telemetry in timer process; keep reconnect health available through journal. Never call a timer delivered until actual acknowledgement boundary; document whether acknowledgement means accepted, played or verified.

**Acceptance:** Offline device, reconnect, duplicate event, failed playback and max-attempt exhaustion each preserve unique delivery outcome. Link timer creation to delivery without keeping original trace open.

### 12. Stored answer, announced answer and heard answer are conflated

**Evidence:** Worker emits `job.result_persisted`, not user delivery. Voice has `voice.reply_ready` and first playback frame; `_ActivityCallbacks` ties transcript to root, but no universal per-turn speaker-drain outcome spans these paths. Notification code is separate from job execution.

**Failure case:** Result saves successfully then notification or final audio fails. Successful execution looks like a successful interaction.

**Patch:** Track result persisted, notification queued, result forwarded, synthesis completed, first playback write, playback drained, acknowledgement and interruption separately. Include stable result/turn/event ID at each hop. Label ALSA write/drain as software playback evidence, not proof a person heard sound. Keep hardware canary/manual confirmation for physical audio.

**Acceptance:** Disconnect after persistence and interrupt halfway through speech. Trace reports completed work plus failed/interrupted delivery, rather than a single success bit.

### 13. Queue, persistence and recovery boundaries hide the real delay

**Evidence:** `DurableJobWorker.run_once()` claims before lifecycle trace; graph wraps routing/model work but database/checkpoint/artifact operations largely appear through parent errors. Timer claim precedes delivery too. `ArtifactCleanupWorker` logs broad pass failures.

**Failure case:** Job never claims, lease expires, pool saturates, checkpoint stalls or disk fills. Model latency looks normal while user waits indefinitely.

**Patch:** Add named boundary observations for enqueue/claim/checkpoint/result commit/artifact write and cleanup. Derive queue age, stale lease count, retry count, storage failures and pool wait from existing durable records plus lightweight metrics. Start with boundaries and failure counts, not every SQL statement; exclude SQL parameters and connection strings. Reconcile accepted jobs against terminal results on a schedule.

**Acceptance:** Claim/database timeout, disk full and lost lease identify failing boundary; queue age alert fires for accepted work that never reaches model execution.

### 14. Release and policy changes cannot be reliably compared

**Evidence:** `LANGFUSE_RELEASE` is optional/blank in deployment example. Startup observation contains only component/status; `_FIELDS` excludes release/config/prompt/SDK provenance. `scripts/release_switch.py` changes symlink without trace manifest. Environment is configurable, which is useful, but actual per-component labels remain deployment-dependent.

**Failure case:** Mixed voice/API/worker canary releases or a Jev policy change alters behavior; traces cannot reliably identify exact code/config/prompt producing outcome.

**Patch:** Generate component release/Git SHA at packaging and include it with SDK version, model IDs, non-secret config fingerprint and prompt/policy hashes at startup and operation roots. Require environment and distinct device pseudonym; do not use mutable `current` as version. Use Langfuse prompt version links if prompts move there; local versioned prompt hashes suffice initially. Record per-component provenance because deployments may differ.

**Acceptance:** Compare canary and baseline by component release and policy hash; model/prompt-only change has a detectable identity change without exporting secrets.

### 15. Stable observation schema lacks contract checks and migration path

**Evidence:** `_name()` uses mutable human labels as actual observation names. `_FIELDS` silently drops unknown fields; `voice.turn.failed` callers override `stage` with failing provider stage. Repo search found no direct tests for `voice.observability` redaction, parentage, flush or failure isolation. Existing behavior tests do not establish exported payload correctness.

**Failure case:** Friendly label rename breaks saved filters/evaluators; new useful correlation fields disappear; SDK drift silently turns instrumentation into no-ops.

**Patch:** Treat names and metadata keys as versioned interface. Keep stable machine operation name and separate readable explanation; preserve old dashboards during deliberate migrations. Split `operation` from `failure_stage`. Add focused adapter tests and schema assertions in tests/development, while production drops invalid values with counted warnings. Cover disabled client, SDK exception, task context propagation, root/child IDs, content policy, terminal outcome and usage.

**Acceptance:** Assert complete exported tree for a representative multi-turn and worker case. A deliberate name/field change fails contract check until query/schema migration is included.

### 16. Dashboards and alerts need independent denominators

**Evidence:** Health response reports PostgreSQL only; no repository-level observability metrics endpoint/alert configuration was found in inspected paths. Connected project lists are empty, subject to project caveat above.

**Failure case:** Only successful requests emit traces; dashboard reports near-perfect success while failed startup/session attempts vanish. Low household traffic makes percentage alerts noisy.

**Patch:** Count attempted/accepted/rejected/terminal operations using local lifecycle records and durable state. Dashboard by environment and component release: attempt-to-session conversion, terminal outcome counts, empty STT after speech, fallback rate, stage latency, first playback/drain, timer lateness, queue age, tool verification and telemetry readback lag. Alert outside Langfuse for heartbeat/export silence and process death. Use Langfuse metric alerts for observed failure/fallback/latency. Start count/age thresholds and show sample size; tune percentiles only after enough data.

**Acceptance:** Zero traffic, traffic with no exported traces and all-success traffic are visibly distinct. Deliver one real alert through approved channel and test recovery/noise suppression during implementation.

### 17. Evaluation results are disconnected from production incidents

**Evidence:** `evaluation/report.py` reports routing, success, correction, latency and cost from supplied observations; replay/Jev experiments exist. CI runs ordinary tests and contract validation, without Langfuse trace-shape or release experiment gate. No trace-linked score write was found in these evaluation paths.

**Failure case:** Prompt tweak fixes one anecdote but regresses follow-up rejection, false wake or timer actions. Good trace detail does not measure outcome quality.

**Patch:** Reuse existing case/report formats. Add optional trace/observation IDs and release/policy identity; map deterministic outcomes to idempotent scores. Build sanitized incident regression cases for empty STT, false accept/reject, wrong route, timeout, fallback, tool verification, persistence and delivery. Human-label correctness/corrections first; do not infer that user silence means success. Add deterministic CI gates and separately approved provider-backed canary experiments. Introduce calibrated judges only for subjective quality after labeled examples exist.

**Acceptance:** Incident can be replayed as a case; baseline/candidate comparison identifies individual regressions and missing cases. Test-only traffic is excluded from household monitoring.

### 18. SDK upgrade will break root input/output without an explicit migration

**Evidence:** The lock remains Langfuse Python 4.17.0 (`>=4,<5`). This PR removes the removed trace-IO setter and private `_raw._otel_span` access from the adapter; custom ID generation and manual remote parenting still depend on SDK/OTel behavior. Current Python SDK v5 has breaking changes and server-version requirements.

**Failure case:** A broad dependency upgrade makes root content update throw; adapter swallows it to preserve voice, leaving apparently healthy but incomplete traces.

**Patch:** Keep reproducible v4 lock until migration is deliberate. Confirm actual server compatibility before changing SDK major version. Validate distributed context, naming/filtering, types, content policy and usage with exact SDK/OTel versions. Do not add new legacy API endpoints; current Langfuse index announces legacy Cloud endpoint removal on 16 November 2026.

**Acceptance:** Real readback verifies root IO, one root per intended unit of work, children/links, usage and safe errors on candidate SDK. Rollback preserves working v4 release/config.

### 19. Error detail loss extends beyond ElevenLabs, including cleanup and local state

**Evidence:** Gateway generic handler (`service/durable_api.py:208`) ignores the exception and returns `INTERNAL_ERROR` without explicit diagnostic logging in that handler. Gateway teardown (`service/live_gateway.py:260–311`) suppresses provider close/abort, usage persistence and session-close failures. `voice/jobs.py:_snapshot` reduces rejected API responses to HTTP status, dropping structured API error code/trace identity; `_persist_link` logs SQLite failure but returns normally. `device/job_notifications.py:220–280` has several type-only warnings without job correlation. `service/artifact_cleanup.py:40` silently continues on individual artifact deletion errors. `device/alsa.py` sends both capture and playback subprocess stderr to `DEVNULL`; audio runtime later reduces failures to stream plus exception class.

**Failure case:** Distinct gateway bugs all return the same internal error; a successful main operation hides failed close/usage save; link persistence failure makes restart recovery lose a job; persistent artifact deletion fails without per-item evidence; an ALSA device-busy/config failure is reduced to process exit or `AudioDeviceError`.

**Patch:** Apply shared structured error contract at every trust/process boundary, not just speech adapters. Preserve safe internal cause/code/fingerprint in operational evidence while keeping public error messages generic. Record server request trace ID alongside client operation ID; parse only validated allowlisted API error fields. Give cleanup its own child outcome without replacing primary failure. Distinguish durable-state write success from in-memory success. Include job/revision, artifact pseudonymous ID and audio stream/process exit status where relevant. Capture a bounded scrubbed ALSA stderr tail with asynchronous draining so subprocess pipes cannot deadlock; avoid full command/environment dumps. Count suppressed failures even when retry is correct.

**Acceptance:** Two different unexpected gateway exceptions produce distinguishable internal fingerprints and safe public errors. Force provider-close/usage-save failure, local SQLite write failure, repeated artifact-delete failure and ALSA exit with diagnostic stderr. Every case is correlated and discoverable; primary outcome survives cleanup failure; sensitive data stays excluded.

### 20. A healthy process can hang without a failed-operation event

**Evidence:** `service/elevenlabs_tts.py:86–112` starts sender task then waits on socket receives; sender is awaited only when provider marks final output. A failed sender need not wake receiver. STT waits for committed transcript after its session-start deadline with no adapter-local transcript deadline. Muse provider accepts `turn_timeout_seconds=None` and wraps turn in that optional timeout; subscription waits (`service/muse_provider.py:570,596`) and Muse event polling can wait without a local progress deadline. Outer session duration policies may eventually end these waits, but that does not reveal the blocked substage. `MuseGadgetHost.run` waits on stop rather than SDK task completion; `wait_for_link` discovers stopped SDK task only on a later link request and loses original task cause. Audio queue drain waits in `device/audio_runtime.py:214` precede ALSA's own bounded drain.

**Failure case:** TTS text sender dies while receiver waits; Muse subscription never becomes ready; STT stops emitting; SDK loop dies while host task still runs. Service heartbeat remains fresh, but no user-visible progress or immediate cause is recorded. A broad eventual session timeout can mislabel the incident.

**Patch:** Observe independent progress and actual effective timeout configuration for each stage. Race sender/receiver failures and supervise SDK task completion; retrieve original task exception safely. Set finite stage deadlines for connection/session readiness, transcript commit, first reply/audio, stream inactivity and queue drain. Preserve overall work deadlines separately; permit intentionally long research with heartbeat and durable progress rather than arbitrary short voice limits. Record last successful stage, time since progress, active task, queue depth, timeout source/budget and cause. Implement an overdue-operation detector independent of the hung task. Validate actual production composition before claiming a specific stage is unbounded there.

**Acceptance:** Fail sender while receive blocks, never signal chat subscription, withhold committed transcript, stop stream mid-response, kill SDK run loop and stall speaker consumption. Each reaches an identifiable terminal timeout/failure or overdue alert within its configured budget; process heartbeat alone cannot satisfy health. Test cancellation during each wait and verify remaining tasks are closed without hiding cause.

## Implementation order

| Batch | Deliverable | Findings | Exit condition |
|---|---|---|---|
| 1 | Confirm project, persistent failure evidence, redaction fix, truthful terminal status | 1–6, 19 | Failed wake/provider/export each discoverable; no credential leak |
| 2 | Turn identity, async delivery and research/provider visibility | 7–13, 20 | Every supported path has start/outcome or overdue detection |
| 3 | Release manifest, telemetry contract tests, dashboards and external alerts | 14–16 | Mixed-release incident locatable; telemetry loss alerts independently |
| 4 | Incident regression dataset and controlled SDK migration | 17–18 | Candidate compared against baseline; real ingestion/IO verified |

Keep application observability adapter as shared seam. Extend existing durable job/timer tables and journal first; do not introduce a separate event bus or broad automatic HTTP/SQL tracing for this load. Retain meaningful manual stages. If a framework integration is adopted for research, restrict capture deliberately and avoid duplicate instrumentation.

## Telemetry contract and remaining work

The adapter now carries some of these fields. This contract still needs full exported-tree assertions and production verification:

- Every unit of work: stable operation name/schema version, operation ID, UTC timestamp, component, environment, component release and eventual outcome. Record monotonic elapsed duration inside one process; monitor clock sync before comparing host timestamps.
- Optional correlations when applicable: attempt ID, conversation/session ID, turn ID, trace/span ID, job/delegation ID, timer/event/action ID, parent operation and attempt count. Do not overload application `request.trace_id` and Langfuse trace ID; document their mapping.
- Failure: reason enum, failure stage, retryability, timeout budget, provider/error code and sanitized provider request ID. Keep restricted stack frames or fingerprint locally for programming errors; scrub exception messages and locals before any remote export.
- Success: distinguish semantic completion, persistence and delivery. A provider returning 200 is not verified device action or complete audible playback.
- Health: process heartbeat, task status, capture progress, oldest queued work, export/readback age and storage capacity. Metrics labels stay low-cardinality; IDs belong in events/spans.
- Content: disabled means no text in any sink; enabled means bounded masked payloads, explicit truncation markers, access controls and retention. Separate operational retention from conversation retention.

## Acceptance drills for patch PRs

Run remaining drills with fakes first; use real household hardware/provider checks only in controlled canary rollout. Focused PR tests cover selected redaction, provider timeout, timer timeout, playback-drain timeout and background-task health cases; they do not establish production readiness.

| Injected condition | Required evidence |
|---|---|
| Invalid config or audio device unavailable before runtime | Boot-stage fatal record; external inactive-unit alert |
| Wake accepted, session HTTP times out | Local attempt ID, failed create boundary, no false success |
| Socket auth/session claim rejection | Sanitized admission reason and request/attempt correlation |
| VAD speech but empty STT | Audio/VAD measurements, STT result and recovery/silent decision |
| Jev fails; fallback succeeds/fails | Separate attempts and final route/outcome, degraded flag |
| Muse never replies or stream ends early | Upstream ID, first-text/completion timing, timeout/close cause |
| TTS partial output then exception | Failed synthesis plus actual playback progress and interruption |
| Speaker writer fails or drain never completes | Output failure, timeout and delivery outcome; no “heard” claim |
| Research tool error/recursion limit | Inner generation/tool sequence and exact validation/limit reason |
| Worker lease loss or kill after result generation | Attempt IDs, abandoned/stale lifecycle, recovery attempt and ledger reconciliation |
| Result commit/artifact storage fails | Named persistence stage, error code, cleanup outcome |
| Timer device offline/duplicate/max attempts | Due-to-delivery delay, retry/replay decision, eventual terminal/overdue state |
| Background task dies with API still running | Failed task supervision, degraded readiness and alert |
| Langfuse DNS/401/429/outage | Local evidence continues, exporter health degrades, external readback alert |
| Abrupt power loss | Missing heartbeat and stale operations detected after grace period; bounded unrecoverable tail stated |
| Content off; adversarial secrets at size limits | No text/secret in captured metadata, IO, status messages or local structured sink |
| Cleanup/usage persistence fails after main work | Separate secondary failure, primary outcome preserved, safe correlation |
| TTS sender/SDK task dies while host/receiver waits | Original task cause surfaced; failure races stop sibling tasks |
| SQLite fails, artifact deletion repeats, ALSA exits | Durable-state failure and scrubbed bounded subprocess cause remain discoverable |
| Stage stops progressing but process stays alive | Progress deadline/overdue detection, blocked stage and last-progress age |
| SDK/config/policy upgrade | Trace contract passes; release provenance and canary readback agree |

## First dashboards and signals to implement

| Priority | Signal | Source | Decision |
|---|---|---|---|
| P0 | Attempt has no session/terminal event | Local lifecycle ledger/journal | Admission/network incident |
| P0 | Component heartbeat/task stale | External watcher | Restart/escalate independent of Langfuse |
| P0 | Canary missing from intended project | Export readback monitor | Telemetry incident; dashboards incomplete |
| P1 | Turn outcome and stage failure count | Turn root/events | Identify failing boundary by release |
| P1 | Empty STT conditioned on sustained VAD speech | Existing VAD/STT metadata | Distinguish input quality from recognizer failure |
| P1 | Time to first playback and full drain | Device events + turn identity | Responsiveness versus incomplete delivery |
| P1 | Queue age, timer lateness and exhausted retries | PostgreSQL durable state | Stalled work/delivery |
| P1 | Tool verification failure and duplicate suppression | Existing verified action/result boundary | Device action correctness |
| P2 | False accept/reject and correction rate | Human labels + existing replay cases | Tune gate/prompt without anecdotal regressions |
| P2 | Measured usage/cost by exact provider/model | Generation usage + application ledger | Budget and retry waste; show unknown coverage |

Choose thresholds after confirming identity and collecting a baseline. For low traffic, initially alert on sustained stale heartbeat, explicit fatal state, accepted work past its configured deadline, and missing canary after ingestion grace. Do not invent a p95 service target without measured data or mix ignored ambient input into accepted-turn success denominators.

## Sources

Documentation fetched during audit; link targets may evolve. Apply examples matching pinned SDK/server.

- [Langfuse trace best practices](https://langfuse.com/docs/observability/best-practices): unit of work, sessions, stable names, meaningful root IO, generation/tool types and provenance.
- [Event queuing and batching](https://langfuse.com/docs/observability/features/queuing-batching): asynchronous batching, flush/shutdown and non-throwing flush failure behavior.
- [Advanced SDK features](https://langfuse.com/docs/observability/sdk/advanced-features): custom export filtering replaces default filter; filtered parents can orphan children.
- [Python SDK v4 to v5 migration](https://langfuse.com/docs/observability/sdk/upgrade-path/python-v4-to-v5): removed setters and minimum server compatibility.
- [Current PyPI package metadata](https://pypi.org/pypi/langfuse/json): 5.0.4 candidate version observed during audit.
- [Langfuse documentation index](https://langfuse.com/llms.txt): current API migration notice and documentation navigation.
- Repository evidence: `docs/langfuse-observability.md`, `docs/langfuse-trace-guide.md`, `voice/observability.py`, `device/voice_runtime.py`, `service/live_gateway.py`, `service/muse_provider.py`, `service/job_graph.py`, `service/job_worker.py`, `service/research_agent.py`, `service/timers.py`, `device/timer_client.py`, `service/durable_api.py`, `evaluation/report.py`, deployment templates and CI.

**Not checked:** physical Pi/canary behavior, deployed unit/env contents, actual Langfuse project identity/server version/retention/access, upstream Muse internals, real provider billing or alert delivery. No claim that all repository paths or all production failure modes were exhaustively verified. Patch acceptance drills remain required before calling observability complete.
