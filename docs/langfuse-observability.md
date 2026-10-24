# Watson Langfuse observability

## Follow-up recovery canary — 23 October 2026, 19:00 EDT

- API `canary-release` and voice `canary-release` run together on the Pi canary; V1 and shared `current` remain unchanged.
- Each turn trace now includes PCM bytes/duration, local VAD speech time, RMS, peak, and clipped sample count. PCM itself is not retained.
- Empty transcription recovery records whether Watson had invited a reply and the VAD speech duration. It asks once only after an open/confirmation invitation and at least 300 ms of VAD speech; other empty turns stay quiet.
- Both units active with zero restarts; systemd verification passed; API `/openapi.json` returned 200. Pi voice/API focused tests passed 44/44. Physical test and Langfuse ingestion check await a real conversation.

## Latest canary update — 23 October 2026, 18:12 EDT

- API canary now runs `codex-observability-20261023-4`; Pi voice canary runs `codex-observability-20261023-3`. V1 and stored data unchanged.
- Turn tasks inherit the session trace. Pi records local speech-candidate and microphone-muted transitions; Jev route confidence, fallback, reply timing, follow-up classification, and close reason are visible under the same session.
- Startup observations flush. Session close flushes run off the audio loop. Trace SDK failures log safe error types and do not block speech.
- Both services active with zero restarts; `systemd-analyze verify` passed; canary API `/openapi.json` returned 200. Candidate Python files compile on the Pi.
- Direct read using the Pi's configured project found startup observations for `voice` and `gateway`. The connected Langfuse tool returned no row for the exact Pi trace ID, which suggests it is connected to a different project. Physical conversation trace still needs a user test.
- No `.env` or recordings copied. New checkpoints store lengths, confidence, state, and timing; existing `CUBE_LANGFUSE_CAPTURE_CONTENT=true` still exports text as documented below.

## Current rollout — 8 October 2026, 21:22 EDT

- Voice: `immutable release`; API: `source-20261008-api-4`; worker: `source-20261008-worker-3`.
- Timer stays timer2; shared `current` stays v2. All four services active, zero restarts.
- SDK pinned 4.17.0; new dedicated OTel provider startup ingestion confirmed for all three instrumented components.
- New grouping confirmed on real user-labeled false wake at 21:23:55 EDT: one root with gateway/bridge and nested Diagnostics. Job/graph input/output confirmed on actual NYC request at 21:28 EDT. Root transcript binding corrected on voice17 after callback order defect; new root content acceptance pending. No synthetic requests or automated tests run.
- Earlier metadata-only export policy changed by explicit user request. Historical traces keep their previous roots/names and missing text.

Read [trace guide](langfuse-trace-guide.md) for the tree, meanings, input/output,
debugging steps and limits. Rollout history lives in
[Watson handoff](openwakeword-watson-model.md).

## Configuration

`/etc/cube-assistant/langfuse.env` is root:root mode 0600, loaded by API/worker/voice.
It holds `CUBE_LANGFUSE_ENABLED=true`, `CUBE_LANGFUSE_CAPTURE_CONTENT=true`, exact
project `LANGFUSE_BASE_URL`, project public/secret keys, and pi-development tracing
environment. Credentials originated privately from local `.env`; never copy or
print protected configuration. Example: `deploy/langfuse.env.example`.

Set capture flag false and restart API/worker/voice to stop future text capture
while retaining metadata. Set tracing-enabled false to disable all export.
Existing Langfuse data is not removed by disabling capture.

Unit backups: `/etc/cube-assistant/langfuse-tree-unit-backups-20261008`. Restore
those units, daemon-reload and restart to return to voice15/API2/worker2. Protected
configuration is separate; rollback does not automatically reset capture flag.

## Tree and data

Device owns one real root per session (`session_id.hex` trace ID, deterministic
root span ID). Remote components attach through OTel parent contexts. Background
job execution inherits its lifecycle span; graph/model/tools are nested below it.
SDK `trace_context` is avoided because SDK 4.17 marks those component spans AS_ROOT.
High-volume transcript/endpoint/audio measurements live under Diagnostics.

Every span has a readable label, internal `stage` and `explanation`. Only explicit
`cube.*` observations exported; no automatic LangChain/OpenAI callbacks. Session
UUID groups each trace; individual questions appear through clock/submit/job
spans. Live transcript segments have approximate turn boundaries.

Content opt-in exports recognized speech, requests/answers, model messages and
tool arguments/results. Known environment credential values, sensitive field
names and common key/Bearer patterns are redacted. Personal conversation/context
can still be present. Payload limits/redaction details in trace guide. No audio,
full provider envelopes, environment dumps or object reprs exported.

SDK/export failures are fail-safe and cannot block conversation or job work.
Startup initialization stays off the audio loop; SDK batches exports, shutdown
flushes in a thread. Missing device telemetry can leave remote children referencing
a missing root; abrupt process shutdown can leave unfinished observations.

References: [SDK instrumentation](https://langfuse.com/docs/observability/sdk/instrumentation),
[distributed tracing](https://langfuse.com/docs/observability/features/trace-ids-and-distributed-tracing),
[Python SDK](https://python.reference.langfuse.com/langfuse).


Latest audio diagnostics: audio.input_levels and audio.forwarded_levels show numeric frame count, cumulative RMS/peak/clipped count under Diagnostics; no PCM exported. Initial-question 10-second readiness window differs from five-second post-reply inactivity. Voice18/API4 deployed; see current handoff for acceptance status.
