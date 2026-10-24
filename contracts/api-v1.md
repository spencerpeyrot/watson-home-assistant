# Internal API contract v1

P00 route and error-code draft. Base path `/v1`; JSON is UTF-8. Owner identity comes from authenticated server context. Device routes require a device credential; body `device_id` must match it. P06 implements bearer authentication and fixed process-environment mappings; host provisioning and TLS termination remain deployment choices.

## HTTP routes

| Method and path | Auth scope | Request | Success response |
|---|---|---|---|
| `POST /v1/sessions` | device | `SessionCreateRequest` | `SessionResponse` — 201 |
| `GET /v1/sessions` | owner | optional `limit` query | `SessionListResponse` — 200 |
| `DELETE /v1/sessions/{session_id}` | device | none | empty — 204 |
| `POST /v1/jobs` | device or owner | `TaskRequest` | `JobAccepted` — 202 |
| `GET /v1/jobs` | owner | optional `limit` query | `JobListResponse` — 200 |
| `GET /v1/jobs/{job_id}` | owner | none | `JobSnapshot` — 200 |
| `GET /v1/jobs/{job_id}/events` | owner | `Last-Event-ID` optional | `JobEvent` SSE — 200 |
| `POST /v1/jobs/{job_id}/corrections` | owner | `JobCorrectionRequest`; `Idempotency-Key` header | `JobSnapshot` — 202 |
| `POST /v1/jobs/{job_id}/cancel` | owner | `Idempotency-Key` header | `JobSnapshot` — 202 |
| `GET /v1/approvals` | owner | none | `ApprovalListResponse` — 200 |
| `POST /v1/approvals/{approval_id}/decision` | owner | `ApprovalDecisionRequest`; `Idempotency-Key` header | `ApprovalDecisionResponse` — 200 |
| `POST /v1/timers` | device or owner | `TimerCreateRequest` | `TimerRecord` — 201 |
| `GET /v1/timers` | owner | none | `TimerListResponse` — 200 |
| `DELETE /v1/timers/{timer_id}` | owner | none | empty — 204 |
| `GET /v1/memories` | owner | none | `MemoryListResponse` — 200 |
| `POST /v1/memories` | owner | `MemoryWriteRequest`; `Idempotency-Key` header | `MemoryRecord` — 201 |
| `PATCH /v1/memories/{memory_id}` | owner | `MemoryUpdateRequest`; `Idempotency-Key` header | `MemoryRecord` — 200 |
| `DELETE /v1/memories/{memory_id}` | owner | `Idempotency-Key` header | empty — 204 |
| `GET /v1/artifacts/{artifact_id}` | owner | none | artifact bytes — 200 |
| `DELETE /v1/artifacts/{artifact_id}` | owner | none | empty — 204 |
| `GET /v1/usage` | owner | optional `session_id` query | `UsageResponse` — 200 |
| `GET /v1/health` | operator | none | `HealthResponse` — 200 |
| `GET /v1/events` | owner | `Last-Event-ID` optional | `JobEvent` SSE — 200 |

`POST /v1/jobs` returns the existing job when `idempotency_key` repeats with the same canonical request. Same key with a different request returns `IDEMPOTENCY_CONFLICT`. Gateway derives key from authenticated device, session, and delegation IDs. Corrections must use current revision plus one; stale writes/results cannot replace a newer revision. Cancel stops queued/running work where possible; it cannot reverse a committed external action.

On the authenticated voice WebSocket, each VAD-bounded PCM turn ends with an `AudioTurnEnd` message. It includes the last audio sequence and optional `vad_speech_ms`, the amount of that turn marked as speech by the Pi's local VAD. Gateway validates the marker before forwarding it to the voice provider.

The durable HTTP gateway requires `Idempotency-Key` on job correction/cancel, approval decisions, and memory mutations. Exact retries replay the original result; key reuse for a different request returns `IDEMPOTENCY_CONFLICT`. Timer creation carries its key in `TimerCreateRequest`.

`GET /v1/health` accepts an operator credential without an owner mapping. Resource routes require an owner mapping resolved by authentication. No route accepts `owner_id` from a request body.

```mermaid
sequenceDiagram
    participant V as Voice adapter
    participant G as Gateway
    participant O as Orchestrator
    participant D as Device executor
    V->>G: finalized transcript + delegation metadata
    G->>G: correlate IDs; commit TaskRequest
    G->>O: TaskRequest
    O-->>G: JobAccepted
    G-->>V: JobAccepted
    O-->>G: JobEvent(status/progress)
    G-->>V: correlated events
    O->>G: LocalToolRequest
    G->>D: device WebSocket relay
    D-->>G: ToolResult with observed state
    G-->>O: correlated ToolResult
    O-->>G: terminal JobEvent + artifact refs
    G-->>V: final status/result
```

Voice adapter must not submit a task until transcript context is committed. Corrections advance `context_revision`; workers discard late results from older revisions.

## Error envelope

Every non-2xx JSON response uses `ApiErrorResponse`:

```json
{
  "schema_version": 1,
  "trace_id": "00000000-0000-4000-8000-000000000004",
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Job not found.",
    "retryable": false,
    "details": null
  }
}
```

Stable codes and HTTP mapping:

| HTTP | Codes |
|---|---|
| 400 | `INVALID_REQUEST`, `UNSUPPORTED_SCHEMA_VERSION` |
| 401 | `AUTHENTICATION_REQUIRED` |
| 403 | `PERMISSION_DENIED`, `ACTION_DENIED` |
| 404 | `RESOURCE_NOT_FOUND` |
| 409 | `RESOURCE_CONFLICT`, `IDEMPOTENCY_CONFLICT`, `STALE_CONTEXT_REVISION`, `APPROVAL_PAYLOAD_MISMATCH`, `JOB_NOT_CANCELLABLE` |
| 410 | `APPROVAL_EXPIRED` |
| 429 | `RATE_LIMITED`, `BUDGET_EXCEEDED` |
| 503 | `DEPENDENCY_UNAVAILABLE`, `DEVICE_OFFLINE` |
| 504 | `DEPENDENCY_TIMEOUT` |
| 500 | `INTERNAL_ERROR` |

`retryable` describes retrying the same operation safely. A write timeout with unknown external outcome returns a `ToolResult` in `needs_reconciliation`; caller reconciles state before retry. Do not encode provider-specific errors as public codes.

The durable HTTP app factory wires owner-scoped persistence routes and finite SSE snapshots for one job or all owner jobs. Clients resume with `Last-Event-ID`; the server returns events after that cursor, and clients reconnect to fetch later events. Authentication is injected by deployment. The voice WebSocket is implemented for authenticated audio relay and delegation context append; the separate device event WebSocket is implemented for timer ACK and correlated local-tool results. The worker model tools dispatch through the authenticated internal API route to this device socket. Pi execution requires an explicitly configured Home Assistant URL, token, and matching light allowlist. Health and playback-meter message kinds remain draft targets.

Artifact deletion first hides owner-scoped metadata, then a background cleanup worker removes the blob and purges its tombstone. Storage failure leaves the tombstone for retry; repeating the delete remains safe.

## Event transports

- SSE uses `Content-Type: text/event-stream`; event `id` is `{job_id}:{sequence}` and `data` is one serialized `JobEvent`. `Last-Event-ID` resumes after that sequence. Dashboard stream scopes events to authenticated owner.
- Device opens outbound authenticated WebSocket at `/v1/devices/{device_id}/connect?session_id={session_id}` for the implemented voice relay. It carries audio metadata/binary frames, provider-neutral voice events, and bounded context append for an observed client delegation. No raw provider event crosses this boundary. Device health and playback-meter messages remain draft targets.
- Device opens outbound authenticated WebSocket at `/v1/devices/{device_id}/events` for timer notifications and local actions. Timer messages use `timer_fired`/`timer_ack`; local actions use `local_tool_request`/`local_tool_result`, matched on action, device, conversation, session, delegation, turn, job, trace, and context revision IDs. The worker model tools dispatch through the authenticated internal API route to this socket. When configured, the device event client executes only the exact Home Assistant light allowlist. Device SQLite persists results before response; an interrupted action becomes `needs_reconciliation`, never automatic re-execution. Timer redelivery remains at-least-once with inbox deduplication. Live Home Assistant credentials and physical verification remain pending.
- Audio metadata declares encoding, sample rate, channels, frame duration, and capture time. Queues are bounded; device discards stale playback on reconnect. Local mute takes effect before network work.
- Playback meter comes from samples actually sent to the speaker, not generated tokens or backend activity.

## Approval binding

Approval is owner-scoped and bound server-side to action ID, exact payload digest, resource, context revision, and expiry. Decision request must echo `payload_digest` and `context_revision`; a payload/revision change invalidates the approval. `Idempotency-Key` is required for approval decisions and memory mutations. Unknown actions remain denied in application policy.

## Versioning

All messages carry `schema_version: 1`. Unknown fields are rejected. Additive optional fields may ship within v1 after consumer review; changing required fields, enum behavior, or meaning requires a new schema version. Every route error includes `trace_id`.
