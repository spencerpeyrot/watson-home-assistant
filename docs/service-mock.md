# Authenticated service mock

P06 draft gateway mock. It uses the v1 Pydantic contracts, in-memory state, bearer credentials, and an authenticated device WebSocket. It does not call OpenAI, Home Assistant, or other external services. All state disappears when the process exits.

## Run locally

Requires Python 3.12 or 3.13 and `uv`.

```sh
cp service/mock.env.example .env.mock
```

Put three distinct random values of at least 32 characters in `.env.mock`. Generate each with:

```sh
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Load only the mock configuration, then start the server:

```sh
set -a
. ./.env.mock
set +a
uv sync --locked
uv run python -m service.mock_server
```

The server binds to `127.0.0.1:8080`, disables access logs, and never logs request bodies or credentials. Open `http://127.0.0.1:8080/dashboard` for the same-origin mock-first dashboard. Paste only the mock owner token from `.env.mock`; the page keeps it in memory and clears it on disconnect. `GET /healthz` is unauthenticated liveness; `/v1/health` requires the operator token. Do not bind this mock to a public interface.

## Credential scopes

| Credential | Allowed routes |
|---|---|
| Device | Session create/close, job submit, timer create, matching device WebSocket |
| Owner | Session/job lists, job submit/read/events/correction/cancel, approvals, timers, memories, artifacts, usage, all-events stream |
| Operator | `/v1/health` |

The device token is bound to `CUBE_MOCK_DEVICE_ID`. Device job bodies and WebSocket paths must use that ID. Error responses use `ApiErrorResponse`; invalid bodies return 400 without echoing submitted values.

## Mock behavior

- Sessions return generated IDs. Closing an unknown session succeeds as a mock no-op.
- Owner session/job lists power the mock dashboard. Session state and jobs remain in memory and vanish on restart.
- Jobs stay in memory, emit an initial progress event, and deduplicate identical `idempotency_key` submissions. Reusing a key with a different body returns `IDEMPOTENCY_CONFLICT`.
- Corrections must use the next `context_revision`; cancellation changes a running mock job to `cancelled`.
- Approval and memory routes use fixture-backed initial records plus in-memory mutations. Approval decisions and all memory mutations require `Idempotency-Key`.
- Timer routes validate `TimerCreateRequest`, deduplicate exact body retries by its idempotency key, reject changed-payload reuse, and allow owner listing/cancellation. Mock timers remain in memory and are not delivered.
- Usage returns the checked-in example with a current timestamp. Artifact fetch returns not found because mock artifacts are not stored.

The device WebSocket accepts JSON envelopes shaped as `{"type":"device_health","payload":{...}}`, using `device_health`, `playback_meter`, `tool_result`, or `audio_frame_metadata` as type names. Payloads validate against the matching contract model. Audio metadata must declare PCM16 little-endian and stay within 48 kHz, two channels, and 100 ms; next binary message must contain exactly `sample_rate_hz * channels * 2 * frame_duration_ms / 1000` bytes. The mock acknowledges valid messages and discards their content.

## Package lock and target support

`pyproject.toml` constrains the runtime to Python 3.12–3.13 and separates the `device` dependency group (`httpx`, Pydantic, and `websockets`) from the default `server` group, which adds FastAPI, OpenAI, LangChain, LangGraph, PostgreSQL, Uvicorn, and Deep Agents. `uv.lock` records the combined resolved graph while `uv sync --no-default-groups --group device --no-dev` installs only device dependencies. A local disposable Python 3.13 device-group install prepared 12 packages and imported the timer/Home Assistant composition without server/agent packages. On 6 October 2026, the previous full locked install prepared 58 packages on the Pi's aarch64 Debian 13/Python 3.13.5 environment and imported FastAPI, OpenAI, LangChain, LangGraph PostgreSQL checkpointing, WebSocket, and psycopg modules successfully. The temporary source bundle, virtual environment, uv runtime, and caches were removed. Minimal device-group ARM64 install and clean cloud-host installation remain pending.
