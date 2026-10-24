"""Authenticated in-memory API and device WebSocket mock.

This mock exercises transport boundaries only. It is not a production gateway:
state disappears on restart, and it performs no external actions.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, Header, Request, Response, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from contracts.models import (
    ApiErrorResponse,
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    ApprovalListResponse,
    ApprovalRecord,
    AudioFrameMetadata,
    DeviceHealthEvent,
    HealthResponse,
    JobAccepted,
    JobCorrectionRequest,
    JobEvent,
    JobListResponse,
    JobSnapshot,
    MemoryListResponse,
    MemoryRecord,
    MemoryUpdateRequest,
    MemoryWriteRequest,
    PlaybackMeterEvent,
    SessionCreateRequest,
    SessionListResponse,
    SessionResponse,
    TaskRequest,
    TimerCreateRequest,
    TimerListResponse,
    TimerRecord,
    ToolResult,
    UsageResponse,
)


FIXTURES = Path(__file__).resolve().parents[1] / "contracts" / "fixtures"
DASHBOARD_ROOT = Path(__file__).resolve().parent.parent / "dashboard"
bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Settings:
    device_id: str
    device_token: str
    owner_token: str
    operator_token: str

    @classmethod
    def from_environment(cls) -> Settings:
        names = {
            "device_id": "CUBE_MOCK_DEVICE_ID",
            "device_token": "CUBE_MOCK_DEVICE_TOKEN",
            "owner_token": "CUBE_MOCK_OWNER_TOKEN",
            "operator_token": "CUBE_MOCK_OPERATOR_TOKEN",
        }
        values = {key: os.environ.get(env_name, "") for key, env_name in names.items()}
        missing = [names[key] for key, value in values.items() if not value]
        if missing:
            raise RuntimeError("Missing required mock configuration: " + ", ".join(missing))
        if not values["device_id"].strip():
            raise RuntimeError("CUBE_MOCK_DEVICE_ID must not be blank")
        for key in ("device_token", "owner_token", "operator_token"):
            if len(values[key]) < 32:
                raise RuntimeError(f"{names[key]} must contain at least 32 characters")
        tokens = [values[key] for key in ("device_token", "owner_token", "operator_token")]
        if len(set(tokens)) != len(tokens):
            raise RuntimeError("Mock role tokens must be distinct")
        return cls(**values)


@dataclass(frozen=True)
class Principal:
    role: Literal["device", "owner", "operator"]


@dataclass
class StoredJob:
    request: TaskRequest
    snapshot: JobSnapshot
    events: list[JobEvent] = field(default_factory=list)


@dataclass
class MockStore:
    sessions: dict[UUID, SessionResponse] = field(default_factory=dict)
    jobs: dict[UUID, StoredJob] = field(default_factory=dict)
    idempotency: dict[str, tuple[str, Any]] = field(default_factory=dict)
    approvals: dict[UUID, ApprovalRecord] = field(default_factory=dict)
    memories: dict[UUID, MemoryRecord] = field(default_factory=dict)
    timers: dict[UUID, TimerRecord] = field(default_factory=dict)

    @classmethod
    def from_fixtures(cls) -> MockStore:
        store = cls()
        approval = _fixture("approval_record.json", ApprovalRecord)
        memory = _fixture("memory_record.json", MemoryRecord)
        now = datetime.now(timezone.utc)
        store.approvals[approval.approval_id] = approval.model_copy(
            update={"expires_at": now + timedelta(minutes=5)}
        )
        store.memories[memory.memory_id] = memory
        return store


class MockError(Exception):
    def __init__(self, status: int, code: str, message: str, retryable: bool = False):
        self.status = status
        self.code = code
        self.message = message
        self.retryable = retryable


def _fixture(name: str, model: type[BaseModel]) -> Any:
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return model.model_validate(payload)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_digest(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")

    def json_default(item: Any) -> Any:
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        return str(item)

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _new_job_event(job: StoredJob, kind: str, summary: str) -> JobEvent:
    request = job.request
    event = JobEvent(
        schema_version=1,
        device_id=request.device_id,
        job_id=job.snapshot.job_id,
        event_id=uuid.uuid4(),
        sequence=len(job.events) + 1,
        kind=kind,
        status=job.snapshot.status,
        conversation_id=request.conversation_id,
        session_id=request.session_id,
        turn_id=request.turn_id,
        delegation_id=request.delegation_id,
        trace_id=request.trace_id,
        context_revision=job.snapshot.context_revision,
        summary=summary,
        timestamp=_now(),
    )
    job.events.append(event)
    job.snapshot = job.snapshot.model_copy(update={"last_sequence": event.sequence, "summary": summary})
    return event


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = Settings.from_environment()
        app.state.store = MockStore.from_fixtures()
        yield

    app = FastAPI(title="Cube Assistant Mock Gateway", version="0.1.0", lifespan=lifespan)
    app.mount(
        "/dashboard/assets",
        StaticFiles(directory=DASHBOARD_ROOT),
        name="dashboard-assets",
    )

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard_page():
        response = FileResponse(DASHBOARD_ROOT / "index.html", media_type="text/html")
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; "
            "img-src 'self' data:; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
        )
        return response

    @app.middleware("http")
    async def assign_trace_id(request: Request, call_next):
        request.state.trace_id = uuid.uuid4()
        return await call_next(request)

    @app.exception_handler(MockError)
    async def mock_error_handler(request: Request, error: MockError):
        response = ApiErrorResponse(
            trace_id=getattr(request.state, "trace_id", uuid.uuid4()),
            error={"code": error.code, "message": error.message, "retryable": error.retryable},
        )
        headers = {"WWW-Authenticate": "Bearer"} if error.status == 401 else None
        return JSONResponse(status_code=error.status, content=response.model_dump(mode="json"), headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, error: RequestValidationError):
        fields = [".".join(str(part) for part in item.get("loc", ()) if part != "body") for item in error.errors()]
        response = ApiErrorResponse(
            trace_id=getattr(request.state, "trace_id", uuid.uuid4()),
            error={
                "code": "INVALID_REQUEST",
                "message": "Request does not match the v1 contract.",
                "retryable": False,
                "details": {"fields": fields},
            },
        )
        return JSONResponse(status_code=400, content=response.model_dump(mode="json"))

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, error: StarletteHTTPException):
        status = error.status_code
        code = "RESOURCE_NOT_FOUND" if status == 404 else "INVALID_REQUEST"
        message = "Route not found." if status == 404 else "Request rejected."
        response = ApiErrorResponse(
            trace_id=getattr(request.state, "trace_id", uuid.uuid4()),
            error={"code": code, "message": message, "retryable": False},
        )
        return JSONResponse(status_code=status, content=response.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, _error: Exception):
        response = ApiErrorResponse(
            trace_id=getattr(request.state, "trace_id", uuid.uuid4()),
            error={
                "code": "INTERNAL_ERROR",
                "message": "Mock gateway failed to process the request.",
                "retryable": False,
            },
        )
        return JSONResponse(status_code=500, content=response.model_dump(mode="json"))

    async def authenticate(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> Principal:
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise MockError(401, "AUTHENTICATION_REQUIRED", "Bearer authentication required.")
        settings: Settings = request.app.state.settings
        role_tokens = (
            ("device", settings.device_token),
            ("owner", settings.owner_token),
            ("operator", settings.operator_token),
        )
        for role, token in role_tokens:
            if hmac.compare_digest(credentials.credentials, token):
                return Principal(role=role)  # type: ignore[arg-type]
        raise MockError(401, "AUTHENTICATION_REQUIRED", "Bearer token is invalid.")

    def require_role(principal: Principal, *roles: str) -> None:
        if principal.role not in roles:
            raise MockError(403, "PERMISSION_DENIED", "Credential lacks required scope.")

    def settings_for(request: Request) -> Settings:
        return request.app.state.settings

    def store_for(request: Request) -> MockStore:
        return request.app.state.store

    def job_for(store: MockStore, job_id: UUID) -> StoredJob:
        job = store.jobs.get(job_id)
        if job is None:
            raise MockError(404, "RESOURCE_NOT_FOUND", "Job not found.")
        return job

    def idempotency_header(value: str | None) -> str:
        if not value:
            raise MockError(400, "INVALID_REQUEST", "Idempotency-Key header is required.")
        return value

    @app.get("/healthz")
    async def liveness():
        return {"status": "healthy", "service": "cube-mock-gateway"}

    @app.get("/v1/health", response_model=HealthResponse)
    async def health(principal: Principal = Depends(authenticate)):
        require_role(principal, "operator")
        return HealthResponse(
            status="healthy",
            service="cube-mock-gateway",
            observed_at=_now(),
            dependencies={"memory_store": "healthy"},
        )

    @app.post("/v1/sessions", response_model=SessionResponse, status_code=201)
    async def create_session(
        body: SessionCreateRequest,
        request: Request,
        principal: Principal = Depends(authenticate),
    ):
        require_role(principal, "device")
        session_id = uuid.uuid4()
        response = SessionResponse(
            session_id=session_id,
            conversation_id=body.conversation_id,
            state="ready",
            opened_at=_now(),
            idle_close_seconds=30,
        )
        store_for(request).sessions[session_id] = response
        return response

    @app.get("/v1/sessions", response_model=SessionListResponse)
    async def list_sessions(request: Request, principal: Principal = Depends(authenticate)):
        require_role(principal, "owner")
        sessions = tuple(
            sorted(store_for(request).sessions.values(), key=lambda item: item.opened_at, reverse=True)
        )
        return SessionListResponse(sessions=sessions)

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    async def close_session(
        session_id: UUID,
        request: Request,
        principal: Principal = Depends(authenticate),
    ):
        require_role(principal, "device")
        store = store_for(request)
        session = store.sessions.get(session_id)
        if session is not None:
            store.sessions[session_id] = session.model_copy(update={"state": "closed"})
        return Response(status_code=204)

    @app.post("/v1/jobs", response_model=JobAccepted, status_code=202)
    async def submit_job(
        body: TaskRequest,
        request: Request,
        principal: Principal = Depends(authenticate),
    ):
        require_role(principal, "device", "owner")
        settings = settings_for(request)
        if principal.role == "device" and body.device_id != settings.device_id:
            raise MockError(403, "PERMISSION_DENIED", "Task device_id does not match credential.")
        store = store_for(request)
        idem_key = "job:" + body.idempotency_key
        digest = _canonical_digest(body)
        prior = store.idempotency.get(idem_key)
        if prior is not None:
            old_digest, prior_job_id = prior
            if not hmac.compare_digest(digest, old_digest):
                raise MockError(409, "IDEMPOTENCY_CONFLICT", "Idempotency key was used for another request.")
            job = job_for(store, prior_job_id)
            return JobAccepted(
                job_id=job.snapshot.job_id,
                status="running",
                trace_id=body.trace_id,
                deduplicated=True,
                first_event_sequence=1,
            )
        job_id = uuid.uuid4()
        snapshot = JobSnapshot(
            job_id=job_id,
            status="running",
            context_revision=body.context_revision,
            last_sequence=0,
            summary="Accepted by mock gateway.",
        )
        job = StoredJob(request=body, snapshot=snapshot)
        store.jobs[job_id] = job
        store.idempotency[idem_key] = (digest, job_id)
        _new_job_event(job, "progress", "Mock job accepted and waiting for a worker.")
        return JobAccepted(
            job_id=job_id,
            status="running",
            trace_id=body.trace_id,
            deduplicated=False,
            first_event_sequence=1,
        )

    @app.get("/v1/jobs/{job_id}", response_model=JobSnapshot)
    async def read_job(job_id: UUID, request: Request, principal: Principal = Depends(authenticate)):
        require_role(principal, "owner")
        return job_for(store_for(request), job_id).snapshot

    @app.get("/v1/jobs", response_model=JobListResponse)
    async def list_jobs(request: Request, principal: Principal = Depends(authenticate)):
        require_role(principal, "owner")
        jobs = tuple(
            sorted(
                (job.snapshot for job in store_for(request).jobs.values()),
                key=lambda item: (item.last_sequence, str(item.job_id)),
                reverse=True,
            )
        )
        return JobListResponse(jobs=jobs)

    def sse_response(events: list[JobEvent], last_event_id: str | None):
        start_index = 0
        if last_event_id:
            try:
                last_job_id_text, last_sequence_text = last_event_id.rsplit(":", 1)
                last_job_id = UUID(last_job_id_text)
                last_sequence = int(last_sequence_text)
                if last_sequence < 1:
                    raise ValueError
            except (IndexError, ValueError):
                raise MockError(400, "INVALID_REQUEST", "Last-Event-ID must use job-id:sequence format.")
            marker_index = next(
                (
                    index
                    for index, event in enumerate(events)
                    if event.job_id == last_job_id and event.sequence == last_sequence
                ),
                None,
            )
            if marker_index is None:
                raise MockError(400, "INVALID_REQUEST", "Last-Event-ID is outside the mock event stream.")
            start_index = marker_index + 1

        async def stream():
            for event in events[start_index:]:
                data = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
                yield f"id: {event.job_id}:{event.sequence}\nevent: {event.kind}\ndata: {data}\n\n"
                await asyncio.sleep(0)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/v1/jobs/{job_id}/events")
    async def job_events(
        job_id: UUID,
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        principal: Principal = Depends(authenticate),
    ):
        require_role(principal, "owner")
        return sse_response(job_for(store_for(request), job_id).events, last_event_id)

    @app.get("/v1/events")
    async def all_events(
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        principal: Principal = Depends(authenticate),
    ):
        require_role(principal, "owner")
        events = [event for job in store_for(request).jobs.values() for event in job.events]
        return sse_response(sorted(events, key=lambda event: (event.timestamp, str(event.job_id), event.sequence)), last_event_id)

    @app.post("/v1/jobs/{job_id}/corrections", response_model=JobSnapshot, status_code=202)
    async def correct_job(
        job_id: UUID,
        body: JobCorrectionRequest,
        request: Request,
        principal: Principal = Depends(authenticate),
    ):
        require_role(principal, "owner")
        job = job_for(store_for(request), job_id)
        if body.context_revision != job.snapshot.context_revision + 1:
            raise MockError(409, "STALE_CONTEXT_REVISION", "Correction must use the next context revision.")
        job.request = job.request.model_copy(
            update={"request_text": body.request_text, "context_revision": body.context_revision, "recent_turns": body.recent_turns}
        )
        job.snapshot = job.snapshot.model_copy(update={"context_revision": body.context_revision})
        _new_job_event(job, "correction_acknowledged", "Correction accepted by mock gateway.")
        return job.snapshot

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobSnapshot, status_code=202)
    async def cancel_job(job_id: UUID, request: Request, principal: Principal = Depends(authenticate)):
        require_role(principal, "owner")
        job = job_for(store_for(request), job_id)
        if job.snapshot.status not in ("queued", "running"):
            raise MockError(409, "JOB_NOT_CANCELLABLE", "Job is already terminal.")
        job.snapshot = job.snapshot.model_copy(update={"status": "cancelled"})
        _new_job_event(job, "status", "Job cancelled by owner.")
        return job.snapshot

    @app.get("/v1/approvals", response_model=ApprovalListResponse)
    async def list_approvals(request: Request, principal: Principal = Depends(authenticate)):
        require_role(principal, "owner")
        return ApprovalListResponse(approvals=tuple(store_for(request).approvals.values()))

    @app.post("/v1/approvals/{approval_id}/decision", response_model=ApprovalDecisionResponse)
    async def decide_approval(
        approval_id: UUID,
        body: ApprovalDecisionRequest,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        principal: Principal = Depends(authenticate),
    ):
        require_role(principal, "owner")
        store = store_for(request)
        key = "approval:" + idempotency_header(idempotency_key)
        digest = _canonical_digest({"approval_id": approval_id, "body": body})
        prior = store.idempotency.get(key)
        if prior is not None:
            if not hmac.compare_digest(digest, prior[0]):
                raise MockError(409, "IDEMPOTENCY_CONFLICT", "Idempotency key was used for another request.")
            return prior[1]
        record = store.approvals.get(approval_id)
        if record is None:
            raise MockError(404, "RESOURCE_NOT_FOUND", "Approval not found.")
        if record.status != "pending":
            raise MockError(409, "RESOURCE_CONFLICT", "Approval is no longer pending.")
        if record.expires_at <= _now():
            raise MockError(410, "APPROVAL_EXPIRED", "Approval has expired.")
        if not hmac.compare_digest(record.payload_digest, body.payload_digest) or record.context_revision != body.context_revision:
            raise MockError(409, "APPROVAL_PAYLOAD_MISMATCH", "Approval payload or revision changed.")
        status = "approved" if body.decision == "approve" else "denied"
        store.approvals[approval_id] = record.model_copy(update={"status": status})
        result = ApprovalDecisionResponse(approval_id=approval_id, status=status, decided_at=_now())
        store.idempotency[key] = (digest, result)
        return result

    @app.post("/v1/timers", response_model=TimerRecord, status_code=201)
    async def create_timer(
        body: TimerCreateRequest,
        request: Request,
        principal: Principal = Depends(authenticate),
    ):
        require_role(principal, "device", "owner")
        settings: Settings = request.app.state.settings
        device_id = settings.device_id
        store = store_for(request)
        key = f"timer:{device_id}:{body.idempotency_key}"
        digest = _canonical_digest({"device_id": device_id, "body": body})
        prior = store.idempotency.get(key)
        if prior is not None:
            if not hmac.compare_digest(digest, prior[0]):
                raise MockError(409, "IDEMPOTENCY_CONFLICT", "Idempotency key was used for another timer.")
            return prior[1]
        record = TimerRecord(
            timer_id=uuid.uuid4(),
            conversation_id=body.conversation_id,
            due_at=body.due_at,
            message=body.message,
            status="scheduled",
            created_at=_now(),
        )
        store.timers[record.timer_id] = record
        store.idempotency[key] = (digest, record)
        return record

    @app.get("/v1/timers", response_model=TimerListResponse)
    async def list_timers(request: Request, principal: Principal = Depends(authenticate)):
        require_role(principal, "owner")
        records = tuple(sorted(
            store_for(request).timers.values(),
            key=lambda item: (item.created_at, str(item.timer_id)),
            reverse=True,
        ))
        return TimerListResponse(timers=records)

    @app.delete("/v1/timers/{timer_id}", status_code=204)
    async def cancel_timer(
        timer_id: UUID, request: Request, principal: Principal = Depends(authenticate)
    ):
        require_role(principal, "owner")
        store = store_for(request)
        record = store.timers.get(timer_id)
        if record is not None and record.status in ("scheduled", "firing"):
            store.timers[timer_id] = record.model_copy(update={"status": "cancelled"})
        return Response(status_code=204)

    @app.get("/v1/memories", response_model=MemoryListResponse)
    async def list_memories(request: Request, principal: Principal = Depends(authenticate)):
        require_role(principal, "owner")
        return MemoryListResponse(memories=tuple(store_for(request).memories.values()))

    @app.post("/v1/memories", response_model=MemoryRecord, status_code=201)
    async def write_memory(
        body: MemoryWriteRequest,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        principal: Principal = Depends(authenticate),
    ):
        require_role(principal, "owner")
        store = store_for(request)
        key = "memory-write:" + idempotency_header(idempotency_key)
        digest = _canonical_digest(body)
        prior = store.idempotency.get(key)
        if prior is not None:
            if not hmac.compare_digest(digest, prior[0]):
                raise MockError(409, "IDEMPOTENCY_CONFLICT", "Idempotency key was used for another request.")
            return prior[1]
        now = _now()
        record = MemoryRecord(
            memory_id=uuid.uuid4(),
            category=body.category,
            content=body.content,
            source_turn_id=body.source_turn_id,
            created_at=now,
            updated_at=now,
        )
        store.memories[record.memory_id] = record
        store.idempotency[key] = (digest, record)
        return record

    @app.patch("/v1/memories/{memory_id}", response_model=MemoryRecord)
    async def update_memory(
        memory_id: UUID,
        body: MemoryUpdateRequest,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        principal: Principal = Depends(authenticate),
    ):
        require_role(principal, "owner")
        store = store_for(request)
        key = "memory-update:" + idempotency_header(idempotency_key)
        digest = _canonical_digest({"memory_id": memory_id, "body": body})
        prior = store.idempotency.get(key)
        if prior is not None:
            if not hmac.compare_digest(digest, prior[0]):
                raise MockError(409, "IDEMPOTENCY_CONFLICT", "Idempotency key was used for another request.")
            return prior[1]
        old = store.memories.get(memory_id)
        if old is None:
            raise MockError(404, "RESOURCE_NOT_FOUND", "Memory not found.")
        updated = old.model_copy(update={"content": body.content, "source_turn_id": body.source_turn_id, "updated_at": _now()})
        store.memories[memory_id] = updated
        store.idempotency[key] = (digest, updated)
        return updated

    @app.delete("/v1/memories/{memory_id}", status_code=204)
    async def delete_memory(memory_id: UUID, request: Request, principal: Principal = Depends(authenticate)):
        require_role(principal, "owner")
        store = store_for(request)
        # Deletion is intentionally idempotent, including repeated requests.
        key = "memory-delete:" + idempotency_header(request.headers.get("Idempotency-Key"))
        digest = _canonical_digest({"memory_id": memory_id})
        prior = store.idempotency.get(key)
        if prior is not None:
            if not hmac.compare_digest(digest, prior[0]):
                raise MockError(409, "IDEMPOTENCY_CONFLICT", "Idempotency key was used for another request.")
            return Response(status_code=204)
        store.memories.pop(memory_id, None)
        store.idempotency[key] = (digest, True)
        return Response(status_code=204)

    @app.get("/v1/usage", response_model=UsageResponse)
    async def usage(principal: Principal = Depends(authenticate)):
        require_role(principal, "owner")
        fixture = _fixture("usage_response.json", UsageResponse)
        return fixture.model_copy(update={"as_of": _now()})

    @app.get("/v1/artifacts/{artifact_id}")
    async def artifact(artifact_id: UUID, principal: Principal = Depends(authenticate)):
        require_role(principal, "owner")
        raise MockError(404, "RESOURCE_NOT_FOUND", "Artifact not found in mock gateway.")

    @app.websocket("/v1/devices/{device_id}/connect")
    async def device_socket(websocket: WebSocket, device_id: str):
        settings: Settings = websocket.app.state.settings
        authorization = websocket.headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        valid = scheme.lower() == "bearer" and bool(token) and hmac.compare_digest(token, settings.device_token)
        if not valid or device_id != settings.device_id:
            await websocket.close(code=1008, reason="authentication or device scope rejected")
            return
        await websocket.accept()
        pending_audio_bytes: int | None = None
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                raw = message.get("bytes")
                if pending_audio_bytes is not None:
                    if raw is None or len(raw) != pending_audio_bytes:
                        await websocket.close(code=1008, reason="binary audio length does not match metadata")
                        return
                    pending_audio_bytes = None
                    await websocket.send_json({"type": "audio_frame_accepted"})
                    continue
                if raw is not None:
                    await websocket.close(code=1003, reason="audio bytes require metadata first")
                    return
                try:
                    envelope = json.loads(message.get("text") or "")
                    kind = envelope["type"]
                    payload = envelope["payload"]
                    if kind == "device_health":
                        event = DeviceHealthEvent.model_validate(payload)
                        _check_device_id(event.device_id, settings)
                    elif kind == "playback_meter":
                        event = PlaybackMeterEvent.model_validate(payload)
                        _check_device_id(event.device_id, settings)
                    elif kind == "tool_result":
                        event = ToolResult.model_validate(payload)
                        _check_device_id(event.device_id, settings)
                    elif kind == "audio_frame_metadata":
                        event = AudioFrameMetadata.model_validate(payload)
                        if event.sample_rate_hz > 48_000 or event.channels > 2 or event.frame_duration_ms > 100:
                            raise ValueError("audio frame exceeds mock limits")
                        byte_numerator = event.sample_rate_hz * event.channels * 2 * event.frame_duration_ms
                        if byte_numerator % 1000:
                            raise ValueError("audio frame duration does not align to PCM samples")
                        pending_audio_bytes = byte_numerator // 1000
                    else:
                        raise ValueError("unknown device message type")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
                    await websocket.close(code=1008, reason="invalid device message")
                    return
                await websocket.send_json({"type": "accepted", "message_type": kind})
        except (asyncio.CancelledError, RuntimeError):
            return

    return app


def _check_device_id(device_id: str, settings: Settings) -> None:
    if device_id != settings.device_id:
        raise ValueError("device id does not match authenticated device")


app = create_app()


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, access_log=False, log_level="info")


if __name__ == "__main__":
    main()
