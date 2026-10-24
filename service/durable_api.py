"""FastAPI routes backed by durable repositories and injected authentication."""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from voice.observability import initialize as initialize_observability, shutdown as shutdown_observability
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Sequence
from urllib.parse import quote
from uuid import UUID

from fastapi import Depends, FastAPI, Header, Query, Request, Response, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from contracts.models import (
    ApiErrorResponse,
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    ApprovalListResponse,
    HealthResponse,
    JobAccepted,
    JobCorrectionRequest,
    JobEvent,
    JobListResponse,
    JobSnapshot,
    LocalToolRequest,
    MemoryListResponse,
    MemoryRecord,
    MemoryUpdateRequest,
    MemoryWriteRequest,
    SessionCreateRequest,
    SessionListResponse,
    SessionResponse,
    TaskRequest,
    TimerCreateRequest,
    TimerListResponse,
    TimerRecord,
    UsageResponse,
    ToolResult,
)
from service.artifact_cleanup import ArtifactCleanupWorker
from service.artifacts import ArtifactError, ArtifactNotFound, ArtifactStore
from service.auth import AuthenticatedPrincipal
from service.device_events import DeviceEventHub, DeviceEventUnavailable
from service.live_gateway import (
    VoiceAuthenticator,
    VoiceProviderFactory,
    serve_voice_session,
)
from service.postgres import (
    IdempotencyConflict,
    LeaseLost,
    PostgresStore,
    StaleContextRevision,
    StorageConflict,
)
from service.timers import TimerDeliveryWorker


class DurableApiFailure(Exception):
    def __init__(self, status: int, code: str, message: str, retryable: bool = False):
        self.status = status
        self.code = code
        self.message = message
        self.retryable = retryable


AuthDependency = Callable[..., Awaitable[AuthenticatedPrincipal]]
DASHBOARD_ROOT = Path(__file__).resolve().parent.parent / "dashboard"


def create_durable_app(
    *,
    store: PostgresStore,
    authenticate: AuthDependency,
    authenticate_websocket: VoiceAuthenticator | None = None,
    live_session_factory: VoiceProviderFactory | None = None,
    artifact_store: ArtifactStore | None = None,
    timer_worker: TimerDeliveryWorker | None = None,
    device_event_hub: DeviceEventHub | None = None,
    idle_close_seconds: int = 30,
    max_session_seconds: int | None = None,
    manage_store: bool = False,
    apply_migrations_on_startup: bool = False,
    background_services: Sequence[Callable[[asyncio.Event], Awaitable[None]]] = (),
) -> FastAPI:
    """Build repository-backed API; caller supplies auth and opens the store."""

    if idle_close_seconds < 0:
        raise ValueError("idle close duration must not be negative")
    if max_session_seconds is not None and max_session_seconds <= 0:
        raise ValueError("maximum session duration must be positive")
    if apply_migrations_on_startup and not manage_store:
        raise ValueError("startup migrations require managed store lifecycle")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        opened_store = False
        if manage_store:
            await store.open()
            opened_store = True
        try:
            if apply_migrations_on_startup:
                await store.apply_migrations()
        except BaseException:
            if opened_store:
                await store.close()

            await shutdown_observability()
            raise

        stop = asyncio.Event()
        background_tasks = []
        if timer_worker is not None:
            background_tasks.append(
                asyncio.create_task(timer_worker.run(stop), name="cube-timer-delivery")
            )
        if artifact_store is not None:
            cleanup_worker = ArtifactCleanupWorker(store=store, artifact_store=artifact_store)
            background_tasks.append(
                asyncio.create_task(cleanup_worker.run(stop), name="cube-artifact-cleanup")
            )
        for index, service in enumerate(background_services):
            background_tasks.append(
                asyncio.create_task(service(stop), name=f"cube-background-service-{index}")
            )
        try:
            await asyncio.to_thread(initialize_observability, "gateway")
            yield
        finally:
            stop.set()
            for task in background_tasks:
                task.cancel()
            await asyncio.gather(*background_tasks, return_exceptions=True)
            if opened_store:
                await store.close()
            await shutdown_observability()

    app = FastAPI(title="Cube Assistant Durable Gateway", version="0.1.0", lifespan=lifespan)
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
    async def trace_request(request: Request, call_next):
        request.state.trace_id = uuid.uuid4()
        return await call_next(request)

    def error_response(request: Request, status: int, code: str, message: str, retryable=False, details=None):
        envelope = ApiErrorResponse(
            trace_id=getattr(request.state, "trace_id", uuid.uuid4()),
            error={"code": code, "message": message, "retryable": retryable, "details": details},
        )
        return JSONResponse(status_code=status, content=envelope.model_dump(mode="json"))

    @app.exception_handler(DurableApiFailure)
    async def durable_api_error(request: Request, error: DurableApiFailure):
        return error_response(request, error.status, error.code, error.message, error.retryable)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        fields = [
            ".".join(str(part) for part in item.get("loc", ()) if part != "body")
            for item in error.errors()
        ]
        return error_response(
            request,
            400,
            "INVALID_REQUEST",
            "Request does not match the v1 contract.",
            details={"fields": fields},
        )

    @app.exception_handler(IdempotencyConflict)
    async def idempotency_conflict(request: Request, _error: IdempotencyConflict):
        return error_response(request, 409, "IDEMPOTENCY_CONFLICT", "Idempotency key belongs to another request.")

    @app.exception_handler(StaleContextRevision)
    async def stale_revision(request: Request, _error: StaleContextRevision):
        return error_response(request, 409, "STALE_CONTEXT_REVISION", "Correction must use the next context revision.")

    @app.exception_handler(StorageConflict)
    async def storage_conflict(request: Request, _error: StorageConflict):
        return error_response(request, 409, "RESOURCE_CONFLICT", "Resource state conflicts with this request.")

    @app.exception_handler(LeaseLost)
    async def lease_lost(request: Request, _error: LeaseLost):
        return error_response(request, 409, "RESOURCE_CONFLICT", "Job lease is no longer active.")

    @app.exception_handler(Exception)
    async def internal_error(request: Request, _error: Exception):
        return error_response(request, 500, "INTERNAL_ERROR", "Gateway could not process request.")

    def require(principal: AuthenticatedPrincipal, *roles: str) -> UUID:
        require_role(principal, *roles)
        if principal.owner_id is None:
            raise DurableApiFailure(403, "PERMISSION_DENIED", "Credential has no owner mapping.")
        return principal.owner_id

    def require_role(principal: AuthenticatedPrincipal, *roles: str) -> None:
        if principal.role not in roles:
            raise DurableApiFailure(403, "PERMISSION_DENIED", "Credential lacks required scope.")

    def require_job(snapshot: JobSnapshot | None) -> JobSnapshot:
        if snapshot is None:
            raise DurableApiFailure(404, "RESOURCE_NOT_FOUND", "Job not found.")
        return snapshot

    async def require_job_access(
        principal: AuthenticatedPrincipal, job_id: UUID
    ) -> UUID:
        owner_id = require(principal, "owner", "device")
        if principal.role == "device":
            if not principal.device_id or not principal.device_id.strip():
                raise DurableApiFailure(
                    403, "PERMISSION_DENIED", "Device credential has no device mapping."
                )
            job_device_id = await store.get_job_device_id(
                owner_id=owner_id, job_id=job_id
            )
            if job_device_id != principal.device_id:
                raise DurableApiFailure(404, "RESOURCE_NOT_FOUND", "Job not found.")
        return owner_id

    def parse_event_cursor(last_event_id: str | None) -> tuple[UUID, int] | None:
        if not last_event_id:
            return None
        try:
            raw_job_id, raw_sequence = last_event_id.rsplit(":", 1)
            marker = UUID(raw_job_id), int(raw_sequence)
        except (ValueError, IndexError):
            raise DurableApiFailure(400, "INVALID_REQUEST", "Last-Event-ID must use job-id:sequence format.") from None
        if marker[1] < 1:
            raise DurableApiFailure(400, "INVALID_REQUEST", "Last-Event-ID is outside this job stream.")
        return marker

    def event_stream(events: tuple[JobEvent, ...], job_id: UUID, last_event_id: str | None):
        start_sequence = 0
        marker = parse_event_cursor(last_event_id)
        if marker is not None:
            marker_job, marker_sequence = marker
            if marker_job != job_id or marker_sequence < 1:
                raise DurableApiFailure(400, "INVALID_REQUEST", "Last-Event-ID is outside this job stream.")
            if not any(item.sequence == marker_sequence for item in events):
                raise DurableApiFailure(400, "INVALID_REQUEST", "Last-Event-ID is outside this job stream.")
            start_sequence = marker_sequence

        async def stream():
            for item in events:
                if item.sequence <= start_sequence:
                    continue
                data = json.dumps(item.model_dump(mode="json"), separators=(",", ":"))
                yield f"id: {job_id}:{item.sequence}\nevent: {item.kind}\ndata: {data}\n\n"
                await asyncio.sleep(0)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    def owner_event_stream(events: tuple[JobEvent, ...]):
        async def stream():
            for item in events:
                data = json.dumps(item.model_dump(mode="json"), separators=(",", ":"))
                yield f"id: {item.job_id}:{item.sequence}\nevent: {item.kind}\ndata: {data}\n\n"
                await asyncio.sleep(0)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/v1/sessions", response_model=SessionResponse, status_code=201)
    async def create_session(body: SessionCreateRequest, principal: AuthenticatedPrincipal = Depends(authenticate)):
        owner_id = require(principal, "device")
        if not principal.device_id or not principal.device_id.strip():
            raise DurableApiFailure(403, "PERMISSION_DENIED", "Device credential has no device mapping.")
        response = SessionResponse(
            session_id=uuid.uuid4(),
            conversation_id=body.conversation_id,
            state="connecting",
            opened_at=datetime.now(timezone.utc),
            idle_close_seconds=idle_close_seconds,
            max_session_seconds=max_session_seconds,
        )
        await store.create_session(owner_id=owner_id, device_id=principal.device_id, request=body, response=response)
        return response

    @app.get("/v1/sessions", response_model=SessionListResponse)
    async def list_sessions(
        limit: int = Query(default=100, ge=1, le=500),
        principal: AuthenticatedPrincipal = Depends(authenticate),
    ):
        owner_id = require(principal, "owner")
        return SessionListResponse(
            sessions=await store.list_sessions(owner_id=owner_id, limit=limit)
        )

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    async def close_session(session_id: UUID, principal: AuthenticatedPrincipal = Depends(authenticate)):
        owner_id = require(principal, "device")
        if not principal.device_id or not principal.device_id.strip():
            raise DurableApiFailure(403, "PERMISSION_DENIED", "Device credential has no device mapping.")
        await store.close_session(
            owner_id=owner_id, session_id=session_id, device_id=principal.device_id
        )
        return Response(status_code=204)

    @app.post("/internal/v1/local-tools", response_model=ToolResult)
    async def request_local_tool(
        body: LocalToolRequest,
        principal: AuthenticatedPrincipal = Depends(authenticate),
    ):
        owner_id = require(principal, "worker")
        if (
            device_event_hub is None
            or not principal.device_id
            or body.device_id != principal.device_id
        ):
            raise DurableApiFailure(403, "PERMISSION_DENIED", "Worker device scope rejected.")
        if body.job_id is None:
            raise DurableApiFailure(400, "INVALID_REQUEST", "Local tool request requires a job ID.")
        now = datetime.now(timezone.utc)
        if (
            body.issued_at > now + timedelta(seconds=5)
            or body.expires_at <= now
            or body.expires_at > now + timedelta(seconds=60)
        ):
            raise DurableApiFailure(409, "RESOURCE_CONFLICT", "Local tool request is expired or out of bounds.")
        job_device = await store.get_job_device_id(owner_id=owner_id, job_id=body.job_id)
        snapshot = await store.get_job(owner_id=owner_id, job_id=body.job_id)
        job_request = await store.get_job_request(owner_id=owner_id, job_id=body.job_id)
        if (
            job_device != body.device_id
            or snapshot is None
            or job_request is None
            or snapshot.status != "running"
        ):
            raise DurableApiFailure(404, "RESOURCE_NOT_FOUND", "Active job not found.")
        if (
            snapshot.context_revision != body.context_revision
            or job_request.context_revision != body.context_revision
            or job_request.device_id != body.device_id
            or job_request.conversation_id != body.conversation_id
            or job_request.session_id != body.session_id
            or job_request.delegation_id != body.delegation_id
            or job_request.turn_id != body.turn_id
            or job_request.trace_id != body.trace_id
        ):
            raise DurableApiFailure(409, "STALE_CONTEXT_REVISION", "Local tool request uses a stale job revision.")
        try:
            return await device_event_hub.deliver_local_tool(body)
        except DeviceEventUnavailable:
            raise DurableApiFailure(
                503,
                "DEVICE_OFFLINE",
                "The device could not complete the local action.",
                retryable=True,
            ) from None

    @app.websocket("/v1/devices/{device_id}/connect")
    async def device_voice_socket(
        websocket: WebSocket,
        device_id: str,
        session_id: UUID = Query(...),
    ):
        if authenticate_websocket is None or live_session_factory is None:
            await websocket.close(code=1013, reason="voice gateway is not configured")
            return
        await serve_voice_session(
            websocket,
            device_id=device_id,
            session_id=session_id,
            store=store,
            authenticate=authenticate_websocket,
            provider_factory=live_session_factory,
        )

    @app.websocket("/v1/devices/{device_id}/events")
    async def device_event_socket(websocket: WebSocket, device_id: str):
        if authenticate_websocket is None or device_event_hub is None:
            await websocket.close(code=1013, reason="device events are not configured")
            return
        try:
            principal = await authenticate_websocket(websocket)
        except Exception:
            await websocket.close(code=1008, reason="authentication rejected")
            return
        if (
            getattr(principal, "role", None) != "device"
            or getattr(principal, "device_id", None) != device_id
            or getattr(principal, "owner_id", None) is None
        ):
            await websocket.close(code=1008, reason="device scope rejected")
            return
        await device_event_hub.serve(websocket, device_id=device_id)

    @app.post("/v1/jobs", response_model=JobAccepted, status_code=202)
    async def submit_job(body: TaskRequest, principal: AuthenticatedPrincipal = Depends(authenticate)):
        owner_id = require(principal, "device", "owner")
        if principal.role == "device" and body.device_id != principal.device_id:
            raise DurableApiFailure(403, "PERMISSION_DENIED", "Task device ID does not match credential.")
        return await store.submit_job(owner_id=owner_id, request=body)

    @app.get("/v1/jobs", response_model=JobListResponse)
    async def list_jobs(
        limit: int = Query(default=100, ge=1, le=500),
        principal: AuthenticatedPrincipal = Depends(authenticate),
    ):
        owner_id = require(principal, "owner")
        return JobListResponse(jobs=await store.list_jobs(owner_id=owner_id, limit=limit))

    @app.get("/v1/jobs/{job_id}", response_model=JobSnapshot)
    async def read_job(job_id: UUID, principal: AuthenticatedPrincipal = Depends(authenticate)):
        owner_id = await require_job_access(principal, job_id)
        return require_job(await store.get_job(owner_id=owner_id, job_id=job_id))

    @app.get("/v1/jobs/{job_id}/events")
    async def read_job_events(
        job_id: UUID,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        principal: AuthenticatedPrincipal = Depends(authenticate),
    ):
        owner_id = await require_job_access(principal, job_id)
        snapshot = await store.get_job(owner_id=owner_id, job_id=job_id)
        if snapshot is None:
            raise DurableApiFailure(404, "RESOURCE_NOT_FOUND", "Job not found.")
        events = await store.list_job_events(owner_id=owner_id, job_id=job_id)
        return event_stream(events, job_id, last_event_id)

    @app.get("/v1/events")
    async def read_owner_events(
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        principal: AuthenticatedPrincipal = Depends(authenticate),
    ):
        owner_id = require(principal, "owner")
        marker = parse_event_cursor(last_event_id)
        events = await store.list_owner_job_events(
            owner_id=owner_id,
            after_job_id=marker[0] if marker else None,
            after_sequence=marker[1] if marker else None,
        )
        if events is None:
            raise DurableApiFailure(400, "INVALID_REQUEST", "Last-Event-ID is outside this owner event stream.")
        return owner_event_stream(events)

    @app.post("/v1/jobs/{job_id}/corrections", response_model=JobSnapshot, status_code=202)
    async def correct_job(
        job_id: UUID,
        body: JobCorrectionRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        principal: AuthenticatedPrincipal = Depends(authenticate),
    ):
        owner_id = await require_job_access(principal, job_id)
        snapshot = await store.correct_job(
            owner_id=owner_id,
            job_id=job_id,
            request=body,
            idempotency_key=idempotency_key,
        )
        return require_job(snapshot)

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobSnapshot, status_code=202)
    async def cancel_job(
        job_id: UUID,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        principal: AuthenticatedPrincipal = Depends(authenticate),
    ):
        owner_id = await require_job_access(principal, job_id)
        snapshot = await store.cancel_job(owner_id=owner_id, job_id=job_id, idempotency_key=idempotency_key)
        return require_job(snapshot)

    @app.get("/v1/approvals", response_model=ApprovalListResponse)
    async def list_approvals(principal: AuthenticatedPrincipal = Depends(authenticate)):
        owner_id = require(principal, "owner")
        return ApprovalListResponse(approvals=await store.list_approvals(owner_id=owner_id))

    @app.post("/v1/approvals/{approval_id}/decision", response_model=ApprovalDecisionResponse)
    async def decide_approval(
        approval_id: UUID,
        body: ApprovalDecisionRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        principal: AuthenticatedPrincipal = Depends(authenticate),
    ):
        owner_id = require(principal, "owner")
        result = await store.decide_approval(
            owner_id=owner_id,
            approval_id=approval_id,
            request=body,
            idempotency_key=idempotency_key,
        )
        if result is None:
            raise DurableApiFailure(404, "RESOURCE_NOT_FOUND", "Approval not found.")
        return result

    @app.post("/v1/timers", response_model=TimerRecord, status_code=201)
    async def create_timer(body: TimerCreateRequest, principal: AuthenticatedPrincipal = Depends(authenticate)):
        owner_id = require(principal, "device", "owner")
        if not principal.device_id:
            raise DurableApiFailure(403, "PERMISSION_DENIED", "Credential has no target device mapping.")
        return await store.create_timer(owner_id=owner_id, device_id=principal.device_id, request=body)

    @app.get("/v1/timers", response_model=TimerListResponse)
    async def list_timers(principal: AuthenticatedPrincipal = Depends(authenticate)):
        owner_id = require(principal, "owner")
        return TimerListResponse(timers=await store.list_timers(owner_id=owner_id))

    @app.delete("/v1/timers/{timer_id}", status_code=204)
    async def cancel_timer(timer_id: UUID, principal: AuthenticatedPrincipal = Depends(authenticate)):
        owner_id = require(principal, "owner")
        await store.cancel_timer(owner_id=owner_id, timer_id=timer_id)
        return Response(status_code=204)

    @app.get("/v1/memories", response_model=MemoryListResponse)
    async def list_memories(principal: AuthenticatedPrincipal = Depends(authenticate)):
        owner_id = require(principal, "owner")
        return MemoryListResponse(memories=await store.list_memories(owner_id=owner_id))

    @app.post("/v1/memories", response_model=MemoryRecord, status_code=201)
    async def write_memory(
        body: MemoryWriteRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        principal: AuthenticatedPrincipal = Depends(authenticate),
    ):
        owner_id = require(principal, "owner")
        memory, _deduplicated = await store.write_memory(
            owner_id=owner_id, request=body, idempotency_key=idempotency_key
        )
        return memory

    @app.patch("/v1/memories/{memory_id}", response_model=MemoryRecord)
    async def update_memory(
        memory_id: UUID,
        body: MemoryUpdateRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        principal: AuthenticatedPrincipal = Depends(authenticate),
    ):
        owner_id = require(principal, "owner")
        memory, _deduplicated = await store.update_memory_idempotent(
            owner_id=owner_id,
            memory_id=memory_id,
            request=body,
            idempotency_key=idempotency_key,
        )
        if memory is None:
            raise DurableApiFailure(404, "RESOURCE_NOT_FOUND", "Memory not found.")
        return memory

    @app.delete("/v1/memories/{memory_id}", status_code=204)
    async def delete_memory(
        memory_id: UUID,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        principal: AuthenticatedPrincipal = Depends(authenticate),
    ):
        owner_id = require(principal, "owner")
        await store.delete_memory_idempotent(
            owner_id=owner_id, memory_id=memory_id, idempotency_key=idempotency_key
        )
        return Response(status_code=204)

    @app.get("/v1/usage", response_model=UsageResponse)
    async def usage(
        session_id: UUID | None = None,
        principal: AuthenticatedPrincipal = Depends(authenticate),
    ):
        owner_id = require(principal, "owner")
        now = datetime.now(timezone.utc)
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return UsageResponse(
            measured_usage=await store.list_usage(owner_id=owner_id, session_id=session_id),
            summary=await store.summarize_usage(
                owner_id=owner_id, period_start=period_start, period_end=now
            ),
            as_of=now,
        )

    @app.get("/v1/artifacts/{artifact_id}")
    async def download_artifact(artifact_id: UUID, principal: AuthenticatedPrincipal = Depends(authenticate)):
        owner_id = require(principal, "owner")
        if artifact_store is None:
            raise DurableApiFailure(503, "DEPENDENCY_UNAVAILABLE", "Artifact storage is unavailable.", True)
        metadata = await store.get_artifact(owner_id=owner_id, artifact_id=artifact_id)
        if metadata is None:
            raise DurableApiFailure(404, "RESOURCE_NOT_FOUND", "Artifact not found.")
        try:
            content = await artifact_store.get(owner_id, artifact_id)
        except ArtifactNotFound:
            raise DurableApiFailure(404, "RESOURCE_NOT_FOUND", "Artifact not found.") from None
        except ArtifactError:
            raise DurableApiFailure(503, "DEPENDENCY_UNAVAILABLE", "Artifact storage is unavailable.", True) from None
        filename = quote(metadata.artifact_ref.label.replace("\r", "").replace("\n", ""), safe="")
        return Response(
            content=content,
            media_type=metadata.artifact_ref.media_type,
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
        )

    @app.delete("/v1/artifacts/{artifact_id}", status_code=204)
    async def delete_artifact(artifact_id: UUID, principal: AuthenticatedPrincipal = Depends(authenticate)):
        owner_id = require(principal, "owner")
        if artifact_store is None:
            raise DurableApiFailure(503, "DEPENDENCY_UNAVAILABLE", "Artifact storage is unavailable.", True)
        await store.delete_artifact_metadata(owner_id=owner_id, artifact_id=artifact_id)
        return Response(status_code=204)

    @app.get("/v1/health", response_model=HealthResponse)
    async def health(principal: AuthenticatedPrincipal = Depends(authenticate)):
        require_role(principal, "operator")
        try:
            healthy = await store.healthcheck()
        except Exception:
            healthy = False
        return HealthResponse(
            status="healthy" if healthy else "unhealthy",
            service="cube-assistant-gateway",
            observed_at=datetime.now(timezone.utc),
            dependencies={"postgres": "healthy" if healthy else "unavailable"},
        )

    return app
