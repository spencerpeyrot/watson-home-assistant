"""PostgreSQL connection, migration, and owner-scoped persistence helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from psycopg import errors
from psycopg_pool import AsyncConnectionPool
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from contracts.models import (
    ArtifactRef,
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    ApprovalRecord,
    JobAccepted,
    JobCorrectionRequest,
    JobEvent,
    JobSnapshot,
    MemoryRecord,
    MemoryUpdateRequest,
    MemoryWriteRequest,
    SessionCreateRequest,
    SessionResponse,
    SourceRef,
    TaskRequest,
    TimerCreateRequest,
    TimerFiredEvent,
    TimerRecord,
    UsageMeasurement,
    UsageRecord,
    UsageSummary,
)
from service.tool_policy import ActionClaim


class StorageConflict(RuntimeError):
    """A record conflicts with an existing immutable identity."""


class IdempotencyConflict(StorageConflict):
    """An idempotency key was reused with a different action payload."""


class StaleContextRevision(StorageConflict):
    """A job correction does not match the next committed revision."""


class LeaseLost(StorageConflict):
    """A worker no longer owns the current job lease."""


class TimerLeaseLost(StorageConflict):
    """A worker no longer owns the current timer-delivery lease."""


@dataclass(frozen=True)
class ClaimedJob:
    job_id: UUID
    owner_id: UUID
    worker_id: UUID
    request: TaskRequest
    snapshot: JobSnapshot
    lease_expires_at: datetime
    attempt_count: int


@dataclass(frozen=True)
class ClaimedTimer:
    owner_id: UUID
    worker_id: UUID
    event: TimerFiredEvent
    lease_expires_at: datetime


@dataclass(frozen=True)
class StoredArtifact:
    artifact_ref: ArtifactRef
    job_id: UUID
    storage_key: str
    byte_size: int
    sha256: str
    created_at: datetime


def _json_safe(value):
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _digest_json(value) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PostgresStore:
    """Small async store for application records; owner IDs come from auth."""

    def __init__(
        self,
        database_url: str,
        *,
        min_size: int = 1,
        max_size: int = 10,
        migration_dir: Path | None = None,
    ) -> None:
        if not database_url.strip():
            raise ValueError("database URL must not be blank")
        if min_size < 1 or max_size < min_size:
            raise ValueError("pool size limits are invalid")
        self._pool = AsyncConnectionPool(
            conninfo=database_url,
            min_size=min_size,
            max_size=max_size,
            open=False,
        )
        self._migration_dir = migration_dir or Path(__file__).parent / "migrations"

    async def open(self) -> None:
        await self._pool.open(wait=True)

    async def close(self) -> None:
        await self._pool.close()

    async def apply_migrations(self) -> tuple[str, ...]:
        """Apply unapplied SQL files in one transaction each, in filename order."""

        applied: list[str] = []
        migrations = sorted(self._migration_dir.glob("[0-9]*_*.sql"))
        async with self._pool.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations ("
                    "version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
                )
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext('cube-assistant-schema-migrations'))"
                )
                cursor = await connection.execute("SELECT version FROM schema_migrations")
                installed = {row[0] for row in await cursor.fetchall()}
                for migration in migrations:
                    version = migration.stem
                    if version in installed:
                        continue
                    script = migration.read_text(encoding="utf-8")
                    await connection.execute(script, prepare=False)
                    cursor = await connection.execute(
                        "SELECT 1 FROM schema_migrations WHERE version = %s", (version,)
                    )
                    if await cursor.fetchone() is None:
                        raise RuntimeError(f"migration {version} did not record completion")
                    applied.append(version)
        return tuple(applied)

    async def healthcheck(self) -> bool:
        async with self._pool.connection() as connection:
            cursor = await connection.execute("SELECT 1")
            return await cursor.fetchone() == (1,)

    async def _reserve_idempotency_tx(
        self,
        connection,
        *,
        owner_id: UUID,
        operation: str,
        idempotency_key: str,
        request_digest: str,
    ):
        """Reserve an operation key inside its write transaction; return replay JSON."""

        if not operation.strip() or not idempotency_key.strip():
            raise ValueError("idempotency operation and key must not be blank")
        await connection.execute(
            "DELETE FROM idempotency_records WHERE owner_id = %s AND operation = %s "
            "AND idempotency_key = %s AND expires_at <= now()",
            (owner_id, operation, idempotency_key),
        )
        await connection.execute(
            "INSERT INTO idempotency_records "
            "(owner_id, operation, idempotency_key, request_digest, expires_at) "
            "VALUES (%s, %s, %s, %s, now() + interval '30 days') "
            "ON CONFLICT (owner_id, operation, idempotency_key) DO NOTHING",
            (owner_id, operation, idempotency_key, request_digest),
        )
        cursor = await connection.execute(
            "SELECT request_digest, response FROM idempotency_records "
            "WHERE owner_id = %s AND operation = %s AND idempotency_key = %s FOR UPDATE",
            (owner_id, operation, idempotency_key),
        )
        row = await cursor.fetchone()
        if row is None:
            raise StorageConflict("idempotency reservation disappeared")
        if row[0] != request_digest:
            raise IdempotencyConflict("idempotency key already belongs to another request")
        return row[1]

    async def _store_idempotency_response_tx(
        self,
        connection,
        *,
        owner_id: UUID,
        operation: str,
        idempotency_key: str,
        response,
    ) -> None:
        await connection.execute(
            "UPDATE idempotency_records SET response = %s, expires_at = now() + interval '30 days' "
            "WHERE owner_id = %s AND operation = %s AND idempotency_key = %s",
            (Jsonb(_json_safe(response)), owner_id, operation, idempotency_key),
        )

    async def submit_job(
        self,
        *,
        owner_id: UUID,
        request: TaskRequest,
        job_id: UUID | None = None,
        parent_job_id: UUID | None = None,
    ) -> JobAccepted:
        """Persist one task and initial event; deduplicate exact submissions."""

        new_job_id = job_id or uuid4()
        request_data = request.model_dump(mode="json")
        request_digest = _digest_json(
            {"request": request_data, "parent_job_id": str(parent_job_id) if parent_job_id else None}
        )
        now = datetime.now(timezone.utc)
        summary = "Accepted and queued."
        snapshot = JobSnapshot(
            job_id=new_job_id,
            status="queued",
            context_revision=request.context_revision,
            last_sequence=1,
            summary=summary,
        )
        event = JobEvent(
            device_id=request.device_id,
            job_id=new_job_id,
            event_id=uuid4(),
            sequence=1,
            kind="status",
            status="queued",
            conversation_id=request.conversation_id,
            session_id=request.session_id,
            turn_id=request.turn_id,
            delegation_id=request.delegation_id,
            trace_id=request.trace_id,
            parent_job_id=parent_job_id,
            context_revision=request.context_revision,
            summary=summary,
            timestamp=now,
        )
        async with self._pool.connection() as connection:
            async with connection.transaction():
                try:
                    cursor = await connection.execute(
                        "INSERT INTO jobs "
                        "(job_id, owner_id, device_id, conversation_id, session_id, turn_id, "
                        "delegation_id, trace_id, parent_job_id, context_revision, status, "
                        "request_text, request, request_digest, idempotency_key, summary, "
                        "last_sequence, snapshot) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'queued', "
                        "%s, %s, %s, %s, %s, 1, %s) "
                        "ON CONFLICT (owner_id, idempotency_key) DO NOTHING "
                        "RETURNING job_id",
                        (
                            new_job_id,
                            owner_id,
                            request.device_id,
                            request.conversation_id,
                            request.session_id,
                            request.turn_id,
                            request.delegation_id,
                            request.trace_id,
                            parent_job_id,
                            request.context_revision,
                            request.request_text,
                            Jsonb(request_data),
                            request_digest,
                            request.idempotency_key,
                            summary,
                            Jsonb(snapshot.model_dump(mode="json")),
                        ),
                    )
                    inserted = await cursor.fetchone()
                except errors.UniqueViolation:
                    raise StorageConflict("job ID or delegated parent is invalid") from None
                if inserted is None:
                    cursor = await connection.execute(
                        "SELECT job_id, request_digest, status FROM jobs "
                        "WHERE owner_id = %s AND idempotency_key = %s",
                        (owner_id, request.idempotency_key),
                    )
                    existing = await cursor.fetchone()
                    if existing is None:
                        raise StorageConflict("job idempotency record disappeared")
                    existing_id, existing_digest, existing_status = existing
                    if existing_digest != request_digest:
                        raise IdempotencyConflict(
                            "idempotency key already belongs to a different task"
                        )
                    cursor = await connection.execute(
                        "SELECT MIN(sequence) FROM job_events "
                        "WHERE owner_id = %s AND job_id = %s",
                        (owner_id, existing_id),
                    )
                    first_sequence = (await cursor.fetchone())[0] or 1
                    accepted_status = "queued" if existing_status == "queued" else "running"
                    return JobAccepted(
                        job_id=existing_id,
                        status=accepted_status,
                        trace_id=request.trace_id,
                        deduplicated=True,
                        first_event_sequence=first_sequence,
                    )
                await connection.execute(
                    "INSERT INTO job_events "
                    "(event_id, job_id, owner_id, sequence, kind, status, context_revision, payload, created_at) "
                    "VALUES (%s, %s, %s, 1, 'status', 'queued', %s, %s, %s)",
                    (
                        event.event_id,
                        new_job_id,
                        owner_id,
                        request.context_revision,
                        Jsonb(event.model_dump(mode="json")),
                        now,
                    ),
                )
        return JobAccepted(
            job_id=new_job_id,
            status="queued",
            trace_id=request.trace_id,
            deduplicated=False,
            first_event_sequence=1,
        )

    async def claim_next_job(
        self, *, worker_id: UUID, lease_seconds: int = 30
    ) -> ClaimedJob | None:
        """Claim oldest queued or expired-running job without blocking peers."""

        if lease_seconds <= 0:
            raise ValueError("job lease must be positive")
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    "WITH candidate AS ("
                    " SELECT job_id, owner_id FROM jobs"
                    " WHERE status = 'queued' OR (status = 'running' AND lease_expires_at <= now())"
                    " ORDER BY created_at, job_id LIMIT 1 FOR UPDATE SKIP LOCKED"
                    ") UPDATE jobs AS job SET status = 'running', claimed_by = %s,"
                    " lease_expires_at = now() + (%s * interval '1 second'),"
                    " attempt_count = job.attempt_count + 1, updated_at = now()"
                    " FROM candidate WHERE job.job_id = candidate.job_id"
                    " AND job.owner_id = candidate.owner_id"
                    " RETURNING job.job_id, job.owner_id, job.request, job.lease_expires_at,"
                    " job.attempt_count",
                    (worker_id, lease_seconds),
                )
                row = await cursor.fetchone()
                if row is None:
                    return None
                job_id, owner_id, request_data, lease_expires_at, attempt_count = row
                snapshot = await self._append_job_event_tx(
                    connection,
                    owner_id=owner_id,
                    job_id=job_id,
                    kind="status",
                    status="running",
                    summary=f"Worker claimed job (attempt {attempt_count}).",
                    worker_id=worker_id,
                )
                return ClaimedJob(
                    job_id=job_id,
                    owner_id=owner_id,
                    worker_id=worker_id,
                    request=TaskRequest.model_validate(request_data),
                    snapshot=snapshot,
                    lease_expires_at=lease_expires_at,
                    attempt_count=attempt_count,
                )

    async def renew_job_lease(
        self, *, job_id: UUID, worker_id: UUID, lease_seconds: int = 30
    ) -> bool:
        if lease_seconds <= 0:
            raise ValueError("job lease must be positive")
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    "UPDATE jobs SET lease_expires_at = now() + (%s * interval '1 second'), "
                    "updated_at = now() WHERE job_id = %s AND claimed_by = %s "
                    "AND status = 'running' AND lease_expires_at > now() "
                    "RETURNING job_id",
                    (lease_seconds, job_id, worker_id),
                )
                return await cursor.fetchone() is not None

    async def append_job_event(
        self,
        *,
        owner_id: UUID,
        job_id: UUID,
        kind: str,
        status: str,
        summary: str,
        context_revision: int | None = None,
        worker_id: UUID | None = None,
        artifact_refs: tuple[ArtifactRef, ...] = (),
        sources: tuple[SourceRef, ...] = (),
        error_code: str | None = None,
        measured_usage: UsageMeasurement | None = None,
        usage_provider: str | None = None,
    ) -> JobEvent:
        """Append sequenced event and update snapshot in one transaction."""

        async with self._pool.connection() as connection:
            async with connection.transaction():
                _, event = await self._append_job_event_tx(
                    connection,
                    owner_id=owner_id,
                    job_id=job_id,
                    kind=kind,
                    status=status,
                    summary=summary,
                    context_revision=context_revision,
                    worker_id=worker_id,
                    artifact_refs=artifact_refs,
                    sources=sources,
                    error_code=error_code,
                    measured_usage=measured_usage,
                    usage_provider=usage_provider,
                    return_event=True,
                )
                return event

    async def _append_job_event_tx(
        self,
        connection,
        *,
        owner_id: UUID,
        job_id: UUID,
        kind: str,
        status: str,
        summary: str,
        context_revision: int | None = None,
        worker_id: UUID | None = None,
        artifact_refs: tuple[ArtifactRef, ...] = (),
        sources: tuple[SourceRef, ...] = (),
        error_code: str | None = None,
        measured_usage: UsageMeasurement | None = None,
        usage_provider: str | None = None,
        return_event: bool = False,
    ):
        cursor = await connection.execute(
            "SELECT device_id, conversation_id, session_id, turn_id, delegation_id, "
            "trace_id, parent_job_id, context_revision, last_sequence, snapshot, "
            "claimed_by, lease_expires_at, status FROM jobs "
            "WHERE owner_id = %s AND job_id = %s FOR UPDATE",
            (owner_id, job_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise StorageConflict("job not found for owner")
        (
            device_id,
            conversation_id,
            session_id,
            turn_id,
            delegation_id,
            trace_id,
            parent_job_id,
            current_revision,
            current_sequence,
            snapshot_data,
            claimed_by,
            lease_expires_at,
            current_status,
        ) = row
        now = datetime.now(timezone.utc)
        if worker_id is not None and (
            claimed_by != worker_id or lease_expires_at is None or lease_expires_at <= now
        ):
            raise LeaseLost("job lease expired or belongs to another worker")
        terminal = {"succeeded", "failed", "cancelled", "needs_reconciliation"}
        if current_status in terminal and status != current_status:
            raise StorageConflict("terminal job status cannot change")
        revision = current_revision if context_revision is None else context_revision
        if revision < current_revision:
            raise StorageConflict("stale job context revision")
        sequence = current_sequence + 1
        event = JobEvent(
            device_id=device_id,
            job_id=job_id,
            event_id=uuid4(),
            sequence=sequence,
            kind=kind,
            status=status,
            conversation_id=conversation_id,
            session_id=session_id,
            turn_id=turn_id,
            delegation_id=delegation_id,
            trace_id=trace_id,
            parent_job_id=parent_job_id,
            context_revision=revision,
            summary=summary,
            timestamp=now,
            artifact_refs=artifact_refs,
            sources=sources,
            error_code=error_code,
            measured_usage=measured_usage,
        )
        prior = JobSnapshot.model_validate(snapshot_data)
        merged_artifacts = {item.artifact_id: item for item in prior.artifact_refs}
        merged_artifacts.update({item.artifact_id: item for item in artifact_refs})
        merged_sources = {str(item.url): item for item in prior.sources}
        merged_sources.update({str(item.url): item for item in sources})
        snapshot = JobSnapshot(
            job_id=job_id,
            status=status,
            context_revision=revision,
            last_sequence=sequence,
            summary=summary,
            artifact_refs=tuple(merged_artifacts.values()),
            sources=tuple(merged_sources.values()),
            error_code=error_code or prior.error_code,
        )
        await connection.execute(
            "INSERT INTO job_events "
            "(event_id, job_id, owner_id, sequence, kind, status, context_revision, payload, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                event.event_id,
                job_id,
                owner_id,
                sequence,
                kind,
                status,
                revision,
                Jsonb(event.model_dump(mode="json")),
                now,
            ),
        )
        usage_event = (kind == "result" and status == "succeeded") or (
            kind == "status" and status in {"failed", "cancelled"}
        )
        if usage_event:
            if measured_usage is not None and usage_provider is None:
                raise ValueError("measured job usage requires a provider")
            if usage_provider is not None and not usage_provider.strip():
                raise ValueError("usage provider must not be blank")
        if usage_event and measured_usage is not None:
            await connection.execute(
                "INSERT INTO usage_measurements "
                "(usage_id, owner_id, session_id, job_id, provider, input_tokens, "
                "output_tokens, connected_seconds, cost_usd, cost_kind, observed_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (usage_id) DO NOTHING",
                (
                    event.event_id,
                    owner_id,
                    session_id,
                    job_id,
                    usage_provider.strip(),
                    measured_usage.input_tokens,
                    measured_usage.output_tokens,
                    measured_usage.connected_seconds,
                    measured_usage.cost_usd,
                    measured_usage.cost_kind,
                    now,
                ),
            )
        release = status in terminal or status in {"queued", "waiting_for_approval"}
        await connection.execute(
            "UPDATE jobs SET status = %s, context_revision = %s, last_sequence = %s, "
            "summary = %s, snapshot = %s, updated_at = now(), "
            "claimed_by = CASE WHEN %s THEN NULL ELSE claimed_by END, "
            "lease_expires_at = CASE WHEN %s THEN NULL ELSE lease_expires_at END "
            "WHERE owner_id = %s AND job_id = %s",
            (
                status,
                revision,
                sequence,
                summary,
                Jsonb(snapshot.model_dump(mode="json")),
                release,
                release,
                owner_id,
                job_id,
            ),
        )
        return (snapshot, event) if return_event else snapshot

    async def get_job(self, *, owner_id: UUID, job_id: UUID) -> JobSnapshot | None:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT snapshot FROM jobs WHERE owner_id = %s AND job_id = %s",
                (owner_id, job_id),
            )
            row = await cursor.fetchone()
        return JobSnapshot.model_validate(row[0]) if row is not None else None

    async def get_job_request(
        self, *, owner_id: UUID, job_id: UUID
    ) -> TaskRequest | None:
        """Return the current owner-scoped job request for internal correlation."""

        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT request FROM jobs WHERE owner_id = %s AND job_id = %s",
                (owner_id, job_id),
            )
            row = await cursor.fetchone()
        return TaskRequest.model_validate(row[0]) if row is not None else None

    async def get_job_device_id(self, *, owner_id: UUID, job_id: UUID) -> str | None:
        """Return owning device for device-scoped job access checks."""

        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT device_id FROM jobs WHERE owner_id = %s AND job_id = %s",
                (owner_id, job_id),
            )
            row = await cursor.fetchone()
        return row[0] if row is not None else None

    async def list_jobs(
        self, *, owner_id: UUID, limit: int = 100
    ) -> tuple[JobSnapshot, ...]:
        """List recent owner-scoped job snapshots for the dashboard."""

        if not 1 <= limit <= 500:
            raise ValueError("job list limit must be between 1 and 500")
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT snapshot FROM jobs WHERE owner_id = %s "
                "ORDER BY updated_at DESC, job_id LIMIT %s",
                (owner_id, limit),
            )
            rows = await cursor.fetchall()
        return tuple(JobSnapshot.model_validate(row[0]) for row in rows)

    async def correct_job(
        self,
        *,
        owner_id: UUID,
        job_id: UUID,
        request: JobCorrectionRequest,
        idempotency_key: str,
    ) -> JobSnapshot | None:
        """Requeue corrected request at next revision and invalidate old worker lease."""

        operation = "job-correction"
        digest = _digest_json({"job_id": job_id, "request": request})
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    "SELECT request, parent_job_id, status, context_revision "
                    "FROM jobs WHERE owner_id = %s AND job_id = %s FOR UPDATE",
                    (owner_id, job_id),
                )
                row = await cursor.fetchone()
                if row is None:
                    return None
                replay = await self._reserve_idempotency_tx(
                    connection,
                    owner_id=owner_id,
                    operation=operation,
                    idempotency_key=idempotency_key,
                    request_digest=digest,
                )
                if replay is not None:
                    return JobSnapshot.model_validate(replay)
                old_request, parent_job_id, status, current_revision = row
                if status not in {"queued", "running", "waiting_for_approval"}:
                    raise StorageConflict("terminal job cannot be corrected")
                if request.context_revision != current_revision + 1:
                    raise StaleContextRevision("correction must use the next context revision")
                corrected = TaskRequest.model_validate(old_request).model_copy(
                    update={
                        "request_text": request.request_text,
                        "context_revision": request.context_revision,
                        "recent_turns": request.recent_turns,
                    }
                )
                request_data = corrected.model_dump(mode="json")
                request_digest = _digest_json(
                    {
                        "request": request_data,
                        "parent_job_id": str(parent_job_id) if parent_job_id else None,
                    }
                )
                await connection.execute(
                    "UPDATE jobs SET request_text = %s, request = %s, request_digest = %s, "
                    "context_revision = %s, status = 'queued', claimed_by = NULL, "
                    "lease_expires_at = NULL, summary = 'Correction accepted and requeued.', "
                    "updated_at = now() WHERE owner_id = %s AND job_id = %s",
                    (
                        corrected.request_text,
                        Jsonb(request_data),
                        request_digest,
                        request.context_revision,
                        owner_id,
                        job_id,
                    ),
                )
                await connection.execute(
                    "UPDATE approvals AS approval SET status = 'expired' "
                    "FROM action_ledger AS action WHERE approval.owner_id = %s "
                    "AND approval.status = 'pending' AND action.owner_id = approval.owner_id "
                    "AND action.action_id = approval.action_id AND action.job_id = %s",
                    (owner_id, job_id),
                )
                snapshot = await self._append_job_event_tx(
                    connection,
                    owner_id=owner_id,
                    job_id=job_id,
                    kind="correction_acknowledged",
                    status="queued",
                    summary="Correction accepted and requeued.",
                    context_revision=request.context_revision,
                )
                await self._store_idempotency_response_tx(
                    connection,
                    owner_id=owner_id,
                    operation=operation,
                    idempotency_key=idempotency_key,
                    response=snapshot,
                )
        return snapshot

    async def cancel_job(
        self, *, owner_id: UUID, job_id: UUID, idempotency_key: str
    ) -> JobSnapshot | None:
        """Cancel owned queued/running job once and invalidate its worker lease."""

        operation = "job-cancel"
        digest = _digest_json({"job_id": job_id})
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    "SELECT status, snapshot FROM jobs WHERE owner_id = %s AND job_id = %s FOR UPDATE",
                    (owner_id, job_id),
                )
                row = await cursor.fetchone()
                if row is None:
                    return None
                replay = await self._reserve_idempotency_tx(
                    connection,
                    owner_id=owner_id,
                    operation=operation,
                    idempotency_key=idempotency_key,
                    request_digest=digest,
                )
                if replay is not None:
                    return JobSnapshot.model_validate(replay)
                status, snapshot_data = row
                if status == "cancelled":
                    snapshot = JobSnapshot.model_validate(snapshot_data)
                elif status not in {"queued", "running", "waiting_for_approval"}:
                    raise StorageConflict("terminal job cannot be cancelled")
                else:
                    snapshot = await self._append_job_event_tx(
                        connection,
                        owner_id=owner_id,
                        job_id=job_id,
                        kind="status",
                        status="cancelled",
                        summary="Job cancelled by owner.",
                    )
                await self._store_idempotency_response_tx(
                    connection,
                    owner_id=owner_id,
                    operation=operation,
                    idempotency_key=idempotency_key,
                    response=snapshot,
                )
        return snapshot

    async def list_job_events(
        self, *, owner_id: UUID, job_id: UUID
    ) -> tuple[JobEvent, ...]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT payload FROM job_events WHERE owner_id = %s AND job_id = %s "
                "ORDER BY sequence",
                (owner_id, job_id),
            )
            rows = await cursor.fetchall()
        return tuple(JobEvent.model_validate(row[0]) for row in rows)

    async def list_owner_job_events(
        self,
        *,
        owner_id: UUID,
        after_job_id: UUID | None = None,
        after_sequence: int | None = None,
        limit: int = 500,
    ) -> tuple[JobEvent, ...] | None:
        """List an owner's ordered event window; None means cursor is not owned."""

        if not 1 <= limit <= 500:
            raise ValueError("event limit must be between 1 and 500")
        if (after_job_id is None) != (after_sequence is None):
            raise ValueError("event cursor needs both job ID and sequence")
        async with self._pool.connection() as connection:
            marker = None
            if after_job_id is not None:
                cursor = await connection.execute(
                    "SELECT created_at FROM job_events WHERE owner_id = %s "
                    "AND job_id = %s AND sequence = %s",
                    (owner_id, after_job_id, after_sequence),
                )
                row = await cursor.fetchone()
                if row is None:
                    return None
                marker = (row[0], after_job_id, after_sequence)
            if marker is None:
                cursor = await connection.execute(
                    "SELECT payload FROM job_events WHERE owner_id = %s "
                    "ORDER BY created_at, job_id, sequence LIMIT %s",
                    (owner_id, limit),
                )
            else:
                cursor = await connection.execute(
                    "SELECT payload FROM job_events WHERE owner_id = %s "
                    "AND (created_at, job_id, sequence) > (%s, %s, %s) "
                    "ORDER BY created_at, job_id, sequence LIMIT %s",
                    (owner_id, *marker, limit),
                )
            rows = await cursor.fetchall()
        return tuple(JobEvent.model_validate(row[0]) for row in rows)

    async def create_session(
        self,
        *,
        owner_id: UUID,
        device_id: str,
        request: SessionCreateRequest,
        response: SessionResponse,
    ) -> None:
        """Persist conversation and session atomically, scoped to owner."""

        if not device_id.strip() or response.conversation_id != request.conversation_id:
            raise ValueError("session identity does not match its request")
        async with self._pool.connection() as connection:
            async with connection.transaction():
                expected = (device_id, request.locale, request.timezone)
                await connection.execute(
                    "INSERT INTO conversations "
                    "(conversation_id, owner_id, device_id, locale, timezone) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (conversation_id, owner_id) DO NOTHING",
                    (request.conversation_id, owner_id, *expected),
                )
                cursor = await connection.execute(
                    "SELECT device_id, locale, timezone FROM conversations "
                    "WHERE conversation_id = %s AND owner_id = %s",
                    (request.conversation_id, owner_id),
                )
                if await cursor.fetchone() != expected:
                    raise StorageConflict("conversation ID already belongs to different context")
                try:
                    await connection.execute(
                        "INSERT INTO sessions "
                        "(session_id, owner_id, conversation_id, state, opened_at, "
                        "idle_close_seconds, max_session_seconds) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (
                            response.session_id,
                            owner_id,
                            response.conversation_id,
                            response.state,
                            response.opened_at,
                            response.idle_close_seconds,
                            response.max_session_seconds,
                        ),
                    )
                except errors.UniqueViolation:
                    raise StorageConflict("session ID already exists") from None

    async def get_connecting_session_for_device(
        self, *, owner_id: UUID, device_id: str, session_id: UUID
    ) -> SessionResponse | None:
        """Load one unclaimed session through its exact owner/device mapping."""

        if not device_id.strip():
            raise ValueError("device ID must not be blank")
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT s.session_id, s.conversation_id, s.state, s.opened_at, "
                "s.idle_close_seconds, s.max_session_seconds "
                "FROM sessions s JOIN conversations c "
                "ON c.owner_id = s.owner_id AND c.conversation_id = s.conversation_id "
                "WHERE s.owner_id = %s AND c.device_id = %s AND s.session_id = %s "
                "AND s.state = 'connecting'",
                (owner_id, device_id, session_id),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return SessionResponse(
            session_id=row[0],
            conversation_id=row[1],
            state=row[2],
            opened_at=row[3],
            idle_close_seconds=row[4],
            max_session_seconds=row[5],
        )

    async def claim_session_start(
        self, *, owner_id: UUID, device_id: str, session_id: UUID
    ) -> bool:
        """Atomically reserve one connecting session before opening provider socket."""

        if not device_id.strip():
            raise ValueError("device ID must not be blank")
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    "UPDATE sessions s SET state = 'starting' "
                    "WHERE s.owner_id = %s AND s.session_id = %s AND s.state = 'connecting' "
                    "AND EXISTS (SELECT 1 FROM conversations c "
                    "WHERE c.owner_id = s.owner_id AND c.conversation_id = s.conversation_id "
                    "AND c.device_id = %s) RETURNING s.session_id",
                    (owner_id, session_id, device_id),
                )
                return await cursor.fetchone() is not None

    async def mark_session_ready(
        self, *, owner_id: UUID, device_id: str, session_id: UUID
    ) -> bool:
        """Claim a connecting session for one owner/device WebSocket."""

        if not device_id.strip():
            raise ValueError("device ID must not be blank")
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    "UPDATE sessions s SET state = 'ready' "
                    "WHERE s.owner_id = %s AND s.session_id = %s AND s.state = 'starting' "
                    "AND EXISTS (SELECT 1 FROM conversations c "
                    "WHERE c.owner_id = s.owner_id AND c.conversation_id = s.conversation_id "
                    "AND c.device_id = %s) RETURNING s.session_id",
                    (owner_id, session_id, device_id),
                )
                return await cursor.fetchone() is not None

    async def close_session(
        self, *, owner_id: UUID, session_id: UUID, device_id: str | None = None
    ) -> bool:
        """Close an owned session, optionally requiring its authenticated device."""

        if device_id is not None and not device_id.strip():
            raise ValueError("device ID must not be blank")
        predicate = "owner_id = %s AND session_id = %s AND state != 'closed'"
        values: tuple[object, ...] = (owner_id, session_id)
        if device_id is not None:
            predicate += (
                " AND EXISTS (SELECT 1 FROM conversations c "
                "WHERE c.owner_id = sessions.owner_id "
                "AND c.conversation_id = sessions.conversation_id AND c.device_id = %s)"
            )
            values += (device_id,)
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    "UPDATE sessions SET state = 'closed', closed_at = now() "
                    f"WHERE {predicate} "
                    "RETURNING opened_at, closed_at",
                    values,
                )
                row = await cursor.fetchone()
                if row is None:
                    return False
                seconds = Decimal(str(max(0.0, (row[1] - row[0]).total_seconds())))
                await connection.execute(
                    "INSERT INTO usage_measurements "
                    "(usage_id, owner_id, session_id, job_id, provider, input_tokens, "
                    "output_tokens, connected_seconds, cost_usd, cost_kind, observed_at) "
                    "VALUES (%s, %s, %s, NULL, 'gateway-session', NULL, NULL, %s, NULL, NULL, %s) "
                    "ON CONFLICT (usage_id) DO NOTHING",
                    (
                        uuid5(NAMESPACE_URL, f"cube-session-usage:{session_id}"),
                        owner_id,
                        session_id,
                        seconds,
                        row[1],
                    ),
                )
                return True

    async def list_sessions(
        self, *, owner_id: UUID, limit: int = 100
    ) -> tuple[SessionResponse, ...]:
        """List recent sessions owned by the authenticated account."""

        if not 1 <= limit <= 500:
            raise ValueError("session list limit must be between 1 and 500")
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT session_id, conversation_id, state, opened_at, "
                "idle_close_seconds, max_session_seconds FROM sessions "
                "WHERE owner_id = %s ORDER BY opened_at DESC, session_id LIMIT %s",
                (owner_id, limit),
            )
            rows = await cursor.fetchall()
        return tuple(
            SessionResponse(
                session_id=row[0],
                conversation_id=row[1],
                state=row[2],
                opened_at=row[3],
                idle_close_seconds=row[4],
                max_session_seconds=row[5],
            )
            for row in rows
        )

    async def save_memory(self, *, owner_id: UUID, memory: MemoryRecord) -> None:
        """Insert one immutable memory record for its authenticated owner."""

        async with self._pool.connection() as connection:
            async with connection.transaction():
                try:
                    await connection.execute(
                        "INSERT INTO memories "
                        "(memory_id, owner_id, category, content, source_turn_id, created_at, updated_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (
                            memory.memory_id,
                            owner_id,
                            memory.category,
                            memory.content,
                            memory.source_turn_id,
                            memory.created_at,
                            memory.updated_at,
                        ),
                    )
                except errors.UniqueViolation:
                    raise StorageConflict("memory ID already exists") from None

    async def list_memories(self, *, owner_id: UUID) -> tuple[MemoryRecord, ...]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT memory_id, category, content, source_turn_id, created_at, updated_at "
                "FROM memories WHERE owner_id = %s ORDER BY updated_at DESC, memory_id",
                (owner_id,),
            )
            rows = await cursor.fetchall()
        return tuple(
            MemoryRecord(
                memory_id=row[0],
                category=row[1],
                content=row[2],
                source_turn_id=row[3],
                created_at=row[4],
                updated_at=row[5],
            )
            for row in rows
        )

    async def delete_memory(self, *, owner_id: UUID, memory_id: UUID) -> bool:
        """Physically delete only a memory owned by the authenticated owner."""

        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    "DELETE FROM memories WHERE owner_id = %s AND memory_id = %s "
                    "RETURNING memory_id",
                    (owner_id, memory_id),
                )
                return await cursor.fetchone() is not None

    async def claim_action(
        self,
        *,
        owner_id: UUID,
        action_id: UUID,
        trace_id: UUID,
        job_id: UUID | None,
        tool_name: str,
        resource_key: str,
        payload: dict,
        idempotency_key: str,
    ) -> ActionClaim:
        """Persist intent before a write can reach an external executor."""

        if not idempotency_key.strip() or not tool_name.strip() or not resource_key.strip():
            raise ValueError("action identity fields must not be blank")
        digest = _digest_json(
            {"tool_name": tool_name, "resource_key": resource_key, "payload": payload}
        )
        async with self._pool.connection() as connection:
            async with connection.transaction():
                try:
                    cursor = await connection.execute(
                        "INSERT INTO action_ledger "
                        "(action_id, owner_id, job_id, trace_id, tool_name, resource_key, "
                        "payload, payload_digest, idempotency_key, state) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'executing') "
                        "ON CONFLICT (owner_id, idempotency_key) DO NOTHING "
                        "RETURNING action_id",
                        (
                            action_id,
                            owner_id,
                            job_id,
                            trace_id,
                            tool_name,
                            resource_key,
                            Jsonb(payload),
                            digest,
                            idempotency_key,
                        ),
                    )
                    inserted = await cursor.fetchone()
                except errors.UniqueViolation:
                    raise StorageConflict("action ID already exists") from None
                if inserted is not None:
                    return ActionClaim(
                        outcome="execute", action_id=action_id, state="executing"
                    )
                cursor = await connection.execute(
                    "SELECT action_id, payload_digest, state, result FROM action_ledger "
                    "WHERE owner_id = %s AND idempotency_key = %s FOR UPDATE",
                    (owner_id, idempotency_key),
                )
                existing = await cursor.fetchone()
                if existing is None:
                    raise StorageConflict("action claim disappeared")
                existing_id, existing_digest, state, result = existing
                if existing_digest != digest:
                    raise IdempotencyConflict(
                        "idempotency key already belongs to a different action"
                    )
                if state == "succeeded":
                    return ActionClaim(
                        outcome="completed",
                        action_id=existing_id,
                        state=state,
                        result=result,
                    )
                return ActionClaim(
                    outcome="blocked", action_id=existing_id, state=state
                )

    async def complete_action(
        self, *, owner_id: UUID, action_id: UUID, result, verified: bool
    ) -> None:
        """Record verified output; repeated identical completion is harmless."""

        if not verified:
            raise ValueError("only verified action output can be marked complete")
        safe_result = _json_safe(result)
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    "UPDATE action_ledger SET state = 'succeeded', result = %s, updated_at = now() "
                    "WHERE owner_id = %s AND action_id = %s AND state = 'executing' "
                    "RETURNING action_id",
                    (Jsonb(safe_result), owner_id, action_id),
                )
                if await cursor.fetchone() is not None:
                    return
                cursor = await connection.execute(
                    "SELECT state, result FROM action_ledger "
                    "WHERE owner_id = %s AND action_id = %s",
                    (owner_id, action_id),
                )
                existing = await cursor.fetchone()
                if existing is None or existing[0] != "succeeded" or existing[1] != safe_result:
                    raise StorageConflict("action cannot be marked complete")

    async def mark_action_uncertain(self, *, owner_id: UUID, action_id: UUID) -> None:
        """Prevent retries after an executor exits with an uncertain outcome."""

        async with self._pool.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    "UPDATE action_ledger SET state = 'needs_reconciliation', updated_at = now() "
                    "WHERE owner_id = %s AND action_id = %s AND state = 'executing'",
                    (owner_id, action_id),
                )

    async def create_timer(
        self,
        *,
        owner_id: UUID,
        device_id: str,
        request: TimerCreateRequest,
        timer_id: UUID | None = None,
    ) -> TimerRecord:
        """Create an owner-scoped timer; exact retries return the original record."""

        if not device_id.strip():
            raise ValueError("device ID must not be blank")
        new_timer_id = timer_id or uuid4()
        request_digest = _digest_json(
            {"device_id": device_id, "request": request.model_dump(mode="json")}
        )
        async with self._pool.connection() as connection:
            async with connection.transaction():
                try:
                    cursor = await connection.execute(
                        "INSERT INTO timers "
                        "(timer_id, owner_id, device_id, conversation_id, due_at, message, "
                        "request_digest, status, idempotency_key) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, 'scheduled', %s) "
                        "ON CONFLICT (owner_id, idempotency_key) DO NOTHING "
                        "RETURNING timer_id, conversation_id, due_at, message, status, created_at, delivered_at",
                        (
                            new_timer_id,
                            owner_id,
                            device_id,
                            request.conversation_id,
                            request.due_at,
                            request.message,
                            request_digest,
                            request.idempotency_key,
                        ),
                    )
                    row = await cursor.fetchone()
                except errors.UniqueViolation:
                    raise StorageConflict("timer ID or conversation is invalid") from None
                if row is None:
                    cursor = await connection.execute(
                        "SELECT timer_id, request_digest, conversation_id, due_at, message, "
                        "status, created_at, delivered_at FROM timers "
                        "WHERE owner_id = %s AND idempotency_key = %s",
                        (owner_id, request.idempotency_key),
                    )
                    existing = await cursor.fetchone()
                    if existing is None:
                        raise StorageConflict("timer idempotency record disappeared")
                    if existing[1] != request_digest:
                        raise IdempotencyConflict(
                            "idempotency key already belongs to a different timer"
                        )
                    row = (existing[0], *existing[2:])
        return TimerRecord(
            timer_id=row[0],
            conversation_id=row[1],
            due_at=row[2],
            message=row[3],
            status=row[4],
            created_at=row[5],
            delivered_at=row[6],
        )

    async def list_timers(
        self, *, owner_id: UUID, limit: int = 100
    ) -> tuple[TimerRecord, ...]:
        """List the owner's newest timers."""

        if not 1 <= limit <= 500:
            raise ValueError("timer list limit must be between 1 and 500")
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT timer_id, conversation_id, due_at, message, status, created_at, delivered_at "
                "FROM timers WHERE owner_id = %s ORDER BY created_at DESC, timer_id LIMIT %s",
                (owner_id, limit),
            )
            rows = await cursor.fetchall()
        return tuple(
            TimerRecord(
                timer_id=row[0],
                conversation_id=row[1],
                due_at=row[2],
                message=row[3],
                status=row[4],
                created_at=row[5],
                delivered_at=row[6],
            )
            for row in rows
        )

    async def cancel_timer(self, *, owner_id: UUID, timer_id: UUID) -> bool:
        """Cancel a timer that has not reached terminal delivery state."""

        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "UPDATE timers SET status = 'cancelled', claimed_by = NULL, "
                "lease_expires_at = NULL, retry_after = NULL "
                "WHERE owner_id = %s AND timer_id = %s AND status IN ('scheduled', 'firing') "
                "RETURNING timer_id",
                (owner_id, timer_id),
            )
            return await cursor.fetchone() is not None

    async def claim_due_timers(
        self,
        *,
        worker_id: UUID,
        lease_seconds: int = 30,
        max_attempts: int = 8,
        limit: int = 10,
    ) -> tuple[ClaimedTimer, ...]:
        """Claim due timers and recover expired deliveries without blocking peers."""

        if lease_seconds <= 0 or max_attempts <= 0 or not 1 <= limit <= 100:
            raise ValueError("timer lease, attempt cap, or batch limit is invalid")
        async with self._pool.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    "UPDATE timers SET status = 'failed', claimed_by = NULL, "
                    "lease_expires_at = NULL, retry_after = NULL "
                    "WHERE attempt_count >= %s AND ((status = 'scheduled' "
                    "AND COALESCE(retry_after, due_at) <= now()) OR "
                    "(status = 'firing' AND lease_expires_at <= now()))",
                    (max_attempts,),
                )
                cursor = await connection.execute(
                    "WITH candidate AS ("
                    " SELECT timer_id FROM timers"
                    " WHERE attempt_count < %s AND ((status = 'scheduled' "
                    " AND COALESCE(retry_after, due_at) <= now()) OR"
                    " (status = 'firing' AND lease_expires_at <= now()))"
                    " ORDER BY COALESCE(retry_after, due_at), timer_id"
                    " LIMIT %s FOR UPDATE SKIP LOCKED"
                    ") UPDATE timers AS timer SET status = 'firing', claimed_by = %s,"
                    " lease_expires_at = now() + (%s * interval '1 second'),"
                    " retry_after = NULL, attempt_count = timer.attempt_count + 1"
                    " FROM candidate WHERE timer.timer_id = candidate.timer_id"
                    " RETURNING timer.owner_id, timer.timer_id, timer.device_id,"
                    " timer.conversation_id, timer.due_at, timer.message,"
                    " timer.attempt_count, timer.lease_expires_at",
                    (max_attempts, limit, worker_id, lease_seconds),
                )
                rows = await cursor.fetchall()
        claimed_at = datetime.now(timezone.utc)
        return tuple(
            ClaimedTimer(
                owner_id=row[0],
                worker_id=worker_id,
                event=TimerFiredEvent(
                    event_id=row[1],
                    timer_id=row[1],
                    device_id=row[2],
                    conversation_id=row[3],
                    due_at=row[4],
                    fired_at=claimed_at,
                    message=row[5],
                    attempt_count=row[6],
                ),
                lease_expires_at=row[7],
            )
            for row in rows
        )

    async def renew_timer_lease(
        self, *, timer_id: UUID, worker_id: UUID, lease_seconds: int = 30
    ) -> bool:
        """Extend a live timer delivery lease held by this worker."""

        if lease_seconds <= 0:
            raise ValueError("timer lease must be positive")
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "UPDATE timers SET lease_expires_at = now() + (%s * interval '1 second') "
                "WHERE timer_id = %s AND status = 'firing' AND claimed_by = %s "
                "AND lease_expires_at > now() RETURNING timer_id",
                (lease_seconds, timer_id, worker_id),
            )
            return await cursor.fetchone() is not None

    async def ack_timer_delivery(
        self, *, owner_id: UUID, timer_id: UUID, worker_id: UUID
    ) -> bool:
        """Mark a delivered timer only while the worker still owns its lease."""

        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "UPDATE timers SET status = 'delivered', delivered_at = now(), "
                "claimed_by = NULL, lease_expires_at = NULL, retry_after = NULL "
                "WHERE owner_id = %s AND timer_id = %s AND status = 'firing' "
                "AND claimed_by = %s AND lease_expires_at > now() RETURNING timer_id",
                (owner_id, timer_id, worker_id),
            )
            return await cursor.fetchone() is not None

    async def retry_timer_delivery(
        self,
        *,
        owner_id: UUID,
        timer_id: UUID,
        worker_id: UUID,
        retry_after: datetime,
        max_attempts: int = 8,
    ) -> bool:
        """Schedule bounded retry or mark terminal failure under a live lease."""

        if max_attempts <= 0:
            raise ValueError("timer attempt cap must be positive")
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "UPDATE timers SET status = CASE WHEN attempt_count >= %s THEN 'failed' "
                "ELSE 'scheduled' END, retry_after = CASE WHEN attempt_count >= %s "
                "THEN NULL ELSE %s END, claimed_by = NULL, lease_expires_at = NULL "
                "WHERE owner_id = %s AND timer_id = %s AND status = 'firing' "
                "AND claimed_by = %s AND lease_expires_at > now() RETURNING timer_id",
                (max_attempts, max_attempts, retry_after, owner_id, timer_id, worker_id),
            )
            return await cursor.fetchone() is not None

    async def create_approval(
        self,
        *,
        owner_id: UUID,
        approval_id: UUID,
        action_id: UUID,
        payload_digest: str,
        context_revision: int,
        expires_at: datetime,
    ) -> ApprovalRecord:
        """Create an approval bound to an existing action intent and exact digest."""

        if context_revision < 0 or expires_at.tzinfo is None or expires_at.utcoffset() is None:
            raise ValueError("approval revision or expiry is invalid")
        if expires_at <= datetime.now(timezone.utc):
            raise ValueError("approval expiry must be in the future")
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    "SELECT action.tool_name, action.resource_key, action.payload_digest, "
                    "action.job_id, job.context_revision FROM action_ledger AS action "
                    "LEFT JOIN jobs AS job ON job.job_id = action.job_id AND job.owner_id = action.owner_id "
                    "WHERE action.owner_id = %s AND action.action_id = %s FOR UPDATE OF action",
                    (owner_id, action_id),
                )
                action = await cursor.fetchone()
                if action is None:
                    raise StorageConflict("approval action does not exist")
                if action[2] != payload_digest:
                    raise StorageConflict("approval digest does not match action intent")
                if action[3] is not None and action[4] != context_revision:
                    raise StorageConflict("approval revision does not match current job context")
                try:
                    await connection.execute(
                        "INSERT INTO approvals "
                        "(approval_id, action_id, owner_id, payload_digest, context_revision, status, expires_at) "
                        "VALUES (%s, %s, %s, %s, %s, 'pending', %s) "
                        "ON CONFLICT (approval_id) DO NOTHING",
                        (
                            approval_id,
                            action_id,
                            owner_id,
                            payload_digest,
                            context_revision,
                            expires_at,
                        ),
                    )
                except errors.UniqueViolation:
                    raise StorageConflict("approval ID already exists") from None
                cursor = await connection.execute(
                    "SELECT approval.action_id, approval.payload_digest, approval.context_revision, "
                    "approval.status, approval.expires_at, action.tool_name, action.resource_key "
                    "FROM approvals AS approval JOIN action_ledger AS action "
                    "ON action.action_id = approval.action_id AND action.owner_id = approval.owner_id "
                    "WHERE approval.approval_id = %s AND approval.owner_id = %s",
                    (approval_id, owner_id),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise StorageConflict("approval ID already belongs to another owner")
                if (
                    row[0] != action_id
                    or row[1] != payload_digest
                    or row[2] != context_revision
                    or row[4] != expires_at
                ):
                    raise StorageConflict("approval ID already belongs to different intent")
        return ApprovalRecord(
            approval_id=approval_id,
            action_id=row[0],
            action_summary=f"{row[5]} on {row[6]}"[:500],
            payload_digest=row[1],
            context_revision=row[2],
            status=row[3],
            expires_at=row[4],
        )

    async def list_approvals(
        self, *, owner_id: UUID, include_decided: bool = False, limit: int = 100
    ) -> tuple[ApprovalRecord, ...]:
        """List owner approvals; expire overdue pending rows before returning them."""

        if not 1 <= limit <= 500:
            raise ValueError("approval list limit must be between 1 and 500")
        async with self._pool.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    "UPDATE approvals SET status = 'expired' "
                    "WHERE owner_id = %s AND status = 'pending' AND expires_at <= now()",
                    (owner_id,),
                )
                status_filter = "" if include_decided else "AND approval.status = 'pending'"
                cursor = await connection.execute(
                    "SELECT approval.approval_id, approval.action_id, action.tool_name, "
                    "action.resource_key, approval.payload_digest, approval.context_revision, "
                    "approval.status, approval.expires_at "
                    "FROM approvals AS approval JOIN action_ledger AS action "
                    "ON action.action_id = approval.action_id AND action.owner_id = approval.owner_id "
                    "WHERE approval.owner_id = %s " + status_filter +
                    " ORDER BY approval.expires_at, approval.approval_id LIMIT %s",
                    (owner_id, limit),
                )
                rows = await cursor.fetchall()
        return tuple(
            ApprovalRecord(
                approval_id=row[0],
                action_id=row[1],
                action_summary=f"{row[2]} on {row[3]}"[:500],
                payload_digest=row[4],
                context_revision=row[5],
                status=row[6],
                expires_at=row[7],
            )
            for row in rows
        )

    async def decide_approval(
        self,
        *,
        owner_id: UUID,
        approval_id: UUID,
        request: ApprovalDecisionRequest,
        idempotency_key: str | None = None,
    ) -> ApprovalDecisionResponse | None:
        """Apply one exact, unexpired approval decision; exact retries are idempotent."""

        target_status = "approved" if request.decision == "approve" else "denied"
        operation = "approval-decision"
        request_digest = _digest_json({"approval_id": approval_id, "request": request})
        async with self._pool.connection() as connection:
            async with connection.transaction():
                if idempotency_key is not None:
                    cursor = await connection.execute(
                        "SELECT 1 FROM approvals WHERE owner_id = %s AND approval_id = %s",
                        (owner_id, approval_id),
                    )
                    if await cursor.fetchone() is None:
                        return None
                    replay = await self._reserve_idempotency_tx(
                        connection,
                        owner_id=owner_id,
                        operation=operation,
                        idempotency_key=idempotency_key,
                        request_digest=request_digest,
                    )
                    if replay is not None:
                        return ApprovalDecisionResponse.model_validate(replay)
                await connection.execute(
                    "UPDATE approvals SET status = 'expired' "
                    "WHERE owner_id = %s AND approval_id = %s AND status = 'pending' "
                    "AND expires_at <= now()",
                    (owner_id, approval_id),
                )
                await connection.execute(
                    "UPDATE approvals AS approval SET status = 'expired' "
                    "FROM action_ledger AS action LEFT JOIN jobs AS job "
                    "ON job.job_id = action.job_id AND job.owner_id = action.owner_id "
                    "WHERE approval.owner_id = %s AND approval.approval_id = %s "
                    "AND approval.status = 'pending' AND action.owner_id = approval.owner_id "
                    "AND action.action_id = approval.action_id AND action.job_id IS NOT NULL "
                    "AND job.context_revision IS DISTINCT FROM approval.context_revision",
                    (owner_id, approval_id),
                )
                cursor = await connection.execute(
                    "UPDATE approvals AS approval SET status = %s, decided_at = now() "
                    "WHERE approval.owner_id = %s AND approval.approval_id = %s AND approval.status = 'pending' "
                    "AND approval.expires_at > now() AND approval.payload_digest = %s "
                    "AND approval.context_revision = %s AND EXISTS ("
                    "SELECT 1 FROM action_ledger AS action LEFT JOIN jobs AS job "
                    "ON job.job_id = action.job_id AND job.owner_id = action.owner_id "
                    "WHERE action.action_id = approval.action_id AND action.owner_id = approval.owner_id "
                    "AND (action.job_id IS NULL OR job.context_revision = approval.context_revision)) "
                    "RETURNING status, decided_at",
                    (
                        target_status,
                        owner_id,
                        approval_id,
                        request.payload_digest,
                        request.context_revision,
                    ),
                )
                row = await cursor.fetchone()
                if row is None:
                    cursor = await connection.execute(
                        "SELECT status, payload_digest, context_revision, decided_at FROM approvals "
                        "WHERE owner_id = %s AND approval_id = %s FOR UPDATE",
                        (owner_id, approval_id),
                    )
                    existing = await cursor.fetchone()
                    if existing is None:
                        return None
                    if (
                        existing[0] != target_status
                        or existing[1] != request.payload_digest
                        or existing[2] != request.context_revision
                        or existing[3] is None
                    ):
                        raise StorageConflict("approval is expired, decided, or bound to different intent")
                    row = existing[0], existing[3]
                result = ApprovalDecisionResponse(
                    approval_id=approval_id, status=row[0], decided_at=row[1]
                )
                if idempotency_key is not None:
                    await self._store_idempotency_response_tx(
                        connection,
                        owner_id=owner_id,
                        operation=operation,
                        idempotency_key=idempotency_key,
                        response=result,
                    )
        return result

    async def update_memory(
        self, *, owner_id: UUID, memory_id: UUID, request: MemoryUpdateRequest
    ) -> MemoryRecord | None:
        """Replace owner memory content and provenance without changing its identity."""

        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "UPDATE memories SET content = %s, source_turn_id = %s, updated_at = now() "
                "WHERE owner_id = %s AND memory_id = %s "
                "RETURNING memory_id, category, content, source_turn_id, created_at, updated_at",
                (request.content, request.source_turn_id, owner_id, memory_id),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return MemoryRecord(
            memory_id=row[0],
            category=row[1],
            content=row[2],
            source_turn_id=row[3],
            created_at=row[4],
            updated_at=row[5],
        )

    async def write_memory(
        self,
        *,
        owner_id: UUID,
        request: MemoryWriteRequest,
        idempotency_key: str,
    ) -> tuple[MemoryRecord, bool]:
        """Create memory once per owner/key and replay its original response."""

        digest = _digest_json(request)
        async with self._pool.connection() as connection:
            async with connection.transaction():
                replay = await self._reserve_idempotency_tx(
                    connection,
                    owner_id=owner_id,
                    operation="memory-write",
                    idempotency_key=idempotency_key,
                    request_digest=digest,
                )
                if replay is not None:
                    return MemoryRecord.model_validate(replay), True
                now = datetime.now(timezone.utc)
                memory = MemoryRecord(
                    memory_id=uuid4(),
                    category=request.category,
                    content=request.content,
                    source_turn_id=request.source_turn_id,
                    created_at=now,
                    updated_at=now,
                )
                await connection.execute(
                    "INSERT INTO memories "
                    "(memory_id, owner_id, category, content, source_turn_id, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        memory.memory_id,
                        owner_id,
                        memory.category,
                        memory.content,
                        memory.source_turn_id,
                        memory.created_at,
                        memory.updated_at,
                    ),
                )
                await self._store_idempotency_response_tx(
                    connection,
                    owner_id=owner_id,
                    operation="memory-write",
                    idempotency_key=idempotency_key,
                    response=memory,
                )
        return memory, False

    async def update_memory_idempotent(
        self,
        *,
        owner_id: UUID,
        memory_id: UUID,
        request: MemoryUpdateRequest,
        idempotency_key: str,
    ) -> tuple[MemoryRecord | None, bool]:
        """Owner-scoped memory update with durable key/digest/response replay."""

        digest = _digest_json({"memory_id": memory_id, "request": request})
        operation = "memory-update"
        async with self._pool.connection() as connection:
            async with connection.transaction():
                replay = await self._reserve_idempotency_tx(
                    connection,
                    owner_id=owner_id,
                    operation=operation,
                    idempotency_key=idempotency_key,
                    request_digest=digest,
                )
                if replay is not None:
                    return MemoryRecord.model_validate(replay), True
                cursor = await connection.execute(
                    "UPDATE memories SET content = %s, source_turn_id = %s, updated_at = now() "
                    "WHERE owner_id = %s AND memory_id = %s "
                    "RETURNING memory_id, category, content, source_turn_id, created_at, updated_at",
                    (request.content, request.source_turn_id, owner_id, memory_id),
                )
                row = await cursor.fetchone()
                if row is None:
                    await connection.execute(
                        "DELETE FROM idempotency_records WHERE owner_id = %s AND operation = %s "
                        "AND idempotency_key = %s",
                        (owner_id, operation, idempotency_key),
                    )
                    return None, False
                memory = MemoryRecord(
                    memory_id=row[0],
                    category=row[1],
                    content=row[2],
                    source_turn_id=row[3],
                    created_at=row[4],
                    updated_at=row[5],
                )
                await self._store_idempotency_response_tx(
                    connection,
                    owner_id=owner_id,
                    operation=operation,
                    idempotency_key=idempotency_key,
                    response=memory,
                )
        return memory, False

    async def delete_memory_idempotent(
        self, *, owner_id: UUID, memory_id: UUID, idempotency_key: str
    ) -> bool:
        """Owner-scoped memory delete with durable response replay."""

        operation = "memory-delete"
        digest = _digest_json({"memory_id": memory_id})
        async with self._pool.connection() as connection:
            async with connection.transaction():
                replay = await self._reserve_idempotency_tx(
                    connection,
                    owner_id=owner_id,
                    operation=operation,
                    idempotency_key=idempotency_key,
                    request_digest=digest,
                )
                if replay is not None:
                    return bool(replay)
                cursor = await connection.execute(
                    "DELETE FROM memories WHERE owner_id = %s AND memory_id = %s RETURNING memory_id",
                    (owner_id, memory_id),
                )
                deleted = await cursor.fetchone() is not None
                await self._store_idempotency_response_tx(
                    connection,
                    owner_id=owner_id,
                    operation=operation,
                    idempotency_key=idempotency_key,
                    response=deleted,
                )
        return deleted

    async def record_usage(
        self,
        *,
        owner_id: UUID,
        provider: str,
        measurement: UsageMeasurement,
        session_id: UUID | None = None,
        job_id: UUID | None = None,
        usage_id: UUID | None = None,
        observed_at: datetime | None = None,
    ) -> UsageRecord:
        """Persist token, connection-time, and cost measurements with explicit labels."""

        if not provider.strip():
            raise ValueError("usage provider must not be blank")
        record_id = usage_id or uuid4()
        timestamp = observed_at or datetime.now(timezone.utc)
        record = UsageRecord(
            usage_id=record_id,
            session_id=session_id,
            job_id=job_id,
            provider=provider,
            measurement=measurement,
            observed_at=timestamp,
        )
        try:
            async with self._pool.connection() as connection:
                await connection.execute(
                    "INSERT INTO usage_measurements "
                    "(usage_id, owner_id, session_id, job_id, provider, input_tokens, output_tokens, "
                    "connected_seconds, cost_usd, cost_kind, observed_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        record_id,
                        owner_id,
                        session_id,
                        job_id,
                        provider,
                        measurement.input_tokens,
                        measurement.output_tokens,
                        measurement.connected_seconds,
                        measurement.cost_usd,
                        measurement.cost_kind,
                        timestamp,
                    ),
                )
        except errors.UniqueViolation:
            raise StorageConflict("usage ID already exists") from None
        except errors.ForeignKeyViolation:
            raise StorageConflict("usage session or job does not belong to owner") from None
        return record

    async def record_usage_once(
        self,
        *,
        owner_id: UUID,
        usage_id: UUID,
        provider: str,
        measurement: UsageMeasurement,
        session_id: UUID | None = None,
        job_id: UUID | None = None,
        observed_at: datetime | None = None,
    ) -> UsageRecord:
        """Persist a stable usage identity; exact retries return original record."""

        if not provider.strip():
            raise ValueError("usage provider must not be blank")
        candidate = UsageRecord(
            usage_id=usage_id,
            session_id=session_id,
            job_id=job_id,
            provider=provider,
            measurement=measurement,
            observed_at=observed_at or datetime.now(timezone.utc),
        )
        timestamp = candidate.observed_at
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    cursor = await connection.execute(
                        "INSERT INTO usage_measurements "
                        "(usage_id, owner_id, session_id, job_id, provider, input_tokens, output_tokens, "
                        "connected_seconds, cost_usd, cost_kind, observed_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (usage_id) DO NOTHING "
                        "RETURNING owner_id, session_id, job_id, provider, input_tokens, output_tokens, "
                        "connected_seconds, cost_usd, cost_kind, observed_at",
                        (
                            usage_id,
                            owner_id,
                            session_id,
                            job_id,
                            provider,
                            measurement.input_tokens,
                            measurement.output_tokens,
                            measurement.connected_seconds,
                            measurement.cost_usd,
                            measurement.cost_kind,
                            timestamp,
                        ),
                    )
                    row = await cursor.fetchone()
                    if row is None:
                        cursor = await connection.execute(
                            "SELECT owner_id, session_id, job_id, provider, input_tokens, output_tokens, "
                            "connected_seconds, cost_usd, cost_kind, observed_at "
                            "FROM usage_measurements WHERE usage_id = %s",
                            (usage_id,),
                        )
                        row = await cursor.fetchone()
                        expected = (
                            owner_id,
                            session_id,
                            job_id,
                            provider,
                            measurement.input_tokens,
                            measurement.output_tokens,
                            measurement.connected_seconds,
                            measurement.cost_usd,
                            measurement.cost_kind,
                        )
                        if row is None or tuple(row[:9]) != expected:
                            raise StorageConflict("usage ID already belongs to another measurement")
        except errors.ForeignKeyViolation:
            raise StorageConflict("usage session or job does not belong to owner") from None
        return UsageRecord(
            usage_id=usage_id,
            session_id=row[1],
            job_id=row[2],
            provider=row[3],
            measurement=UsageMeasurement(
                input_tokens=row[4],
                output_tokens=row[5],
                connected_seconds=row[6],
                cost_usd=row[7],
                cost_kind=row[8],
            ),
            observed_at=row[9],
        )

    async def list_usage(
        self,
        *,
        owner_id: UUID,
        session_id: UUID | None = None,
        job_id: UUID | None = None,
        limit: int = 500,
    ) -> tuple[UsageRecord, ...]:
        """List usage records for one owner, optionally narrowed to a session or job."""

        if not 1 <= limit <= 2000:
            raise ValueError("usage list limit must be between 1 and 2000")
        clauses = ["owner_id = %s"]
        values: list[object] = [owner_id]
        if session_id is not None:
            clauses.append("session_id = %s")
            values.append(session_id)
        if job_id is not None:
            clauses.append("job_id = %s")
            values.append(job_id)
        values.append(limit)
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT usage_id, session_id, job_id, provider, input_tokens, output_tokens, "
                "connected_seconds, cost_usd, cost_kind, observed_at FROM usage_measurements "
                "WHERE " + " AND ".join(clauses) +
                " ORDER BY observed_at DESC, usage_id LIMIT %s",
                values,
            )
            rows = await cursor.fetchall()
        return tuple(
            UsageRecord(
                usage_id=row[0],
                session_id=row[1],
                job_id=row[2],
                provider=row[3],
                measurement=UsageMeasurement(
                    input_tokens=row[4],
                    output_tokens=row[5],
                    connected_seconds=row[6],
                    cost_usd=row[7],
                    cost_kind=row[8],
                ),
                observed_at=row[9],
            )
            for row in rows
        )

    async def summarize_usage(
        self,
        *,
        owner_id: UUID,
        period_start: datetime,
        period_end: datetime | None = None,
    ) -> UsageSummary:
        """Aggregate owner usage in a half-open UTC billing period."""

        if period_start.tzinfo is None or period_start.utcoffset() is None:
            raise ValueError("usage period start must be timezone-aware")
        if period_end is not None and (
            period_end.tzinfo is None
            or period_end.utcoffset() is None
            or period_end <= period_start
        ):
            raise ValueError("usage period end must follow its aware start")
        async with self._pool.connection() as connection:
            bounds = "observed_at >= %s"
            parameters: tuple = (owner_id, period_start)
            if period_end is not None:
                bounds += " AND observed_at < %s"
                parameters += (period_end,)
            cursor = await connection.execute(
                "SELECT COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), "
                "COALESCE(SUM(connected_seconds) FILTER (WHERE provider != 'gateway-session'), 0), "
                "SUM(cost_usd) FILTER (WHERE cost_kind = 'measured'), "
                "SUM(cost_usd) FILTER (WHERE cost_kind = 'estimated') "
                f"FROM usage_measurements WHERE owner_id = %s AND {bounds}",
                parameters,
            )
            row = await cursor.fetchone()
        return UsageSummary(
            period_start=period_start,
            input_tokens=row[0],
            output_tokens=row[1],
            connected_seconds=row[2],
            measured_cost_usd=row[3],
            estimated_cost_usd=row[4],
        )

    async def register_artifact(
        self,
        *,
        owner_id: UUID,
        job_id: UUID,
        artifact_ref: ArtifactRef,
        storage_key: str,
        byte_size: int,
        sha256: str,
    ) -> StoredArtifact:
        """Persist immutable owner-scoped metadata after blob storage succeeds."""

        if (
            not storage_key.strip()
            or byte_size < 0
            or re.fullmatch(r"[a-f0-9]{64}", sha256) is None
        ):
            raise ValueError("artifact metadata is invalid")
        try:
            async with self._pool.connection() as connection:
                cursor = await connection.execute(
                    "INSERT INTO artifacts "
                    "(artifact_id, owner_id, job_id, label, media_type, storage_key, byte_size, sha256) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                    "RETURNING created_at",
                    (
                        artifact_ref.artifact_id,
                        owner_id,
                        job_id,
                        artifact_ref.label,
                        artifact_ref.media_type,
                        storage_key,
                        byte_size,
                        sha256,
                    ),
                )
                row = await cursor.fetchone()
        except errors.UniqueViolation:
            raise StorageConflict("artifact ID or storage key already exists") from None
        except errors.ForeignKeyViolation:
            raise StorageConflict("artifact job does not belong to owner") from None
        return StoredArtifact(
            artifact_ref=artifact_ref,
            job_id=job_id,
            storage_key=storage_key,
            byte_size=byte_size,
            sha256=sha256,
            created_at=row[0],
        )

    async def get_artifact(
        self, *, owner_id: UUID, artifact_id: UUID
    ) -> StoredArtifact | None:
        """Read active metadata only for the authenticated owner."""

        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT job_id, label, media_type, storage_key, byte_size, sha256, created_at "
                "FROM artifacts WHERE owner_id = %s AND artifact_id = %s AND deleted_at IS NULL",
                (owner_id, artifact_id),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return StoredArtifact(
            artifact_ref=ArtifactRef(artifact_id=artifact_id, label=row[1], media_type=row[2]),
            job_id=row[0],
            storage_key=row[3],
            byte_size=row[4],
            sha256=row[5],
            created_at=row[6],
        )

    async def list_artifacts(
        self, *, owner_id: UUID, job_id: UUID | None = None, limit: int = 100
    ) -> tuple[StoredArtifact, ...]:
        """List active artifact metadata for an owner or one owned job."""

        if not 1 <= limit <= 500:
            raise ValueError("artifact list limit must be between 1 and 500")
        params: tuple[object, ...]
        if job_id is None:
            where, params = "owner_id = %s AND deleted_at IS NULL", (owner_id, limit)
        else:
            where, params = (
                "owner_id = %s AND job_id = %s AND deleted_at IS NULL",
                (owner_id, job_id, limit),
            )
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT artifact_id, job_id, label, media_type, storage_key, byte_size, sha256, created_at "
                "FROM artifacts WHERE " + where + " ORDER BY created_at DESC, artifact_id LIMIT %s",
                params,
            )
            rows = await cursor.fetchall()
        return tuple(
            StoredArtifact(
                artifact_ref=ArtifactRef(artifact_id=row[0], label=row[2], media_type=row[3]),
                job_id=row[1],
                storage_key=row[4],
                byte_size=row[5],
                sha256=row[6],
                created_at=row[7],
            )
            for row in rows
        )

    async def delete_artifact_metadata(self, *, owner_id: UUID, artifact_id: UUID) -> bool:
        """Soft-delete metadata so a cleanup worker can remove the backing blob."""

        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "UPDATE artifacts SET deleted_at = now() WHERE owner_id = %s "
                "AND artifact_id = %s AND deleted_at IS NULL RETURNING artifact_id",
                (owner_id, artifact_id),
            )
            return await cursor.fetchone() is not None

    async def list_deleted_artifacts(
        self, *, limit: int = 100
    ) -> tuple[tuple[UUID, UUID], ...]:
        """Return owner/artifact pairs whose blob cleanup remains pending."""

        if not 1 <= limit <= 500:
            raise ValueError("artifact cleanup batch must be between 1 and 500")
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT owner_id, artifact_id FROM artifacts WHERE deleted_at IS NOT NULL "
                "ORDER BY deleted_at, artifact_id LIMIT %s",
                (limit,),
            )
            rows = await cursor.fetchall()
        return tuple((row[0], row[1]) for row in rows)

    async def purge_deleted_artifact(
        self, *, owner_id: UUID, artifact_id: UUID
    ) -> bool:
        """Remove tombstoned metadata only after blob deletion succeeds."""

        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "DELETE FROM artifacts WHERE owner_id = %s AND artifact_id = %s "
                "AND deleted_at IS NOT NULL RETURNING artifact_id",
                (owner_id, artifact_id),
            )
            return await cursor.fetchone() is not None
