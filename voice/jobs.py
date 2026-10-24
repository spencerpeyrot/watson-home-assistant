"""Typed gateway client and per-session delegation-to-job correlation."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

import httpx

from contracts.models import (
    JobAccepted,
    JobCorrectionRequest,
    JobEvent,
    JobSnapshot,
    TaskRequest,
)


class JobBridgeError(RuntimeError):
    """Sanitized gateway or correlation failure."""


_LOGGER = logging.getLogger(__name__)


@dataclass
class JobLink:
    delegation_id: str
    job_id: UUID
    session_id: UUID
    conversation_id: UUID
    context_revision: int
    last_sequence: int = 0
    event_ids: dict[int, UUID] = field(default_factory=dict)


class SQLiteJobLinkStore:
    """Persist accepted job identity and cursor; never store request text."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            database_path = Path(path)
            database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            flags = os.O_CREAT | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(database_path, flags, 0o600)
            try:
                os.fchmod(descriptor, 0o600)
            finally:
                os.close(descriptor)
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS voice_job_links ("
                "session_id TEXT NOT NULL, delegation_id TEXT NOT NULL, "
                "job_id TEXT NOT NULL, conversation_id TEXT NOT NULL, "
                "context_revision INTEGER NOT NULL, last_sequence INTEGER NOT NULL, "
                "event_ids_json TEXT NOT NULL DEFAULT '{}', "
                "PRIMARY KEY (session_id, delegation_id))"
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(voice_job_links)")
            }
            if "event_ids_json" not in columns:
                connection.execute(
                    "ALTER TABLE voice_job_links ADD COLUMN "
                    "event_ids_json TEXT NOT NULL DEFAULT '{}'"
                )

    def load(self) -> tuple[JobLink, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT delegation_id, job_id, session_id, conversation_id, "
                "context_revision, last_sequence, event_ids_json "
                "FROM voice_job_links "
                "ORDER BY session_id, delegation_id"
            ).fetchall()
        links = []
        for row in rows:
            try:
                raw_ids = json.loads(row[6])
                if not isinstance(raw_ids, dict):
                    raise ValueError("event ID cache must be an object")
                event_ids = {int(sequence): UUID(value) for sequence, value in raw_ids.items()}
                if any(sequence < 1 for sequence in event_ids):
                    raise ValueError("event ID sequence must be positive")
            except (ValueError, TypeError, AttributeError):
                raise ValueError("persisted voice event ID cache is invalid") from None
            links.append(
                JobLink(
                    delegation_id=row[0],
                    job_id=UUID(row[1]),
                    session_id=UUID(row[2]),
                    conversation_id=UUID(row[3]),
                    context_revision=row[4],
                    last_sequence=row[5],
                    event_ids=event_ids,
                )
            )
        return tuple(links)

    def save(self, link: JobLink) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO voice_job_links "
                "(session_id, delegation_id, job_id, conversation_id, "
                "context_revision, last_sequence, event_ids_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id, delegation_id) DO UPDATE SET "
                "job_id = excluded.job_id, conversation_id = excluded.conversation_id, "
                "context_revision = excluded.context_revision, "
                "last_sequence = excluded.last_sequence, "
                "event_ids_json = excluded.event_ids_json",
                (
                    str(link.session_id),
                    link.delegation_id,
                    str(link.job_id),
                    str(link.conversation_id),
                    link.context_revision,
                    link.last_sequence,
                    json.dumps(
                        {str(sequence): str(event_id) for sequence, event_id in sorted(link.event_ids.items())},
                        separators=(",", ":"),
                    ),
                ),
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


class GatewayJobClient:
    """Submit, follow, correct, and cancel jobs by their originating delegation."""

    MAX_EVENT_RESPONSE_BYTES = 2_000_000

    def __init__(
        self,
        gateway_url: str,
        token: str,
        *,
        timeout_seconds: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
        link_database_path: str | Path | None = None,
    ) -> None:
        parsed = urlsplit(gateway_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("gateway URL must be absolute HTTP(S)")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("gateway URL must not include embedded credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("gateway URL must not include a query or fragment")
        if parsed.scheme == "http" and parsed.hostname.lower() not in {
            "localhost", "127.0.0.1", "::1"
        }:
            raise ValueError("unencrypted gateway access is allowed only on loopback")
        if len(token) < 32 or timeout_seconds <= 0:
            raise ValueError("gateway token or timeout is invalid")
        self._client = httpx.AsyncClient(
            base_url=gateway_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )
        self._link_store = (
            SQLiteJobLinkStore(link_database_path)
            if link_database_path is not None
            else None
        )
        self._links: dict[str, JobLink] = {}
        if self._link_store is not None:
            for link in self._link_store.load():
                if link.delegation_id in self._links:
                    raise ValueError(
                        "persisted delegation IDs collide across job sessions"
                    )
                self._links[link.delegation_id] = link

    def link(self, delegation_id: str) -> JobLink | None:
        """Return the job currently owned by one committed delegation."""

        return self._links.get(delegation_id)

    @property
    def links(self) -> tuple[JobLink, ...]:
        """Return persisted job links without exposing request text."""

        return tuple(self._links.values())

    def refresh_links(self) -> None:
        """Reload links written by another session-scoped client."""

        if self._link_store is None:
            return
        links = self._link_store.load()
        refreshed: dict[str, JobLink] = {}
        for link in links:
            if link.delegation_id in refreshed:
                raise ValueError("persisted delegation IDs collide across job sessions")
            refreshed[link.delegation_id] = link
        self._links = refreshed

    async def submit(self, request: TaskRequest) -> JobAccepted:
        """Submit one committed task and reject duplicate local delegation IDs."""

        if request.delegation_id in self._links:
            raise JobBridgeError("delegation already owns a job")
        try:
            response = await self._client.post(
                "/v1/jobs", json=request.model_dump(mode="json")
            )
        except httpx.RequestError as error:
            raise JobBridgeError(f"job submission transport failed ({type(error).__name__})") from None
        if response.status_code != 202:
            raise JobBridgeError(f"job submission rejected (HTTP {response.status_code})")
        try:
            accepted = JobAccepted.model_validate(response.json())
        except (ValueError, TypeError):
            raise JobBridgeError("job gateway returned an invalid acceptance") from None
        if accepted.trace_id != request.trace_id:
            raise JobBridgeError("job acceptance trace does not match request")
        self._links[request.delegation_id] = JobLink(
            delegation_id=request.delegation_id,
            job_id=accepted.job_id,
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            context_revision=request.context_revision,
            last_sequence=accepted.first_event_sequence - 1,
        )
        self._persist_link(self._links[request.delegation_id])
        return accepted

    async def events(self, delegation_id: str) -> tuple[JobEvent, ...]:
        """Read new ordered SSE events and verify job/delegation correlation."""

        link = self._require_link(delegation_id)
        headers = {}
        if link.last_sequence:
            headers["Last-Event-ID"] = f"{link.job_id}:{link.last_sequence}"
        try:
            response = await self._client.get(
                f"/v1/jobs/{link.job_id}/events", headers=headers
            )
        except httpx.RequestError as error:
            raise JobBridgeError(f"job event transport failed ({type(error).__name__})") from None
        if response.status_code != 200:
            raise JobBridgeError(f"job event request rejected (HTTP {response.status_code})")
        if len(response.content) > self.MAX_EVENT_RESPONSE_BYTES:
            raise JobBridgeError("job event response exceeded size limit")
        if "text/event-stream" not in response.headers.get("content-type", ""):
            raise JobBridgeError("job event response has invalid content type")
        new_events: list[JobEvent] = []
        next_sequence = link.last_sequence
        event_ids = dict(link.event_ids)
        normalized_stream = response.text.replace("\r\n", "\n")
        for block in normalized_stream.split("\n\n"):
            if not block.strip():
                continue
            fields: dict[str, str] = {}
            for line in block.splitlines():
                if line.startswith(":") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                fields[key] = value.lstrip()
            if "data" not in fields:
                continue
            try:
                event = JobEvent.model_validate(json.loads(fields["data"]))
                event_id = fields.get("id", "")
                expected_id = f"{link.job_id}:{event.sequence}"
            except (ValueError, TypeError):
                raise JobBridgeError("job event payload is invalid") from None
            if (
                event.job_id != link.job_id
                or event.delegation_id != delegation_id
                or event.conversation_id != link.conversation_id
                or event.session_id != link.session_id
                or event_id != expected_id
            ):
                raise JobBridgeError("job event correlation does not match active delegation")
            prior_event_id = event_ids.get(event.sequence)
            if event.sequence <= next_sequence:
                if prior_event_id is None:
                    raise JobBridgeError(
                        "job event replay is outside the retained verification window"
                    )
                if prior_event_id != event.event_id:
                    raise JobBridgeError("job event sequence was reused")
                continue
            if event.sequence != next_sequence + 1:
                raise JobBridgeError("job event sequence has a gap")
            next_sequence = event.sequence
            event_ids[event.sequence] = event.event_id
            new_events.append(event)
        if new_events:
            link.last_sequence = next_sequence
            link.event_ids = event_ids
            if len(link.event_ids) > 256:
                for sequence in sorted(link.event_ids)[:-256]:
                    del link.event_ids[sequence]
            self._persist_link(link)
        return tuple(new_events)

    async def snapshot(self, delegation_id: str) -> JobSnapshot:
        """Read the latest device-authorized snapshot for one persisted job link."""

        link = self._require_link(delegation_id)
        try:
            response = await self._client.get(f"/v1/jobs/{link.job_id}")
        except httpx.RequestError as error:
            raise JobBridgeError(
                f"job snapshot transport failed ({type(error).__name__})"
            ) from None
        if response.status_code != 200:
            raise JobBridgeError(
                f"job snapshot request rejected (HTTP {response.status_code})"
            )
        if len(response.content) > 256_000:
            raise JobBridgeError("job snapshot response exceeded size limit")
        try:
            snapshot = JobSnapshot.model_validate(response.json())
        except (ValueError, TypeError):
            raise JobBridgeError("job gateway returned an invalid snapshot") from None
        if snapshot.job_id != link.job_id:
            raise JobBridgeError("job snapshot does not match active delegation")
        return snapshot

    async def correct(
        self, delegation_id: str, request: JobCorrectionRequest
    ) -> JobSnapshot:
        """Correct only the job tied to this delegation and next revision."""

        link = self._require_link(delegation_id)
        if request.context_revision != link.context_revision + 1:
            raise JobBridgeError("correction does not use the next context revision")
        try:
            response = await self._client.post(
                f"/v1/jobs/{link.job_id}/corrections",
                json=request.model_dump(mode="json"),
                headers={
                    "Idempotency-Key": (
                        f"job/{link.job_id}/correction/{request.context_revision}"
                    )
                },
            )
        except httpx.RequestError as error:
            raise JobBridgeError(f"job correction transport failed ({type(error).__name__})") from None
        snapshot = self._snapshot(response, expected_status=202, expected_job=link.job_id)
        link.context_revision = request.context_revision
        link.last_sequence = snapshot.last_sequence
        self._persist_link(link)
        return snapshot

    async def cancel(self, delegation_id: str) -> JobSnapshot:
        """Cancel only the job tied to this delegation."""

        link = self._require_link(delegation_id)
        try:
            response = await self._client.post(
                f"/v1/jobs/{link.job_id}/cancel",
                headers={"Idempotency-Key": f"job/{link.job_id}/cancel"},
            )
        except httpx.RequestError as error:
            raise JobBridgeError(f"job cancellation transport failed ({type(error).__name__})") from None
        snapshot = self._snapshot(response, expected_status=202, expected_job=link.job_id)
        link.last_sequence = snapshot.last_sequence
        self._persist_link(link)
        return snapshot

    async def close(self) -> None:
        await self._client.aclose()

    def _require_link(self, delegation_id: str) -> JobLink:
        link = self._links.get(delegation_id)
        if link is None:
            raise JobBridgeError("delegation has no active job")
        return link

    def _persist_link(self, link: JobLink) -> None:
        if self._link_store is None:
            return
        try:
            self._link_store.save(link)
        except sqlite3.Error as error:
            _LOGGER.warning("voice job link persistence failed (%s)", type(error).__name__)

    @staticmethod
    def _snapshot(
        response: httpx.Response, *, expected_status: int, expected_job: UUID
    ) -> JobSnapshot:
        if response.status_code != expected_status:
            raise JobBridgeError(f"job request rejected (HTTP {response.status_code})")
        try:
            snapshot = JobSnapshot.model_validate(response.json())
        except (ValueError, TypeError):
            raise JobBridgeError("job gateway returned an invalid snapshot") from None
        if snapshot.job_id != expected_job:
            raise JobBridgeError("job response does not match active delegation")
        return snapshot
