"""Persist terminal delegated-job results for idle-only local notification."""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from contracts.models import JobSnapshot
from voice.job_orchestrator import TERMINAL_JOB_STATES
from voice.jobs import GatewayJobClient


_LOGGER = logging.getLogger(__name__)


class SQLiteJobNotificationStore:
    """Keep terminal summaries locally; never store the originating request."""

    def __init__(self, path: str | Path) -> None:
        database_path = Path(path)
        if str(database_path) == ":memory:":
            raise ValueError("job notification inbox requires a durable file path")
        database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(database_path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        self._path = str(database_path)
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS voice_job_notifications ("
                "job_id TEXT NOT NULL, context_revision INTEGER NOT NULL, "
                "delegation_id TEXT NOT NULL, snapshot_json TEXT NOT NULL, "
                "chime_state TEXT NOT NULL CHECK(chime_state IN ('pending', 'chimed')), "
                "spoken_at TEXT, superseded INTEGER NOT NULL DEFAULT 0, "
                "created_at TEXT NOT NULL, "
                "PRIMARY KEY (job_id, context_revision))"
            )
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(voice_job_notifications)"
                )
            }
            if "superseded" not in columns:
                connection.execute(
                    "ALTER TABLE voice_job_notifications ADD COLUMN "
                    "superseded INTEGER NOT NULL DEFAULT 0"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS voice_job_notifications_pending_chime "
                "ON voice_job_notifications (chime_state, created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS voice_job_notifications_pending_speech "
                "ON voice_job_notifications (spoken_at, created_at)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS voice_job_terminal_observations ("
                "job_id TEXT NOT NULL, context_revision INTEGER NOT NULL, "
                "last_sequence INTEGER NOT NULL, observed_at TEXT NOT NULL, "
                "PRIMARY KEY (job_id, context_revision))"
            )

    def record_terminal_once(
        self, delegation_id: str, snapshot: JobSnapshot
    ) -> bool:
        """Insert one terminal result per job revision, without replacing it."""

        if snapshot.status not in TERMINAL_JOB_STATES:
            raise ValueError("only terminal job snapshots can be recorded")
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO voice_job_notifications "
                "(job_id, context_revision, delegation_id, snapshot_json, "
                "chime_state, spoken_at, created_at) "
                "VALUES (?, ?, ?, ?, 'pending', NULL, ?)",
                (
                    str(snapshot.job_id),
                    snapshot.context_revision,
                    delegation_id,
                    snapshot.model_dump_json(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return cursor.rowcount == 1

    def pending_chimes(self) -> tuple[JobSnapshot, ...]:
        return self._load(
            "SELECT job_id, context_revision, snapshot_json "
            "FROM voice_job_notifications WHERE chime_state = 'pending' "
            "AND superseded = 0 "
            "ORDER BY created_at, job_id, context_revision"
        )

    def pending_summaries(self) -> tuple[JobSnapshot, ...]:
        """Return persisted results awaiting a future, explicitly chosen speech flow."""

        return self._load(
            "SELECT job_id, context_revision, snapshot_json "
            "FROM voice_job_notifications WHERE spoken_at IS NULL "
            "AND superseded = 0 "
            "ORDER BY created_at, job_id, context_revision"
        )

    def mark_chimed(self, job_id: UUID, context_revision: int) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE voice_job_notifications SET chime_state = 'chimed' "
                "WHERE job_id = ? AND context_revision = ?",
                (str(job_id), context_revision),
            )

    def mark_spoken(self, job_id: UUID, context_revision: int) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE voice_job_notifications SET spoken_at = ? "
                "WHERE job_id = ? AND context_revision = ? AND spoken_at IS NULL",
                (datetime.now(timezone.utc).isoformat(), str(job_id), context_revision),
            )

    def supersede_before(self, job_id: UUID, context_revision: int) -> None:
        """Hide results made stale by a newer correction revision."""

        with self._connection() as connection:
            connection.execute(
                "UPDATE voice_job_notifications SET superseded = 1 "
                "WHERE job_id = ? AND context_revision < ?",
                (str(job_id), context_revision),
            )

    def has_terminal_observation(self, job_id: UUID, context_revision: int) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM voice_job_terminal_observations "
                "WHERE job_id = ? AND context_revision = ?",
                (str(job_id), context_revision),
            ).fetchone()
        return row is not None

    def record_terminal_observation(
        self, job_id: UUID, context_revision: int, last_sequence: int
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO voice_job_terminal_observations "
                "(job_id, context_revision, last_sequence, observed_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(job_id, context_revision) DO UPDATE SET "
                "last_sequence = MAX(last_sequence, excluded.last_sequence), "
                "observed_at = excluded.observed_at",
                (
                    str(job_id),
                    context_revision,
                    last_sequence,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def _load(self, query: str) -> tuple[JobSnapshot, ...]:
        with self._connection() as connection:
            rows = connection.execute(query).fetchall()
        snapshots: list[JobSnapshot] = []
        for raw_job_id, raw_revision, raw_snapshot in rows:
            try:
                snapshot = JobSnapshot.model_validate_json(raw_snapshot)
                if (
                    snapshot.job_id != UUID(raw_job_id)
                    or snapshot.context_revision != raw_revision
                    or snapshot.status not in TERMINAL_JOB_STATES
                ):
                    raise ValueError("persisted job notification identity is invalid")
            except (ValueError, TypeError):
                raise ValueError("persisted job notification is invalid") from None
            snapshots.append(snapshot)
        return tuple(snapshots)

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


class JobCompletionMonitor:
    """Poll persisted same-device links and chime for unseen terminal results."""

    def __init__(
        self,
        jobs: GatewayJobClient,
        inbox: SQLiteJobNotificationStore,
        notify: Callable[[], Awaitable[None]],
    ) -> None:
        self._jobs = jobs
        self._inbox = inbox
        self._notify = notify

    async def __call__(self) -> None:
        await self.check_once()

    async def check_once(self) -> int:
        """Persist newly completed jobs and retry pending chimes; return count."""

        try:
            self._jobs.refresh_links()
        except Exception as error:
            _LOGGER.warning("job link refresh failed (%s)", type(error).__name__)
            return 0

        for link in self._jobs.links:
            try:
                self._inbox.supersede_before(link.job_id, link.context_revision)
            except Exception as error:
                _LOGGER.warning(
                    "stale job result cleanup failed (%s)", type(error).__name__
                )
                continue
            try:
                if self._inbox.has_terminal_observation(
                    link.job_id, link.context_revision
                ):
                    continue
            except Exception as error:
                _LOGGER.warning(
                    "job terminal cursor read failed (%s)", type(error).__name__
                )
                continue
            try:
                snapshot = await self._jobs.snapshot(link.delegation_id)
            except Exception as error:
                _LOGGER.warning("job snapshot poll failed (%s)", type(error).__name__)
                continue
            if snapshot.status not in TERMINAL_JOB_STATES:
                continue
            if snapshot.context_revision < link.context_revision:
                continue
            try:
                if snapshot.last_sequence > link.last_sequence:
                    self._inbox.record_terminal_once(link.delegation_id, snapshot)
                self._inbox.record_terminal_observation(
                    snapshot.job_id,
                    snapshot.context_revision,
                    snapshot.last_sequence,
                )
            except Exception as error:
                _LOGGER.warning("job result persistence failed (%s)", type(error).__name__)

        chimed = 0
        try:
            pending = self._inbox.pending_chimes()
        except Exception as error:
            _LOGGER.warning("pending job result read failed (%s)", type(error).__name__)
            return 0
        for snapshot in pending:
            try:
                await self._notify()
            except Exception as error:
                _LOGGER.warning("job result chime failed (%s)", type(error).__name__)
                break
            try:
                self._inbox.mark_chimed(snapshot.job_id, snapshot.context_revision)
            except Exception as error:
                _LOGGER.warning("job result chime state failed (%s)", type(error).__name__)
                break
            chimed += 1
        return chimed


class DeviceJobCompletionService:
    """Own the long-lived monitor client and shared per-device job-link path."""

    def __init__(
        self,
        *,
        gateway_url: str,
        device_token: str,
        data_dir: str | Path,
        notify: Callable[[], Awaitable[None]],
    ) -> None:
        root = Path(data_dir)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.link_database_path = root / "voice-job-links.sqlite3"
        inbox = SQLiteJobNotificationStore(root / "job-notifications.sqlite3")
        self._inbox = inbox
        self._gateway_url = gateway_url
        self._device_token = device_token
        self._jobs = GatewayJobClient(
            gateway_url,
            device_token,
            link_database_path=self.link_database_path,
        )
        self._monitor = JobCompletionMonitor(self._jobs, inbox, notify)

    async def __call__(self) -> None:
        """Poll linked jobs and chime for new terminal results."""

        await self._monitor.check_once()

    def new_session_job_client(self) -> GatewayJobClient:
        """Create a session client backed by the monitor's same durable links."""

        return GatewayJobClient(
            self._gateway_url,
            self._device_token,
            link_database_path=self.link_database_path,
        )

    def pending_summaries(self) -> tuple[JobSnapshot, ...]:
        """Return local terminal summaries; caller must obtain user consent first."""

        return self._inbox.pending_summaries()

    def mark_spoken(self, job_id: UUID, context_revision: int) -> None:
        """Mark a result after its assistant transcript begins in the active session."""

        self._inbox.mark_spoken(job_id, context_revision)

    async def close(self) -> None:
        """Close the long-lived monitor's HTTP client."""

        await self._jobs.close()
