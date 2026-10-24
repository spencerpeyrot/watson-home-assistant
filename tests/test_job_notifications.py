from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from stat import S_IMODE
from uuid import UUID

import httpx

from contracts.models import JobSnapshot
from device.job_notifications import (
    DeviceJobCompletionService,
    JobCompletionMonitor,
    SQLiteJobNotificationStore,
)
from voice.jobs import GatewayJobClient, JobLink, SQLiteJobLinkStore


JOB_ID = UUID("00000000-0000-4000-8000-000000000981")
CONVERSATION_ID = UUID("00000000-0000-4000-8000-000000000982")
SESSION_ID = UUID("00000000-0000-4000-8000-000000000983")


def snapshot(
    *, status: str = "succeeded", sequence: int = 5, revision: int = 0
) -> JobSnapshot:
    return JobSnapshot(
        job_id=JOB_ID,
        status=status,
        context_revision=revision,
        last_sequence=sequence,
        summary="Result ready.",
    )


class JobNotificationStoreTests(unittest.TestCase):
    def test_old_inbox_schema_adds_revision_supersede_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notifications.sqlite"
            with closing(sqlite3.connect(path)) as connection:
                with connection:
                    connection.execute(
                        "CREATE TABLE voice_job_notifications ("
                        "job_id TEXT NOT NULL, context_revision INTEGER NOT NULL, "
                        "delegation_id TEXT NOT NULL, snapshot_json TEXT NOT NULL, "
                        "chime_state TEXT NOT NULL, spoken_at TEXT, "
                        "created_at TEXT NOT NULL, "
                        "PRIMARY KEY (job_id, context_revision))"
                    )

            SQLiteJobNotificationStore(path)

            with closing(sqlite3.connect(path)) as connection:
                columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(voice_job_notifications)"
                    )
                }
            self.assertIn("superseded", columns)

    def test_terminal_summary_persists_once_and_waits_for_explicit_speech(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notifications.sqlite"
            path.touch(mode=0o644)
            path.chmod(0o644)
            inbox = SQLiteJobNotificationStore(path)
            self.assertEqual(S_IMODE(path.stat().st_mode), 0o600)

            self.assertTrue(inbox.record_terminal_once("delegation-1", snapshot()))
            self.assertFalse(inbox.record_terminal_once("delegation-1", snapshot()))
            self.assertEqual(inbox.pending_chimes(), (snapshot(),))
            self.assertEqual(inbox.pending_summaries(), (snapshot(),))

            inbox.mark_chimed(JOB_ID, 0)
            self.assertEqual(inbox.pending_chimes(), ())
            self.assertEqual(inbox.pending_summaries(), (snapshot(),))

            inbox.mark_spoken(JOB_ID, 0)
            self.assertEqual(inbox.pending_summaries(), ())
            self.assertNotIn(b"request_text", path.read_bytes())

    def test_only_terminal_snapshots_can_enter_inbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inbox = SQLiteJobNotificationStore(Path(directory) / "notifications.sqlite")
            with self.assertRaisesRegex(ValueError, "only terminal"):
                inbox.record_terminal_once("delegation-1", snapshot(status="running"))


class JobCompletionMonitorTests(unittest.IsolatedAsyncioTestCase):
    async def test_service_session_client_shares_monitor_link_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            async def notify() -> None:
                return None

            service = DeviceJobCompletionService(
                gateway_url="https://gateway.example",
                device_token="t" * 40,
                data_dir=directory,
                notify=notify,
            )
            session_jobs = service.new_session_job_client()
            try:
                self.assertIsNone(await service())
                self.assertEqual(
                    service.link_database_path,
                    Path(directory) / "voice-job-links.sqlite3",
                )
                SQLiteJobLinkStore(service.link_database_path).save(
                    JobLink(
                        delegation_id="delegation-1",
                        job_id=JOB_ID,
                        session_id=SESSION_ID,
                        conversation_id=CONVERSATION_ID,
                        context_revision=0,
                    )
                )
                session_jobs.refresh_links()
                link = session_jobs.link("delegation-1")
                self.assertIsNotNone(link)
                assert link is not None
                self.assertEqual(link.job_id, JOB_ID)
            finally:
                await session_jobs.close()
                await service.close()

    async def test_unseen_terminal_snapshot_is_chimed_once_and_queued(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            links_path = Path(directory) / "links.sqlite"
            inbox = SQLiteJobNotificationStore(Path(directory) / "inbox.sqlite")
            SQLiteJobLinkStore(links_path).save(
                JobLink(
                    delegation_id="delegation-1",
                    job_id=JOB_ID,
                    session_id=SESSION_ID,
                    conversation_id=CONVERSATION_ID,
                    context_revision=0,
                    last_sequence=4,
                )
            )

            snapshot_calls = 0

            async def handle(request: httpx.Request) -> httpx.Response:
                nonlocal snapshot_calls
                snapshot_calls += 1
                self.assertEqual(request.method, "GET")
                self.assertEqual(request.url.path, f"/v1/jobs/{JOB_ID}")
                return httpx.Response(200, json=snapshot().model_dump(mode="json"))

            jobs = GatewayJobClient(
                "https://gateway.example",
                "t" * 40,
                transport=httpx.MockTransport(handle),
                link_database_path=links_path,
            )
            chimes = 0

            async def notify() -> None:
                nonlocal chimes
                chimes += 1

            monitor = JobCompletionMonitor(jobs, inbox, notify)
            try:
                self.assertEqual(await monitor.check_once(), 1)
                self.assertEqual(await monitor.check_once(), 0)
                self.assertEqual(snapshot_calls, 1)
                self.assertEqual(chimes, 1)
                self.assertEqual(inbox.pending_chimes(), ())
                self.assertEqual(inbox.pending_summaries(), (snapshot(),))
            finally:
                await jobs.close()

    async def test_consumed_terminal_sequence_is_not_chimed_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            links_path = Path(directory) / "links.sqlite"
            inbox = SQLiteJobNotificationStore(Path(directory) / "inbox.sqlite")
            SQLiteJobLinkStore(links_path).save(
                JobLink(
                    delegation_id="delegation-1",
                    job_id=JOB_ID,
                    session_id=SESSION_ID,
                    conversation_id=CONVERSATION_ID,
                    context_revision=0,
                    last_sequence=5,
                )
            )

            async def handle(_request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, json=snapshot().model_dump(mode="json"))

            jobs = GatewayJobClient(
                "https://gateway.example",
                "t" * 40,
                transport=httpx.MockTransport(handle),
                link_database_path=links_path,
            )
            called = False

            async def notify() -> None:
                nonlocal called
                called = True

            try:
                self.assertEqual(await JobCompletionMonitor(jobs, inbox, notify).check_once(), 0)
                self.assertFalse(called)
                self.assertEqual(inbox.pending_summaries(), ())
            finally:
                await jobs.close()

    async def test_failed_chime_retries_from_persistent_inbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            links_path = Path(directory) / "links.sqlite"
            inbox = SQLiteJobNotificationStore(Path(directory) / "inbox.sqlite")
            SQLiteJobLinkStore(links_path).save(
                JobLink(
                    delegation_id="delegation-1",
                    job_id=JOB_ID,
                    session_id=SESSION_ID,
                    conversation_id=CONVERSATION_ID,
                    context_revision=0,
                    last_sequence=4,
                )
            )

            async def handle(_request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, json=snapshot().model_dump(mode="json"))

            jobs = GatewayJobClient(
                "https://gateway.example",
                "t" * 40,
                transport=httpx.MockTransport(handle),
                link_database_path=links_path,
            )
            attempts = 0

            async def notify() -> None:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("speaker unavailable")

            monitor = JobCompletionMonitor(jobs, inbox, notify)
            try:
                self.assertEqual(await monitor.check_once(), 0)
                self.assertEqual(inbox.pending_summaries(), (snapshot(),))
                self.assertEqual(await monitor.check_once(), 1)
                self.assertEqual(attempts, 2)
                self.assertEqual(inbox.pending_chimes(), ())
            finally:
                await jobs.close()

    async def test_new_correction_revision_supersedes_older_pending_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            links_path = Path(directory) / "links.sqlite"
            inbox = SQLiteJobNotificationStore(Path(directory) / "inbox.sqlite")
            link_store = SQLiteJobLinkStore(links_path)
            link_store.save(
                JobLink(
                    delegation_id="delegation-1",
                    job_id=JOB_ID,
                    session_id=SESSION_ID,
                    conversation_id=CONVERSATION_ID,
                    context_revision=0,
                    last_sequence=4,
                )
            )
            current_snapshot = snapshot()

            async def handle(_request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200, json=current_snapshot.model_dump(mode="json")
                )

            jobs = GatewayJobClient(
                "https://gateway.example",
                "t" * 40,
                transport=httpx.MockTransport(handle),
                link_database_path=links_path,
            )
            chimes = 0

            async def notify() -> None:
                nonlocal chimes
                chimes += 1

            monitor = JobCompletionMonitor(jobs, inbox, notify)
            try:
                self.assertEqual(await monitor.check_once(), 1)
                self.assertEqual(inbox.pending_summaries(), (snapshot(),))

                link_store.save(
                    JobLink(
                        delegation_id="delegation-1",
                        job_id=JOB_ID,
                        session_id=SESSION_ID,
                        conversation_id=CONVERSATION_ID,
                        context_revision=1,
                        last_sequence=5,
                    )
                )
                current_snapshot = snapshot(sequence=6, revision=1)
                self.assertEqual(await monitor.check_once(), 1)
                self.assertEqual(inbox.pending_summaries(), (current_snapshot,))
                self.assertEqual(chimes, 2)
            finally:
                await jobs.close()
