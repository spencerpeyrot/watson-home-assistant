from __future__ import annotations

import json
import os
import sqlite3
import stat
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import httpx

from contracts.models import (
    JobAccepted,
    JobCorrectionRequest,
    JobEvent,
    JobSnapshot,
    TaskRequest,
)
from voice.jobs import GatewayJobClient, JobBridgeError, SQLiteJobLinkStore


JOB_ID = UUID("00000000-0000-4000-8000-000000000981")
CONVERSATION_ID = UUID("00000000-0000-4000-8000-000000000982")
SESSION_ID = UUID("00000000-0000-4000-8000-000000000983")
TURN_ID = UUID("00000000-0000-4000-8000-000000000984")
TRACE_ID = UUID("00000000-0000-4000-8000-000000000985")


def request() -> TaskRequest:
    return TaskRequest(
        device_id="cube-01",
        conversation_id=CONVERSATION_ID,
        session_id=SESSION_ID,
        turn_id=TURN_ID,
        delegation_id="delegation-job-1",
        trace_id=TRACE_ID,
        request_text="Compare three options.",
        context_revision=0,
        locale="en-US",
        timezone="America/New_York",
        idempotency_key="cube-01/session/delegation-job-1",
    )


def event(sequence: int, status: str = "running") -> JobEvent:
    return JobEvent(
        device_id="cube-01",
        job_id=JOB_ID,
        event_id=UUID(f"00000000-0000-4000-8000-{sequence:012d}"),
        sequence=sequence,
        kind="progress" if sequence == 1 else "result",
        status=status,
        conversation_id=CONVERSATION_ID,
        session_id=SESSION_ID,
        turn_id=TURN_ID,
        delegation_id="delegation-job-1",
        trace_id=TRACE_ID,
        context_revision=0,
        summary=f"Event {sequence}",
        timestamp=datetime.now(timezone.utc),
    )


def sse(item: JobEvent) -> str:
    return (
        f"id: {item.job_id}:{item.sequence}\n"
        f"event: {item.kind}\n"
        f"data: {json.dumps(item.model_dump(mode='json'), separators=(',', ':'))}\n\n"
    )


class GatewayJobClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_link_database_is_private_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "job-links.sqlite"
            database_path.touch(mode=0o644)
            os.chmod(database_path, 0o644)

            SQLiteJobLinkStore(database_path)

            self.assertEqual(stat.S_IMODE(database_path.stat().st_mode), 0o600)
            target = Path(directory) / "target.sqlite"
            target.touch()
            os.chmod(target, 0o644)
            link = Path(directory) / "link.sqlite"
            link.symlink_to(target)
            with self.assertRaises(OSError):
                SQLiteJobLinkStore(link)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

    async def test_existing_link_database_adds_event_id_cache_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "job-links.sqlite"
            with closing(sqlite3.connect(database_path)) as connection:
                with connection:
                    connection.execute(
                        "CREATE TABLE voice_job_links ("
                        "session_id TEXT NOT NULL, delegation_id TEXT NOT NULL, "
                        "job_id TEXT NOT NULL, conversation_id TEXT NOT NULL, "
                        "context_revision INTEGER NOT NULL, last_sequence INTEGER NOT NULL, "
                        "PRIMARY KEY (session_id, delegation_id))"
                    )
                    connection.execute(
                        "INSERT INTO voice_job_links VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            str(SESSION_ID),
                            "delegation-job-1",
                            str(JOB_ID),
                            str(CONVERSATION_ID),
                            0,
                            2,
                        ),
                    )

            client = GatewayJobClient(
                "https://gateway.example",
                "t" * 40,
                transport=httpx.MockTransport(
                    lambda _request: httpx.Response(
                        200,
                        text=sse(event(1)),
                        headers={"content-type": "text/event-stream"},
                    )
                ),
                link_database_path=database_path,
            )
            try:
                restored = client.link("delegation-job-1")
                self.assertIsNotNone(restored)
                assert restored is not None
                self.assertEqual(restored.last_sequence, 2)
                self.assertEqual(restored.event_ids, {})
                with self.assertRaisesRegex(
                    JobBridgeError, "outside the retained verification window"
                ):
                    await client.events("delegation-job-1")
            finally:
                await client.close()

    async def test_event_cursor_advances_only_after_entire_batch_validates(self) -> None:
        requests: list[str | None] = []
        event_calls = 0

        async def handle(request: httpx.Request) -> httpx.Response:
            nonlocal event_calls
            if request.method == "POST" and request.url.path == "/v1/jobs":
                accepted = JobAccepted(
                    job_id=JOB_ID,
                    status="running",
                    trace_id=TRACE_ID,
                    deduplicated=False,
                    first_event_sequence=1,
                )
                return httpx.Response(202, json=accepted.model_dump(mode="json"))
            requests.append(request.headers.get("last-event-id"))
            event_calls += 1
            body = (
                sse(event(1)) + sse(event(3, "succeeded"))
                if event_calls == 1
                else sse(event(1)) + sse(event(2, "succeeded"))
            )
            return httpx.Response(
                200,
                text=body,
                headers={"content-type": "text/event-stream"},
            )

        client = GatewayJobClient(
            "https://gateway.example",
            "t" * 40,
            transport=httpx.MockTransport(handle),
        )
        try:
            await client.submit(request())
            with self.assertRaisesRegex(JobBridgeError, "sequence has a gap"):
                await client.events("delegation-job-1")
            link = client.link("delegation-job-1")
            self.assertIsNotNone(link)
            assert link is not None
            self.assertEqual(link.last_sequence, 0)
            self.assertEqual(
                [item.sequence for item in await client.events("delegation-job-1")],
                [1, 2],
            )
            self.assertEqual(requests, [None, None])
        finally:
            await client.close()

    async def test_job_link_and_event_cursor_survive_client_restart_without_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "job-links.sqlite"
            progress, result = event(1), event(2, "succeeded")

            async def first_handle(request: httpx.Request) -> httpx.Response:
                if request.method == "POST" and request.url.path == "/v1/jobs":
                    accepted = JobAccepted(
                        job_id=JOB_ID,
                        status="running",
                        trace_id=TRACE_ID,
                        deduplicated=False,
                        first_event_sequence=1,
                    )
                    return httpx.Response(202, json=accepted.model_dump(mode="json"))
                if request.url.path.endswith("/events"):
                    return httpx.Response(
                        200,
                        text=sse(progress) + sse(result),
                        headers={"content-type": "text/event-stream"},
                    )
                if request.url.path.endswith("/corrections"):
                    snapshot = JobSnapshot(
                        job_id=JOB_ID,
                        status="running",
                        context_revision=1,
                        last_sequence=3,
                        summary="Correction accepted.",
                    )
                    return httpx.Response(202, json=snapshot.model_dump(mode="json"))
                return httpx.Response(404)

            first_client = GatewayJobClient(
                "https://gateway.example",
                "t" * 40,
                transport=httpx.MockTransport(first_handle),
                link_database_path=database_path,
            )
            try:
                await first_client.submit(request())
                self.assertEqual(len(await first_client.events("delegation-job-1")), 2)
                await first_client.correct(
                    "delegation-job-1",
                    JobCorrectionRequest(
                        context_revision=1, request_text="Compare two options."
                    ),
                )
            finally:
                await first_client.close()

            stored = database_path.read_bytes()
            self.assertNotIn(b"Compare three options.", stored)

            resumed_cursors: list[str | None] = []
            resumed_calls = 0

            async def resumed_handle(request: httpx.Request) -> httpx.Response:
                nonlocal resumed_calls
                resumed_calls += 1
                resumed_cursors.append(request.headers.get("last-event-id"))
                if resumed_calls == 1:
                    altered_replay = event(1).model_copy(
                        update={"event_id": UUID("00000000-0000-4000-8000-000000000999")}
                    )
                    return httpx.Response(
                        200,
                        text=sse(altered_replay) + sse(event(4, "succeeded")),
                        headers={"content-type": "text/event-stream"},
                    )
                return httpx.Response(
                    200,
                    text=sse(event(4, "succeeded")),
                    headers={"content-type": "text/event-stream"},
                )

            resumed_client = GatewayJobClient(
                "https://gateway.example",
                "t" * 40,
                transport=httpx.MockTransport(resumed_handle),
                link_database_path=database_path,
            )
            try:
                restored = resumed_client.link("delegation-job-1")
                self.assertIsNotNone(restored)
                assert restored is not None
                self.assertEqual(restored.job_id, JOB_ID)
                self.assertEqual(restored.context_revision, 1)
                self.assertEqual(restored.last_sequence, 3)
                with self.assertRaisesRegex(JobBridgeError, "sequence was reused"):
                    await resumed_client.events("delegation-job-1")
                self.assertEqual(restored.last_sequence, 3)
                self.assertEqual(
                    [item.sequence for item in await resumed_client.events("delegation-job-1")],
                    [4],
                )
                self.assertEqual(resumed_cursors, [f"{JOB_ID}:3", f"{JOB_ID}:3"])
            finally:
                await resumed_client.close()

    async def test_submission_events_correction_and_cancel_stay_correlated(self) -> None:
        calls: list[tuple[str, str, str | None, str | None]] = []
        progress, result = event(1), event(2, "succeeded")

        async def handle(request: httpx.Request) -> httpx.Response:
            calls.append(
                (
                    request.method,
                    request.url.path,
                    request.headers.get("last-event-id"),
                    request.headers.get("idempotency-key"),
                )
            )
            if request.method == "POST" and request.url.path == "/v1/jobs":
                body = JobAccepted(
                    job_id=JOB_ID,
                    status="running",
                    trace_id=TRACE_ID,
                    deduplicated=False,
                    first_event_sequence=1,
                )
                return httpx.Response(202, json=body.model_dump(mode="json"))
            if request.method == "GET" and request.url.path.endswith("/events"):
                return httpx.Response(
                    200,
                    text=sse(progress) + sse(result),
                    headers={"content-type": "text/event-stream"},
                )
            if request.url.path.endswith("/corrections"):
                body = JobSnapshot(
                    job_id=JOB_ID,
                    status="running",
                    context_revision=1,
                    last_sequence=3,
                    summary="Correction accepted.",
                )
                return httpx.Response(202, json=body.model_dump(mode="json"))
            if request.url.path.endswith("/cancel"):
                body = JobSnapshot(
                    job_id=JOB_ID,
                    status="cancelled",
                    context_revision=1,
                    last_sequence=4,
                    summary="Cancelled.",
                )
                return httpx.Response(202, json=body.model_dump(mode="json"))
            return httpx.Response(404)

        client = GatewayJobClient(
            "https://gateway.example",
            "t" * 40,
            transport=httpx.MockTransport(handle),
        )
        try:
            accepted = await client.submit(request())
            self.assertEqual(accepted.job_id, JOB_ID)
            with self.assertRaisesRegex(JobBridgeError, "already owns"):
                await client.submit(request())
            events = await client.events("delegation-job-1")
            self.assertEqual([item.sequence for item in events], [1, 2])
            self.assertEqual(calls[-1][2], None)
            self.assertEqual(await client.events("delegation-job-1"), ())
            self.assertEqual(calls[-1][2], f"{JOB_ID}:2")
            corrected = await client.correct(
                "delegation-job-1",
                JobCorrectionRequest(context_revision=1, request_text="Compare two options."),
            )
            self.assertEqual(corrected.context_revision, 1)
            cancelled = await client.cancel("delegation-job-1")
            self.assertEqual(cancelled.status, "cancelled")
            self.assertEqual(calls[-2][1], f"/v1/jobs/{JOB_ID}/corrections")
            self.assertEqual(calls[-1][1], f"/v1/jobs/{JOB_ID}/cancel")
            self.assertEqual(calls[-2][3], f"job/{JOB_ID}/correction/1")
            self.assertEqual(calls[-1][3], f"job/{JOB_ID}/cancel")
        finally:
            await client.close()

    async def test_rejects_foreign_job_event_and_wrong_correction_revision(self) -> None:
        progress = event(1)

        async def handle(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                accepted = JobAccepted(
                    job_id=JOB_ID,
                    status="running",
                    trace_id=TRACE_ID,
                    deduplicated=False,
                    first_event_sequence=1,
                )
                return httpx.Response(202, json=accepted.model_dump(mode="json"))
            foreign = progress.model_copy(update={"delegation_id": "different"})
            return httpx.Response(
                200,
                text=sse(foreign),
                headers={"content-type": "text/event-stream"},
            )

        client = GatewayJobClient(
            "http://localhost",
            "t" * 40,
            transport=httpx.MockTransport(handle),
        )
        try:
            await client.submit(request())
            with self.assertRaisesRegex(JobBridgeError, "correction"):
                await client.correct(
                    "delegation-job-1",
                    JobCorrectionRequest(context_revision=2, request_text="Wrong revision."),
                )
            with self.assertRaisesRegex(JobBridgeError, "correlation"):
                await client.events("delegation-job-1")
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
