from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone
from uuid import UUID, uuid4

from contracts.models import JobAccepted, JobEvent, VoiceSessionEvent
from voice.job_orchestrator import VoiceJobCoordinator


OWNER_DEVICE = "cube-01"
CONVERSATION_ID = UUID("00000000-0000-4000-8000-000000000501")
SESSION_ID = UUID("00000000-0000-4000-8000-000000000502")


class FakeSocket:
    def __init__(self) -> None:
        self.contexts = []
        self.sent = asyncio.Event()

    async def send_context_append(self, request):
        self.contexts.append(request)
        self.sent.set()


class FakeJobs:
    def __init__(self) -> None:
        self.requests = []
        self.closed = False
        self.event = None

    async def submit(self, request):
        self.requests.append(request)
        job_id = uuid4()
        now = datetime.now(timezone.utc)
        self.event = JobEvent(
            device_id=request.device_id,
            job_id=job_id,
            event_id=uuid4(),
            sequence=1,
            kind="result",
            status="succeeded",
            conversation_id=request.conversation_id,
            session_id=request.session_id,
            turn_id=request.turn_id,
            delegation_id=request.delegation_id,
            trace_id=request.trace_id,
            context_revision=request.context_revision,
            summary="The light is on.",
            timestamp=now,
        )
        self.job_id = job_id
        return JobAccepted(
            job_id=job_id,
            status="queued",
            trace_id=request.trace_id,
            deduplicated=False,
            first_event_sequence=1,
        )

    async def events(self, delegation_id):
        return (self.event,)

    async def close(self):
        self.closed = True


class VoiceJobCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn_submits_only_after_vad_end_and_returns_result_to_live(self):
        jobs = FakeJobs()
        socket = FakeSocket()
        coordinator = VoiceJobCoordinator(
            jobs=jobs,  # type: ignore[arg-type]
            device_id=OWNER_DEVICE,
            conversation_id=CONVERSATION_ID,
            session_id=SESSION_ID,
            locale="en-US",
            timezone_name="America/New_York",
            transcript_quiet_seconds=0.01,
            poll_interval_seconds=0.01,
        )
        await coordinator.on_event(
            VoiceSessionEvent(
                type="transcript_delta",
                session_id=SESSION_ID,
                direction="user",
                delta="Turn on the kitchen light",
                start_ms=0,
                end_ms=900,
            ),
            socket,  # type: ignore[arg-type]
        )
        await coordinator.on_event(
            VoiceSessionEvent(
                type="delegation_created",
                session_id=SESSION_ID,
                delegation_id="item_live_123",
                target="client",
                offset_ms=900,
            ),
            socket,  # type: ignore[arg-type]
        )
        await asyncio.sleep(0.02)
        self.assertEqual(jobs.requests, [])

        coordinator.mark_audio_finished()
        await asyncio.wait_for(socket.sent.wait(), timeout=1)

        self.assertEqual(len(jobs.requests), 1)
        self.assertEqual(jobs.requests[0].request_text, "Turn on the kitchen light")
        self.assertEqual(jobs.requests[0].delegation_id, "item_live_123")
        self.assertEqual(len(socket.contexts), 1)
        self.assertEqual(socket.contexts[0].kind, "commentary")
        self.assertEqual(socket.contexts[0].delegation_id, "item_live_123")
        self.assertEqual(socket.contexts[0].content, "The light is on.")

        await coordinator.close()
        self.assertTrue(jobs.closed)

    async def test_muse_session_does_not_offer_legacy_pending_job_results(self):
        jobs = FakeJobs()
        socket = FakeSocket()
        pending_reads = []

        def pending_summaries():
            pending_reads.append(True)
            return ()

        coordinator = VoiceJobCoordinator(
            jobs=jobs,  # type: ignore[arg-type]
            device_id=OWNER_DEVICE,
            conversation_id=CONVERSATION_ID,
            session_id=SESSION_ID,
            locale="en-US",
            timezone_name="America/New_York",
            pending_summaries=pending_summaries,
        )
        await coordinator.on_event(
            VoiceSessionEvent(
                type="session_ready", session_id=SESSION_ID, half_duplex=True
            ),
            socket,  # type: ignore[arg-type]
        )

        self.assertEqual(pending_reads, [])
        self.assertEqual(socket.contexts, [])
        await coordinator.close()
        self.assertTrue(jobs.closed)


if __name__ == "__main__":
    unittest.main()
