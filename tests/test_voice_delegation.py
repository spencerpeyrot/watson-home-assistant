from __future__ import annotations

import unittest
from uuid import UUID

from voice.delegation import DelegationTranscriptBuffer, build_task_request


CONVERSATION = UUID("00000000-0000-4000-8000-000000000971")
SESSION = UUID("00000000-0000-4000-8000-000000000972")
TURN = UUID("00000000-0000-4000-8000-000000000973")
TRACE = UUID("00000000-0000-4000-8000-000000000974")


class DelegationTranscriptBufferTests(unittest.TestCase):
    def test_waits_for_audio_and_transcript_quiet_then_builds_exact_task(self) -> None:
        buffer = DelegationTranscriptBuffer(transcript_quiet_seconds=1)
        buffer.record_transcript_delta("Compare ", received_at=0.5)
        buffer.record_delegation(delegation_id="delegation-1", target="client", offset_ms=5800)
        self.assertIsNone(buffer.ready(now=4.0))
        buffer.mark_audio_finished()
        self.assertIsNone(buffer.ready(now=1.0))
        buffer.record_transcript_delta("three options.", received_at=1.2)
        self.assertIsNone(buffer.ready(now=2.0))
        committed = buffer.ready(now=2.21)
        assert committed is not None
        self.assertEqual(committed.transcript, "Compare three options.")
        self.assertEqual(committed.offset_ms, 5800)

        request = build_task_request(
            committed,
            device_id="cube-01",
            conversation_id=CONVERSATION,
            session_id=SESSION,
            context_revision=3,
            locale="en-US",
            timezone_name="America/New_York",
            turn_id=TURN,
            trace_id=TRACE,
        )
        self.assertEqual(request.request_text, committed.transcript)
        self.assertEqual(request.delegation_id, "delegation-1")
        self.assertEqual(request.idempotency_key, f"cube-01/{SESSION}/delegation-1")
        self.assertEqual(request.recent_turns[-1].text, committed.transcript)

    def test_missing_transcript_and_duplicate_delegation_never_commit(self) -> None:
        empty = DelegationTranscriptBuffer()
        empty.record_delegation(delegation_id="empty", target="client", offset_ms=0)
        empty.mark_audio_finished()
        self.assertIsNone(empty.ready(now=5))

        buffer = DelegationTranscriptBuffer()
        buffer.record_transcript_delta("hello", received_at=0)
        buffer.record_delegation(delegation_id="repeat", target="client", offset_ms=0)
        buffer.mark_audio_finished()
        self.assertIsNotNone(buffer.ready(now=2))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            buffer.record_delegation(delegation_id="repeat", target="client", offset_ms=0)

    def test_ignores_non_client_delegation(self) -> None:
        buffer = DelegationTranscriptBuffer()
        buffer.record_delegation(delegation_id="server-tool", target="server", offset_ms=0)
        buffer.mark_audio_finished()
        self.assertIsNone(buffer.ready(now=2))


if __name__ == "__main__":
    unittest.main()
