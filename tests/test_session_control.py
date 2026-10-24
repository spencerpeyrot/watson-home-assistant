from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from uuid import UUID

from device.audio_buffer import BufferedAudioFrame
from device.session_control import (
    MAX_CANDIDATE_SPEECH_SECONDS,
    LocalSessionController,
    PreRollBuffer,
)


SESSION_ID = UUID("00000000-0000-4000-8000-000000000951")


class LocalSessionControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_reply_expectation_sets_window_and_ignored_speech_does_not_extend_it(self):
        async def no_op(*_args) -> None:
            return None

        async def start_session(_frames) -> UUID:
            return SESSION_ID

        controller = LocalSessionController(
            pre_roll=PreRollBuffer(frame_bytes=4),
            wake_detect=lambda _payload: _resolved(True),
            start_session=start_session,
            stop_session=no_op,
            send_audio=no_op,
            flush_playback=no_op,
            speech_detect=lambda payload: payload == b"talk",
            idle_timeout_seconds=3,
        )
        await controller.feed(b"wake")
        controller.note_response_activity("open_answer")
        deadline = controller._last_activity
        self.assertEqual(controller._idle_timeout, 10)

        await controller.feed(b"talk")
        self.assertTrue(controller._speech_active)
        controller.restore_response_deadline()

        self.assertEqual(controller._last_activity, deadline)
        self.assertFalse(controller._speech_active)
        self.assertIsNone(controller._speech_started_at)
        controller.note_response_activity("confirmation")
        self.assertEqual(controller._idle_timeout, 8)
        controller.note_response_activity("closing")
        self.assertEqual(controller._idle_timeout, 0.5)

    async def test_vad_candidate_pauses_close_without_restarting_idle_deadline(self) -> None:
        start = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)

        async def no_op(*_args) -> None:
            return None

        async def start_session(_frames) -> UUID:
            return SESSION_ID

        controller = LocalSessionController(
            pre_roll=PreRollBuffer(frame_bytes=4),
            wake_detect=lambda _payload: _resolved(True),
            start_session=start_session,
            stop_session=no_op,
            send_audio=no_op,
            flush_playback=no_op,
            speech_detect=lambda payload: payload == b"talk",
            idle_timeout_seconds=3,
            speech_updates_idle=False,
        )
        await controller.feed(b"wake", captured_at=start)
        await controller.feed(b"talk", captured_at=start + timedelta(seconds=2))

        self.assertFalse(await controller.close_if_idle(now=start + timedelta(seconds=4)))
        await controller.feed(b"siln", captured_at=start + timedelta(seconds=4))
        self.assertTrue(await controller.close_if_idle(now=start + timedelta(seconds=4)))

    async def test_continuous_speech_candidate_cannot_hold_session_open_forever(self):
        start = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)

        async def no_op(*_args) -> None:
            return None

        async def start_session(_frames) -> UUID:
            return SESSION_ID

        controller = LocalSessionController(
            pre_roll=PreRollBuffer(frame_bytes=4),
            wake_detect=lambda _payload: _resolved(True),
            start_session=start_session,
            stop_session=no_op,
            send_audio=no_op,
            flush_playback=no_op,
            speech_detect=lambda payload: payload == b"talk",
            idle_timeout_seconds=3,
        )
        await controller.feed(b"wake", captured_at=start)
        await controller.feed(b"talk", captured_at=start + timedelta(seconds=1))
        await controller.feed(
            b"talk",
            captured_at=start + timedelta(seconds=MAX_CANDIDATE_SPEECH_SECONDS + 1),
        )

        self.assertTrue(
            await controller.close_if_idle(
                now=start + timedelta(seconds=MAX_CANDIDATE_SPEECH_SECONDS + 4)
            )
        )

    async def test_half_duplex_response_suppresses_echo_vad_but_keeps_audio_relay(self):
        speech_calls = []
        sent = []

        async def no_op(*_args) -> None:
            return None

        async def start_session(_frames) -> UUID:
            return SESSION_ID

        async def send_audio(_session_id, frame) -> None:
            sent.append(frame.payload)

        def speech(payload: bytes) -> bool:
            speech_calls.append(payload)
            return True

        controller = LocalSessionController(
            pre_roll=PreRollBuffer(frame_bytes=4),
            wake_detect=lambda _payload: _resolved(True),
            start_session=start_session,
            stop_session=no_op,
            send_audio=send_audio,
            flush_playback=no_op,
            speech_detect=speech,
            speech_updates_idle=True,
        )
        await controller.feed(b"wake")
        await controller.feed(b"echo", suppress_speech=True)

        self.assertEqual(speech_calls, [])
        self.assertEqual(sent, [b"echo"])

    async def test_wake_handoff_preserves_preroll_interrupts_and_idles_out(self) -> None:
        start = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
        started_with: list[tuple[BufferedAudioFrame, ...]] = []
        sent: list[bytes] = []
        stopped: list[UUID] = []
        flushes = 0

        async def detect(payload: bytes) -> bool:
            return payload == b"wake"

        async def start_session(frames: tuple[BufferedAudioFrame, ...]) -> UUID:
            started_with.append(frames)
            return SESSION_ID

        async def stop_session(session_id: UUID) -> None:
            stopped.append(session_id)

        async def send_audio(_session_id: UUID, frame: BufferedAudioFrame) -> None:
            sent.append(frame.payload)

        async def flush_playback() -> None:
            nonlocal flushes
            flushes += 1

        controller = LocalSessionController(
            pre_roll=PreRollBuffer(frame_bytes=4, capacity_frames=2),
            wake_detect=detect,
            start_session=start_session,
            stop_session=stop_session,
            send_audio=send_audio,
            flush_playback=flush_playback,
            speech_detect=lambda payload: payload == b"talk",
            idle_timeout_seconds=5,
        )

        self.assertFalse(await controller.feed(b"one!", captured_at=start))
        self.assertFalse(
            await controller.feed(b"two!", captured_at=start + timedelta(milliseconds=20))
        )
        self.assertTrue(await controller.feed(b"wake", captured_at=start + timedelta(milliseconds=40)))
        self.assertEqual(controller.phase, "active")
        self.assertEqual(controller.session_id, SESSION_ID)
        self.assertEqual([frame.payload for frame in started_with[0]], [b"two!", b"wake"])
        self.assertEqual(
            [frame.captured_at for frame in started_with[0]],
            [start + timedelta(milliseconds=20), start + timedelta(milliseconds=40)],
        )

        await controller.feed(b"talk", captured_at=start + timedelta(seconds=1))
        self.assertEqual(sent, [b"talk"])
        self.assertTrue(await controller.interrupt())
        self.assertEqual(flushes, 1)
        self.assertFalse(await controller.close_if_idle(now=start + timedelta(seconds=5)))
        await controller.feed(b"nois", captured_at=start + timedelta(seconds=7))
        self.assertTrue(await controller.close_if_idle(now=start + timedelta(seconds=7)))
        self.assertEqual(stopped, [SESSION_ID])
        self.assertEqual(controller.phase, "idle")
        self.assertIsNone(controller.session_id)

    async def test_hard_session_limit_closes_without_more_audio_frames(self) -> None:
        stopped: list[UUID] = []

        async def no_op(*_args) -> None:
            return None

        async def start_session(_frames) -> UUID:
            return SESSION_ID

        async def stop_session(session_id: UUID) -> None:
            stopped.append(session_id)

        controller = LocalSessionController(
            pre_roll=PreRollBuffer(frame_bytes=4),
            wake_detect=lambda _payload: _resolved(True),
            start_session=start_session,
            stop_session=stop_session,
            send_audio=no_op,
            flush_playback=no_op,
            speech_detect=lambda _payload: True,
            max_session_seconds=0.01,
        )

        self.assertTrue(await controller.feed(b"wake"))
        await asyncio.sleep(0.05)

        self.assertEqual(stopped, [SESSION_ID])
        self.assertEqual(controller.phase, "idle")
        self.assertIsNone(controller.session_id)

    async def test_vad_speech_resets_idle_timeout_but_noise_does_not(self) -> None:
        start = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
        stopped: list[UUID] = []

        async def no_op(*_args) -> None:
            return None

        async def start_session(_frames) -> UUID:
            return SESSION_ID

        async def stop_session(session_id: UUID) -> None:
            stopped.append(session_id)

        controller = LocalSessionController(
            pre_roll=PreRollBuffer(frame_bytes=4),
            wake_detect=lambda _payload: _resolved(True),
            start_session=start_session,
            stop_session=stop_session,
            send_audio=no_op,
            flush_playback=no_op,
            speech_detect=lambda payload: payload == b"talk",
            idle_timeout_seconds=5,
            speech_updates_idle=True,
        )

        await controller.feed(b"wake", captured_at=start)
        await controller.feed(b"talk", captured_at=start + timedelta(seconds=4))
        await controller.feed(b"nois", captured_at=start + timedelta(seconds=8))

        self.assertFalse(
            await controller.close_if_idle(now=start + timedelta(seconds=8.9))
        )
        self.assertTrue(
            await controller.close_if_idle(now=start + timedelta(seconds=9.1))
        )
        self.assertEqual(stopped, [SESSION_ID])

    async def test_nonpositive_hard_session_limit_is_rejected(self) -> None:
        async def no_op(*_args) -> None:
            return None

        with self.assertRaises(ValueError):
            LocalSessionController(
                pre_roll=PreRollBuffer(frame_bytes=4),
                wake_detect=lambda _payload: _resolved(False),
                start_session=no_op,
                stop_session=no_op,
                send_audio=no_op,
                flush_playback=no_op,
                speech_detect=lambda _payload: False,
                max_session_seconds=0,
            )

    async def test_failed_start_clears_audio_and_returns_to_idle(self) -> None:
        async def start_session(_frames) -> UUID:
            raise ConnectionError("service unavailable")

        async def no_op(*_args) -> None:
            return None

        controller = LocalSessionController(
            pre_roll=PreRollBuffer(frame_bytes=4, capacity_frames=3),
            wake_detect=lambda payload: _resolved(payload == b"wake"),
            start_session=start_session,
            stop_session=no_op,
            send_audio=no_op,
            flush_playback=no_op,
            speech_detect=lambda _payload: False,
        )
        await controller.feed(b"pre!", captured_at=datetime.now(timezone.utc))
        with self.assertRaises(ConnectionError):
            await controller.feed(b"wake", captured_at=datetime.now(timezone.utc))
        self.assertEqual(controller.phase, "idle")
        self.assertIsNone(controller.session_id)
        self.assertEqual(controller._pre_roll.snapshot(), ())


async def _resolved(value: bool) -> bool:
    return value


if __name__ == "__main__":
    unittest.main()
