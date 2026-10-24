from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import UUID

from contracts.models import TimerFiredEvent
from device.alsa import PcmFormat
from device.timer_chime import AlsaTimerChime, timer_chime_frames


EVENT_ID = UUID("00000000-0000-4000-8000-000000000501")


class FakePlayer:
    def __init__(self, *, fail_write: bool = False) -> None:
        self.fail_write = fail_write
        self.started = False
        self.closed = False
        self.aborted = False
        self.frames: list[bytes] = []

    async def start(self) -> None:
        self.started = True

    async def write_frame(self, payload: bytes) -> None:
        if self.fail_write:
            raise OSError("device unavailable")
        self.frames.append(payload)

    async def drain_and_close(self) -> None:
        self.closed = True

    async def abort(self) -> None:
        self.aborted = True


def timer_event() -> TimerFiredEvent:
    now = datetime.now(timezone.utc)
    return TimerFiredEvent(
        event_id=EVENT_ID,
        timer_id=EVENT_ID,
        device_id="cube-01",
        due_at=now,
        fired_at=now,
        message="Tea is ready",
        attempt_count=1,
    )


class TimerChimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_chime_uses_short_bounded_pcm_frames_and_drains(self) -> None:
        player = FakePlayer()
        chime = AlsaTimerChime(
            "plughw:CARD=seeed2micvoicec,DEV=0",
            player_factory=lambda _pcm, _format: player,
        )

        await chime(timer_event())

        self.assertTrue(player.started)
        self.assertTrue(player.closed)
        self.assertFalse(player.aborted)
        self.assertEqual(len(player.frames), 18)
        self.assertTrue(all(len(frame) == 960 for frame in player.frames))
        silence_frames = (*player.frames[4:6], *player.frames[10:12])
        self.assertTrue(all(not any(frame) for frame in silence_frames))
        samples = [
            int.from_bytes(frame[offset : offset + 2], "little", signed=True)
            for frame in player.frames
            for offset in range(0, len(frame), 2)
        ]
        self.assertLessEqual(max(samples), 1_639)
        self.assertGreaterEqual(min(samples), -1_639)

    async def test_playback_failure_aborts_player(self) -> None:
        player = FakePlayer(fail_write=True)
        chime = AlsaTimerChime("hw:2,0", player_factory=lambda _pcm, _format: player)

        with self.assertRaisesRegex(OSError, "device unavailable"):
            await chime(timer_event())

        self.assertTrue(player.aborted)
        self.assertFalse(player.closed)

    async def test_chime_rejects_excessive_amplitude(self) -> None:
        with self.assertRaises(ValueError):
            AlsaTimerChime("hw:2,0", amplitude=0.11)

        with self.assertRaises(ValueError):
            timer_chime_frames(PcmFormat(), amplitude=0.11)


if __name__ == "__main__":
    unittest.main()
