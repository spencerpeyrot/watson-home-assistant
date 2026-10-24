from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import UUID

from device.alsa import AudioDeviceError
from device.audio_buffer import BufferedAudioFrame
from device.audio_runtime import PiAudioRuntime


SESSION = UUID("00000000-0000-4000-8000-000000000010")


class FakeCapture:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.started = False
        self.closed = False
        self.sent = False

    async def start(self) -> None:
        self.started = True

    async def read_frame(self) -> bytes:
        if not self.sent:
            self.sent = True
            return self.payload
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def close(self) -> None:
        self.closed = True


class FakePlayback:
    def __init__(self) -> None:
        self.started = False
        self.aborted = False
        self.start_count = 0
        self.abort_count = 0
        self.drain_count = 0
        self.frames: list[bytes] = []
        self.wrote_frame = asyncio.Event()

    async def start(self) -> None:
        self.started = True
        self.start_count += 1

    async def write_frame(self, payload: bytes) -> None:
        self.frames.append(payload)
        self.wrote_frame.set()

    async def abort(self) -> None:
        self.aborted = True
        self.abort_count += 1

    async def drain_and_close(self) -> None:
        self.drain_count += 1


class BlockingPlayback(FakePlayback):
    def __init__(self) -> None:
        super().__init__()
        self.writing = asyncio.Event()
        self.release_write = asyncio.Event()

    async def write_frame(self, payload: bytes) -> None:
        if not self.frames:
            self.writing.set()
            await self.release_write.wait()
        await super().write_frame(payload)


class FailingCapture(FakeCapture):
    async def read_frame(self) -> bytes:
        raise OSError("simulated device failure")


class PiAudioRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_playback_workers_start_and_stop(self) -> None:
        payload = bytes(960)
        capture = FakeCapture(payload)
        playback = FakePlayback()
        runtime = PiAudioRuntime(
            capture_pcm="unused-capture",
            playback_pcm="unused-playback",
            capture_stream=capture,
            playback_stream=playback,
        )

        await runtime.start(SESSION)
        captured = await asyncio.wait_for(runtime.next_capture_frame(), timeout=1)
        queued = await runtime.queue_playback_frame(SESSION, payload)
        await asyncio.wait_for(playback.wrote_frame.wait(), timeout=1)
        stopped = await runtime.stop(SESSION)

        self.assertTrue(capture.started)
        self.assertTrue(playback.started)
        self.assertIsNotNone(captured)
        assert captured is not None
        self.assertEqual(captured.payload, payload)
        self.assertTrue(queued)
        self.assertEqual(playback.frames, [payload])
        self.assertTrue(stopped)
        self.assertTrue(capture.closed)
        self.assertTrue(playback.aborted)
        self.assertIsNone(runtime.session_id)
        self.assertFalse(await runtime.queue_playback_frame(SESSION, payload))

    async def test_playback_queue_backpressures_without_dropping_speech_frames(self) -> None:
        capture = FakeCapture(bytes(960))
        playback = BlockingPlayback()
        runtime = PiAudioRuntime(
            capture_pcm="unused-capture",
            playback_pcm="unused-playback",
            capacity_frames=2,
            capture_stream=capture,
            playback_stream=playback,
        )
        payloads = [bytes([value]) * 960 for value in (1, 2, 3, 4)]
        await runtime.start(SESSION)
        await runtime.queue_playback_frame(SESSION, payloads[0])
        await asyncio.wait_for(playback.writing.wait(), timeout=1)
        await runtime.queue_playback_frame(SESSION, payloads[1])
        await runtime.queue_playback_frame(SESSION, payloads[2])

        blocked_put = asyncio.create_task(
            runtime.queue_playback_frame(SESSION, payloads[3])
        )
        await asyncio.sleep(0)
        self.assertFalse(blocked_put.done())

        playback.release_write.set()
        self.assertTrue(await asyncio.wait_for(blocked_put, timeout=1))
        for _ in range(100):
            if len(playback.frames) == len(payloads):
                break
            await asyncio.sleep(0.001)
        stats = await runtime.stats()
        await runtime.stop(SESSION)

        self.assertEqual(playback.frames, payloads)
        self.assertEqual(stats[1].dropped_overflow_frames, 0)

    async def test_playback_drain_waits_for_speaker_worker_before_closing_stream(self) -> None:
        capture = FakeCapture(bytes(960))
        playback = BlockingPlayback()
        runtime = PiAudioRuntime(
            capture_pcm="unused-capture",
            playback_pcm="unused-playback",
            capture_stream=capture,
            playback_stream=playback,
        )
        await runtime.start(SESSION)
        await runtime.queue_playback_frame(SESSION, bytes([4]) * 960)
        await asyncio.wait_for(playback.writing.wait(), timeout=1)

        drain = asyncio.create_task(runtime.drain_playback())
        await asyncio.sleep(0)
        self.assertFalse(drain.done())
        self.assertEqual(playback.drain_count, 0)

        playback.release_write.set()
        await asyncio.wait_for(drain, timeout=1)
        await runtime.stop(SESSION)

        self.assertEqual(playback.drain_count, 1)

    async def test_playback_drain_reopens_stream_for_followup_turn(self) -> None:
        playback = FakePlayback()
        runtime = PiAudioRuntime(
            capture_pcm="unused-capture",
            playback_pcm="unused-playback",
            capture_stream=FakeCapture(bytes(960)),
            playback_stream=playback,
        )
        await runtime.start(SESSION)

        await runtime.drain_playback(reopen=True)
        started_count = playback.start_count
        queued = await runtime.queue_playback_frame(SESSION, bytes([5]) * 960)
        await asyncio.wait_for(playback.wrote_frame.wait(), timeout=1)
        await runtime.stop(SESSION)

        self.assertTrue(queued)
        self.assertEqual(playback.drain_count, 1)
        self.assertEqual(started_count, 2)

    async def test_playback_queue_drain_has_a_deadline(self) -> None:
        runtime = PiAudioRuntime(
            capture_pcm="unused-capture",
            playback_pcm="unused-playback",
            capture_stream=FakeCapture(bytes(960)),
            playback_stream=FakePlayback(),
        )
        await runtime.start(SESSION)
        runtime._playback_drained.clear()
        with patch("device.audio_runtime.DEFAULT_PLAYBACK_DRAIN_TIMEOUT_SECONDS", 0.001):
            with self.assertRaisesRegex(AudioDeviceError, "playback queue drain timed out"):
                await runtime.drain_playback()
        self.assertIsNone(runtime.session_id)
        await runtime.stop()

    async def test_capture_preroll_is_first_audio_seen_by_provider_worker(self) -> None:
        capture = FakeCapture(bytes([3]) * 960)
        playback = FakePlayback()
        runtime = PiAudioRuntime(
            capture_pcm="unused-capture",
            playback_pcm="unused-playback",
            capture_stream=capture,
            playback_stream=playback,
        )
        start = datetime.now(timezone.utc)
        preroll = (
            BufferedAudioFrame(bytes([1]) * 960, start),
            BufferedAudioFrame(bytes([2]) * 960, start + timedelta(milliseconds=20)),
        )

        await runtime.start(SESSION, capture_preroll=preroll)
        captured = [await asyncio.wait_for(runtime.next_capture_frame(), timeout=1) for _ in range(3)]
        await runtime.stop()

        self.assertEqual([frame.payload[0] for frame in captured if frame is not None], [1, 2, 3])
        self.assertEqual([frame.metadata.sequence for frame in captured if frame is not None], [0, 1, 2])

    async def test_flush_playback_restarts_pcm_and_discards_stale_audio(self) -> None:
        capture = FakeCapture(bytes(960))
        playback = BlockingPlayback()
        runtime = PiAudioRuntime(
            capture_pcm="unused-capture",
            playback_pcm="unused-playback",
            capture_stream=capture,
            playback_stream=playback,
        )
        await runtime.start(SESSION)
        first = bytes([1]) * 960
        stale = bytes([2]) * 960
        await runtime.queue_playback_frame(SESSION, first)
        await asyncio.wait_for(playback.writing.wait(), timeout=1)
        await runtime.queue_playback_frame(SESSION, stale)

        flush = asyncio.create_task(runtime.flush_playback())
        for _ in range(100):
            _, playback_stats = await runtime.stats()
            if playback_stats.discarded_on_flush_frames == 1:
                break
            await asyncio.sleep(0.001)
        playback.release_write.set()
        flushed = await flush

        self.assertTrue(flushed)
        self.assertEqual(playback.frames, [first])
        self.assertEqual(playback.abort_count, 1)
        self.assertEqual(playback.start_count, 2)
        stats = await runtime.stats()
        self.assertEqual(stats[1].discarded_on_flush_frames, 1)
        self.assertTrue(await runtime.stop(SESSION))

    async def test_worker_failure_is_signaled_and_stop_closes_other_stream(self) -> None:
        capture = FailingCapture(bytes(960))
        playback = FakePlayback()
        runtime = PiAudioRuntime(
            capture_pcm="unused-capture",
            playback_pcm="unused-playback",
            capture_stream=capture,
            playback_stream=playback,
        )

        await runtime.start(SESSION)
        failure = await asyncio.wait_for(runtime.wait_failure(), timeout=1)
        stopped = await runtime.stop(SESSION)

        self.assertEqual(failure, "capture stream failed (OSError)")
        self.assertTrue(stopped)
        self.assertTrue(capture.closed)
        self.assertTrue(playback.aborted)
        self.assertIsNone(runtime._capture_task)
        self.assertIsNone(runtime._playback_task)


if __name__ == "__main__":
    unittest.main()
