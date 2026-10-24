from __future__ import annotations

import asyncio
import unittest
from datetime import datetime
from uuid import UUID

from device.audio_buffer import AudioFrameBuffer


SESSION_A = UUID("00000000-0000-4000-8000-000000000001")
SESSION_B = UUID("00000000-0000-4000-8000-000000000002")


class AudioFrameBufferTests(unittest.IsolatedAsyncioTestCase):
    async def test_frame_has_contract_metadata_and_exact_payload(self) -> None:
        buffer = AudioFrameBuffer(capacity_frames=2)
        await buffer.start_session(SESSION_A)
        payload = bytes(buffer.frame_bytes)

        accepted = await buffer.put(SESSION_A, payload)
        frame = await buffer.get()

        self.assertTrue(accepted)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.payload, payload)
        self.assertEqual(frame.metadata.session_id, SESSION_A)
        self.assertEqual(frame.metadata.sequence, 0)
        self.assertEqual(frame.metadata.encoding, "pcm_s16le")
        self.assertEqual(frame.metadata.frame_duration_ms, 20)
        self.assertIsNotNone(frame.metadata.captured_at.utcoffset())

    async def test_overflow_drops_oldest_and_preserves_sequence_gap(self) -> None:
        buffer = AudioFrameBuffer(capacity_frames=2)
        await buffer.start_session(SESSION_A)

        for value in (1, 2, 3):
            await buffer.put(SESSION_A, bytes([value]) * buffer.frame_bytes)

        stats = await buffer.stats()
        first = await buffer.get()
        second = await buffer.get()

        self.assertEqual(stats.queued_frames, 2)
        self.assertEqual(stats.queued_milliseconds, 40)
        self.assertEqual(stats.dropped_overflow_frames, 1)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertEqual((first.metadata.sequence, second.metadata.sequence), (1, 2))
        self.assertEqual(first.payload[0], 2)
        self.assertEqual(second.payload[0], 3)

    async def test_put_wait_backpressures_until_consumer_frees_capacity(self) -> None:
        buffer = AudioFrameBuffer(capacity_frames=1)
        await buffer.start_session(SESSION_A)
        first_payload = bytes([1]) * buffer.frame_bytes
        second_payload = bytes([2]) * buffer.frame_bytes
        await buffer.put(SESSION_A, first_payload)

        waiting_put = asyncio.create_task(buffer.put_wait(SESSION_A, second_payload))
        await asyncio.sleep(0)
        self.assertFalse(waiting_put.done())

        first = await buffer.get()
        self.assertTrue(await asyncio.wait_for(waiting_put, timeout=1))
        second = await buffer.get()
        stats = await buffer.stats()

        self.assertEqual(first.payload, first_payload)
        self.assertEqual(second.payload, second_payload)
        self.assertEqual((first.metadata.sequence, second.metadata.sequence), (0, 1))
        self.assertEqual(stats.dropped_overflow_frames, 0)

    async def test_put_wait_unblocks_when_session_closes(self) -> None:
        buffer = AudioFrameBuffer(capacity_frames=1)
        await buffer.start_session(SESSION_A)
        await buffer.put(SESSION_A, bytes([1]) * buffer.frame_bytes)
        waiting_put = asyncio.create_task(
            buffer.put_wait(SESSION_A, bytes([2]) * buffer.frame_bytes)
        )
        await asyncio.sleep(0)

        await buffer.close_session(SESSION_A)

        self.assertFalse(await asyncio.wait_for(waiting_put, timeout=1))

    async def test_put_wait_does_not_resurrect_frames_after_flush(self) -> None:
        buffer = AudioFrameBuffer(capacity_frames=1)
        await buffer.start_session(SESSION_A)
        await buffer.put(SESSION_A, bytes([1]) * buffer.frame_bytes)
        waiting_put = asyncio.create_task(
            buffer.put_wait(SESSION_A, bytes([2]) * buffer.frame_bytes)
        )
        await asyncio.sleep(0)

        self.assertTrue(await buffer.flush_session(SESSION_A))

        self.assertFalse(await asyncio.wait_for(waiting_put, timeout=1))
        self.assertEqual((await buffer.stats()).queued_frames, 0)

    async def test_new_session_flushes_old_frames_and_rejects_late_audio(self) -> None:
        buffer = AudioFrameBuffer(capacity_frames=2)
        await buffer.start_session(SESSION_A)
        await buffer.put(SESSION_A, bytes(buffer.frame_bytes))

        await buffer.start_session(SESSION_B)
        accepted_old = await buffer.put(SESSION_A, bytes(buffer.frame_bytes))
        accepted_new = await buffer.put(SESSION_B, bytes(buffer.frame_bytes))
        frame = await buffer.get()
        stats = await buffer.stats()

        self.assertFalse(accepted_old)
        self.assertTrue(accepted_new)
        self.assertEqual(stats.discarded_on_session_change, 1)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.metadata.session_id, SESSION_B)
        self.assertEqual(frame.metadata.sequence, 0)

    async def test_close_wakes_waiter_and_rejects_following_frames(self) -> None:
        buffer = AudioFrameBuffer()
        await buffer.start_session(SESSION_A)
        waiter = asyncio.create_task(buffer.get())
        await asyncio.sleep(0)

        closed = await buffer.close_session(SESSION_A)

        self.assertTrue(closed)
        self.assertIsNone(await waiter)
        self.assertFalse(await buffer.put(SESSION_A, bytes(buffer.frame_bytes)))

    async def test_stale_close_cannot_close_new_session(self) -> None:
        buffer = AudioFrameBuffer()
        await buffer.start_session(SESSION_A)
        await buffer.start_session(SESSION_B)

        closed = await buffer.close_session(SESSION_A)
        accepted = await buffer.put(SESSION_B, bytes(buffer.frame_bytes))

        self.assertFalse(closed)
        self.assertTrue(accepted)

    async def test_flush_drops_queued_frames_and_invalidates_dequeued_frame(self) -> None:
        buffer = AudioFrameBuffer(capacity_frames=2)
        await buffer.start_session(SESSION_A)
        await buffer.put(SESSION_A, bytes([1]) * buffer.frame_bytes)
        in_flight = await buffer.get()
        await buffer.put(SESSION_A, bytes([2]) * buffer.frame_bytes)
        await buffer.put(SESSION_A, bytes([3]) * buffer.frame_bytes)

        self.assertTrue(await buffer.flush_session(SESSION_A))
        self.assertFalse(await buffer.is_current(in_flight))
        stats = await buffer.stats()
        self.assertEqual(stats.queued_frames, 0)
        self.assertEqual(stats.discarded_on_flush_frames, 2)
        self.assertFalse(await buffer.flush_session(SESSION_B))

        self.assertTrue(await buffer.put(SESSION_A, bytes([4]) * buffer.frame_bytes))
        fresh = await buffer.get()
        self.assertIsNotNone(fresh)
        assert fresh is not None
        self.assertTrue(await buffer.is_current(fresh))
        self.assertEqual(fresh.metadata.sequence, 3)

    async def test_rejects_bad_frame_size_naive_time_and_excess_buffer(self) -> None:
        buffer = AudioFrameBuffer(capacity_frames=2)
        await buffer.start_session(SESSION_A)

        with self.assertRaises(ValueError):
            await buffer.put(SESSION_A, b"short")
        with self.assertRaises(ValueError):
            await buffer.put(
                SESSION_A,
                bytes(buffer.frame_bytes),
                captured_at=datetime(2026, 10, 6),
            )
        with self.assertRaises(ValueError):
            AudioFrameBuffer(frame_duration_ms=100, capacity_frames=6)


if __name__ == "__main__":
    unittest.main()
