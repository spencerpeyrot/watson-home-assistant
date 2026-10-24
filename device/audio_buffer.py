"""Bounded, session-scoped PCM frame buffer for the Pi audio runtime."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from contracts.models import AudioFrameMetadata


@dataclass(frozen=True)
class AudioFrame:
    """One in-memory PCM frame and the buffer generation that owns it."""

    metadata: AudioFrameMetadata
    payload: bytes
    generation: int


@dataclass(frozen=True)
class BufferedAudioFrame:
    """PCM frame held in memory before a provider session exists."""

    payload: bytes
    captured_at: datetime


@dataclass(frozen=True)
class BufferStats:
    """Current queue size and cumulative loss counters."""

    queued_frames: int
    queued_milliseconds: int
    dropped_overflow_frames: int
    discarded_on_session_change: int
    discarded_on_flush_frames: int


class AudioFrameBuffer:
    """Bounded low-latency PCM queue; a new session flushes stale frames."""

    MAX_BUFFERED_MILLISECONDS = 500

    def __init__(
        self,
        *,
        sample_rate_hz: int = 24_000,
        channels: int = 1,
        frame_duration_ms: int = 20,
        capacity_frames: int = 10,
    ) -> None:
        if sample_rate_hz <= 0 or channels <= 0 or frame_duration_ms <= 0:
            raise ValueError("audio format values must be positive")
        if frame_duration_ms > 100:
            raise ValueError("frame duration must not exceed 100 ms")
        if capacity_frames <= 0:
            raise ValueError("capacity_frames must be positive")
        if capacity_frames * frame_duration_ms > self.MAX_BUFFERED_MILLISECONDS:
            raise ValueError(
                f"buffer capacity must not exceed {self.MAX_BUFFERED_MILLISECONDS} ms"
            )
        byte_numerator = sample_rate_hz * channels * 2 * frame_duration_ms
        if byte_numerator % 1000:
            raise ValueError("frame duration must align to whole PCM samples")

        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        self.capacity_frames = capacity_frames
        self.frame_bytes = byte_numerator // 1000
        self._frames: deque[AudioFrame] = deque()
        self._condition = asyncio.Condition()
        self._session_id: UUID | None = None
        self._generation = 0
        self._next_sequence = 0
        self._closed = True
        self._dropped_overflow_frames = 0
        self._discarded_on_session_change = 0
        self._discarded_on_flush_frames = 0

    async def start_session(self, session_id: UUID) -> int:
        """Flush prior audio and start a fresh sequence/generation."""

        async with self._condition:
            self._discarded_on_session_change += len(self._frames)
            self._frames.clear()
            self._generation += 1
            self._session_id = session_id
            self._next_sequence = 0
            self._closed = False
            self._condition.notify_all()
            return self._generation

    async def close_session(self, session_id: UUID | None = None) -> bool:
        """Close the active session and discard queued audio."""

        async with self._condition:
            if session_id is not None and session_id != self._session_id:
                return False
            self._discarded_on_session_change += len(self._frames)
            self._frames.clear()
            self._session_id = None
            self._generation += 1
            self._closed = True
            self._condition.notify_all()
            return True

    async def flush_session(self, session_id: UUID) -> bool:
        """Discard queued frames and invalidate any frame already dequeued."""

        async with self._condition:
            if self._closed or session_id != self._session_id:
                return False
            self._discarded_on_flush_frames += len(self._frames)
            self._frames.clear()
            self._generation += 1
            self._condition.notify_all()
            return True

    async def put(
        self,
        session_id: UUID,
        payload: bytes | bytearray | memoryview,
        *,
        captured_at: datetime | None = None,
    ) -> bool:
        """Queue one exact-size frame; drop oldest frame on capacity overflow.

        Returns false for late audio from a closed or superseded session.
        """

        raw = bytes(payload)
        if len(raw) != self.frame_bytes:
            raise ValueError(f"PCM frame must contain exactly {self.frame_bytes} bytes")
        timestamp = captured_at or datetime.now(timezone.utc)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")

        async with self._condition:
            if self._closed or session_id != self._session_id:
                return False
            frame = AudioFrame(
                metadata=AudioFrameMetadata(
                    schema_version=1,
                    session_id=session_id,
                    sequence=self._next_sequence,
                    captured_at=timestamp,
                    encoding="pcm_s16le",
                    sample_rate_hz=self.sample_rate_hz,
                    channels=self.channels,
                    frame_duration_ms=self.frame_duration_ms,
                ),
                payload=raw,
                generation=self._generation,
            )
            self._next_sequence += 1
            if len(self._frames) == self.capacity_frames:
                self._frames.popleft()
                self._dropped_overflow_frames += 1
            self._frames.append(frame)
            self._condition.notify(1)
            return True

    async def put_wait(
        self,
        session_id: UUID,
        payload: bytes | bytearray | memoryview,
        *,
        captured_at: datetime | None = None,
    ) -> bool:
        """Queue a frame without dropping audio; wait for capacity if full.

        Returns false when the session closes or changes while waiting.
        """

        raw = bytes(payload)
        if len(raw) != self.frame_bytes:
            raise ValueError(f"PCM frame must contain exactly {self.frame_bytes} bytes")
        timestamp = captured_at or datetime.now(timezone.utc)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")

        async with self._condition:
            generation = self._generation
            while (
                len(self._frames) >= self.capacity_frames
                and not self._closed
                and session_id == self._session_id
            ):
                await self._condition.wait()
            if (
                self._closed
                or session_id != self._session_id
                or generation != self._generation
            ):
                return False
            frame = AudioFrame(
                metadata=AudioFrameMetadata(
                    schema_version=1,
                    session_id=session_id,
                    sequence=self._next_sequence,
                    captured_at=timestamp,
                    encoding="pcm_s16le",
                    sample_rate_hz=self.sample_rate_hz,
                    channels=self.channels,
                    frame_duration_ms=self.frame_duration_ms,
                ),
                payload=raw,
                generation=self._generation,
            )
            self._next_sequence += 1
            self._frames.append(frame)
            self._condition.notify(1)
            return True

    async def get(self) -> AudioFrame | None:
        """Wait for a frame; return None after the active session closes."""

        async with self._condition:
            while not self._frames and not self._closed:
                await self._condition.wait()
            if self._closed:
                return None
            frame = self._frames.popleft()
            self._condition.notify_all()
            if (
                frame.generation != self._generation
                or frame.metadata.session_id != self._session_id
            ):
                self._discarded_on_session_change += 1
                return None
            return frame

    async def is_current(self, frame: AudioFrame) -> bool:
        """Check frame generation immediately before transport or playback."""

        async with self._condition:
            return (
                not self._closed
                and frame.generation == self._generation
                and frame.metadata.session_id == self._session_id
            )

    async def stats(self) -> BufferStats:
        """Return a consistent snapshot without exposing queued audio."""

        async with self._condition:
            return BufferStats(
                queued_frames=len(self._frames),
                queued_milliseconds=len(self._frames) * self.frame_duration_ms,
                dropped_overflow_frames=self._dropped_overflow_frames,
                discarded_on_session_change=self._discarded_on_session_change,
                discarded_on_flush_frames=self._discarded_on_flush_frames,
            )
