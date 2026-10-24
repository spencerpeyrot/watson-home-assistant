"""Session-bound capture/playback workers for the Pi audio device."""

from __future__ import annotations

import asyncio
from math import sqrt
import sys
from time import monotonic
from array import array
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Protocol
from uuid import UUID

from device.alsa import AlsaCapture, AlsaPlayback, AudioDeviceError, PcmFormat
from device.audio_buffer import (
    AudioFrame,
    AudioFrameBuffer,
    BufferedAudioFrame,
    BufferStats,
)

_OUTPUT_ACTIVITY_RMS = 64 / 32768


def _normalized_output_level(payload: bytes) -> float:
    """Return PCM16 RMS from 0.0 to 1.0 without retaining audio."""

    samples = array("h")
    samples.frombytes(payload)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0
    return sqrt(sum(sample * sample for sample in samples) / len(samples)) / 32768


class CaptureStream(Protocol):
    async def start(self) -> None: ...

    async def read_frame(self) -> bytes: ...

    async def close(self) -> None: ...


class PlaybackStream(Protocol):
    async def start(self) -> None: ...

    async def write_frame(self, payload: bytes) -> None: ...

    async def drain_and_close(self) -> None: ...

    async def abort(self) -> None: ...


class PiAudioRuntime:
    """Run bounded PCM capture and playback workers for one session."""

    def __init__(
        self,
        *,
        capture_pcm: str,
        playback_pcm: str,
        audio_format: PcmFormat = PcmFormat(),
        capacity_frames: int = 10,
        capture_stream: CaptureStream | None = None,
        playback_stream: PlaybackStream | None = None,
    ) -> None:
        self.audio_format = audio_format
        buffer_options = {
            "sample_rate_hz": audio_format.sample_rate_hz,
            "channels": audio_format.channels,
            "frame_duration_ms": audio_format.frame_duration_ms,
            "capacity_frames": capacity_frames,
        }
        self.capture_buffer = AudioFrameBuffer(**buffer_options)
        self.playback_buffer = AudioFrameBuffer(**buffer_options)
        self._capture = (
            capture_stream
            if capture_stream is not None
            else AlsaCapture(capture_pcm, audio_format)
        )
        self._playback = (
            playback_stream
            if playback_stream is not None
            else AlsaPlayback(playback_pcm, audio_format)
        )
        self._session_id: UUID | None = None
        self._capture_task: asyncio.Task[None] | None = None
        self._playback_task: asyncio.Task[None] | None = None
        self._playback_lock = asyncio.Lock()
        self._playback_drained = asyncio.Event()
        self._playback_drained.set()
        self._failure: str | None = None
        self._failure_event = asyncio.Event()
        self._last_playback_activity: datetime | None = None
        self._last_playback_level = 0.0
        self._last_playback_level_at: float | None = None
        self._preroll_remaining = 0
        self._preroll_drained = asyncio.Event()
        self._preroll_drained.set()

    @property
    def session_id(self) -> UUID | None:
        return self._session_id

    @property
    def failure(self) -> str | None:
        """Sanitized first worker failure, if a stream exited unexpectedly."""

        return self._failure

    @property
    def last_playback_activity(self) -> datetime | None:
        """Last time a playback frame was written to the ALSA stream."""

        return self._last_playback_activity

    @property
    def playback_level(self) -> float:
        """Fresh normalized PCM RMS, or zero after playback frames stop."""

        if (
            self._last_playback_level_at is None
            or monotonic() - self._last_playback_level_at > 0.12
        ):
            return 0.0
        return self._last_playback_level

    async def wait_failure(self) -> str:
        """Wait until a capture or playback worker fails; return safe detail."""

        await self._failure_event.wait()
        return self._failure or "audio worker failed"

    def adopt_capture(self, capture: CaptureStream) -> None:
        """Take ownership of an idempotently started continuous mic stream."""

        if self._session_id is not None or self._capture_task is not None:
            raise AudioDeviceError("cannot replace capture during an active session")
        self._capture = capture

    async def start(
        self,
        session_id: UUID,
        *,
        capture_preroll: Sequence[BufferedAudioFrame] = (),
    ) -> None:
        """Open HAT streams and start fresh session-scoped workers."""

        if self._session_id is not None:
            raise AudioDeviceError("audio runtime already has an active session")
        if self._capture_task is not None or self._playback_task is not None:
            await self.stop()
        if len(capture_preroll) > self.capture_buffer.capacity_frames:
            raise ValueError("capture pre-roll exceeds the configured audio queue")
        self._failure = None
        self._failure_event.clear()
        self._last_playback_activity = None
        self._last_playback_level = 0.0
        self._last_playback_level_at = None
        self._playback_drained.set()
        self._preroll_remaining = len(capture_preroll)
        if self._preroll_remaining:
            self._preroll_drained.clear()
        else:
            self._preroll_drained.set()
        await self.capture_buffer.start_session(session_id)
        await self.playback_buffer.start_session(session_id)
        try:
            for frame in capture_preroll:
                accepted = await self.capture_buffer.put(
                    session_id, frame.payload, captured_at=frame.captured_at
                )
                if not accepted:
                    raise AudioDeviceError("capture pre-roll could not enter the active session")
            await self._capture.start()
            await self._playback.start()
        except BaseException:
            await self.capture_buffer.close_session(session_id)
            await self.playback_buffer.close_session(session_id)
            await self._capture.close()
            await self._playback.abort()
            raise
        self._session_id = session_id
        self._capture_task = asyncio.create_task(
            self._capture_worker(session_id), name="cube-audio-capture"
        )
        self._playback_task = asyncio.create_task(
            self._playback_worker(session_id), name="cube-audio-playback"
        )

    async def next_capture_frame(self) -> AudioFrame | None:
        """Return next mic frame for the voice transport, or None after close."""

        frame = await self.capture_buffer.get()
        if frame is not None and self._preroll_remaining:
            self._preroll_remaining -= 1
            if self._preroll_remaining == 0:
                self._preroll_drained.set()
        return frame

    async def queue_playback_frame(self, session_id: UUID, payload: bytes) -> bool:
        """Queue output losslessly, applying backpressure when speaker is behind."""

        accepted = await self.playback_buffer.put_wait(session_id, payload)
        if accepted:
            self._playback_drained.clear()
        return accepted

    async def drain_playback(self, *, reopen: bool = False) -> None:
        """Drain queued PCM through ALSA, optionally reopening for the next turn."""

        await self._playback_drained.wait()
        drain = getattr(self._playback, "drain_and_close", None)
        if drain is None:
            return
        try:
            async with self._playback_lock:
                await drain()
                if reopen and self._session_id is not None:
                    await self._playback.start()
        except Exception as error:
            session_id = self._session_id
            if session_id is not None:
                await self._fail_session(session_id, "playback", error)
            raise AudioDeviceError(
                f"playback drain failed ({type(error).__name__})"
            ) from None

    async def flush_playback(self) -> bool:
        """Stop queued/current speaker audio and reopen the active PCM stream."""

        session_id = self._session_id
        if session_id is None:
            return False
        if not await self.playback_buffer.flush_session(session_id):
            return False
        try:
            async with self._playback_lock:
                await self._playback.abort()
                if self._session_id != session_id:
                    return False
                await self._playback.start()
            playback_stats = await self.playback_buffer.stats()
            if playback_stats.queued_frames == 0:
                self._playback_drained.set()
        except Exception as error:
            await self._fail_session(session_id, "playback", error)
            raise AudioDeviceError(
                f"playback flush failed ({type(error).__name__})"
            ) from None
        return True

    async def stats(self) -> tuple[BufferStats, BufferStats]:
        """Return capture and playback queue counters."""

        return await self.capture_buffer.stats(), await self.playback_buffer.stats()

    async def stop(self, session_id: UUID | None = None) -> bool:
        """Flush audio, cancel workers, and close streams; safe to repeat."""

        active_id = self._session_id
        has_workers = self._capture_task is not None or self._playback_task is not None
        if active_id is None and not has_workers:
            return False
        if active_id is not None and session_id is not None and session_id != active_id:
            return False
        self._session_id = None
        self._last_playback_activity = None
        self._last_playback_level = 0.0
        self._last_playback_level_at = None
        if active_id is not None:
            await self.capture_buffer.close_session(active_id)
            await self.playback_buffer.close_session(active_id)
        tasks = [task for task in (self._capture_task, self._playback_task) if task]
        self._capture_task = None
        self._playback_task = None
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._capture.close()
        await self._playback.abort()
        return True

    async def _capture_worker(self, session_id: UUID) -> None:
        try:
            # A continuous startup stream can already have PCM ready. Let the
            # relay consume pre-roll first so its sequence zero cannot overflow.
            await self._preroll_drained.wait()
            while self._session_id == session_id:
                frame = await self._capture.read_frame()
                await self.capture_buffer.put(session_id, frame)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._fail_session(session_id, "capture", error)
        finally:
            await self._capture.close()

    async def _playback_worker(self, session_id: UUID) -> None:
        try:
            while self._session_id == session_id:
                frame = await self.playback_buffer.get()
                if frame is None:
                    return
                async with self._playback_lock:
                    if not await self.playback_buffer.is_current(frame):
                        continue
                    await self._playback.write_frame(frame.payload)
                    self._last_playback_level_at = monotonic()
                    self._last_playback_level = _normalized_output_level(frame.payload)
                    if self._last_playback_level >= _OUTPUT_ACTIVITY_RMS:
                        self._last_playback_activity = datetime.now(timezone.utc)
                    playback_stats = await self.playback_buffer.stats()
                    if playback_stats.queued_frames == 0:
                        self._playback_drained.set()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._fail_session(session_id, "playback", error)
        finally:
            await self._playback.abort()

    async def _fail_session(self, session_id: UUID, stream: str, error: Exception) -> None:
        if self._session_id != session_id:
            return
        self._failure = f"{stream} stream failed ({type(error).__name__})"
        self._failure_event.set()
        self._session_id = None
        self._last_playback_activity = None
        self._last_playback_level = 0.0
        self._last_playback_level_at = None
        await self.capture_buffer.close_session(session_id)
        await self.playback_buffer.close_session(session_id)
