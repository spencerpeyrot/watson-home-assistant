"""Local wake/session state machine with bounded in-memory audio pre-roll."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

from device.audio_buffer import BufferedAudioFrame


WakeDetector = Callable[[bytes], Awaitable[bool]]
StartSession = Callable[[tuple[BufferedAudioFrame, ...]], Awaitable[UUID]]
StopSession = Callable[[UUID], Awaitable[None]]
SendAudio = Callable[[UUID, BufferedAudioFrame], Awaitable[None]]
FlushPlayback = Callable[[], Awaitable[None]]
SpeechDetector = Callable[[bytes], bool]
MAX_CANDIDATE_SPEECH_SECONDS = 40.0
_LOGGER = logging.getLogger(__name__)


class PreRollBuffer:
    """Keep only the newest bounded PCM frames in memory."""

    def __init__(
        self,
        *,
        frame_bytes: int,
        frame_duration_ms: int = 20,
        capacity_frames: int = 10,
    ) -> None:
        if frame_bytes <= 0 or frame_duration_ms <= 0 or capacity_frames <= 0:
            raise ValueError("pre-roll format and capacity must be positive")
        if frame_duration_ms > 100 or frame_duration_ms * capacity_frames > 500:
            raise ValueError("pre-roll must not exceed 500 ms")
        self.frame_bytes = frame_bytes
        self.frame_duration_ms = frame_duration_ms
        self.capacity_frames = capacity_frames
        self._frames: deque[BufferedAudioFrame] = deque(maxlen=capacity_frames)

    @property
    def buffered_milliseconds(self) -> int:
        return len(self._frames) * self.frame_duration_ms

    def append(
        self, payload: bytes, *, captured_at: datetime | None = None
    ) -> None:
        raw = bytes(payload)
        if len(raw) != self.frame_bytes:
            raise ValueError(f"PCM frame must contain exactly {self.frame_bytes} bytes")
        timestamp = captured_at or datetime.now(timezone.utc)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        self._frames.append(BufferedAudioFrame(raw, timestamp))

    def snapshot(self) -> tuple[BufferedAudioFrame, ...]:
        return tuple(self._frames)

    def take(self) -> tuple[BufferedAudioFrame, ...]:
        frames = self.snapshot()
        self.clear()
        return frames

    def clear(self) -> None:
        self._frames.clear()


class LocalSessionController:
    """Route frames through local wake detection, session start, and idle close."""

    def __init__(
        self,
        *,
        pre_roll: PreRollBuffer,
        wake_detect: WakeDetector,
        start_session: StartSession,
        stop_session: StopSession,
        send_audio: SendAudio,
        flush_playback: FlushPlayback,
        speech_detect: SpeechDetector,
        idle_timeout_seconds: float = 3.0,
        max_session_seconds: float | None = None,
        speech_updates_idle: bool = True,
    ) -> None:
        if idle_timeout_seconds <= 0:
            raise ValueError("idle timeout must be positive")
        if max_session_seconds is not None and max_session_seconds <= 0:
            raise ValueError("maximum session duration must be positive")
        self._pre_roll = pre_roll
        self._wake_detect = wake_detect
        self._start_session = start_session
        self._stop_session = stop_session
        self._send_audio = send_audio
        self._flush_playback = flush_playback
        self._speech_detect = speech_detect
        self._speech_updates_idle = speech_updates_idle
        self._base_idle_timeout = idle_timeout_seconds
        self._idle_timeout = idle_timeout_seconds
        self._max_session_seconds = max_session_seconds
        self._lock = asyncio.Lock()
        self._phase = "idle"
        self._session_id: UUID | None = None
        self._last_activity: datetime | None = None
        self._last_response_activity: datetime | None = None
        self._speech_active = False
        self._speech_started_at: datetime | None = None
        self._duration_task: asyncio.Task[None] | None = None

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def session_id(self) -> UUID | None:
        return self._session_id

    @property
    def idle_timeout_seconds(self) -> float:
        return self._idle_timeout

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    @property
    def seconds_since_activity(self) -> float | None:
        """Expose idle timing without exposing microphone data."""

        if self._last_activity is None:
            return None
        return (datetime.now(timezone.utc) - self._last_activity).total_seconds()

    async def feed(
        self,
        payload: bytes,
        *,
        captured_at: datetime | None = None,
        suppress_speech: bool = False,
    ) -> bool:
        """Process one capture frame; return true when this frame starts a session."""

        frame = BufferedAudioFrame(
            bytes(payload), captured_at or datetime.now(timezone.utc)
        )
        if len(frame.payload) != self._pre_roll.frame_bytes:
            raise ValueError(
                f"PCM frame must contain exactly {self._pre_roll.frame_bytes} bytes"
            )
        if frame.captured_at.tzinfo is None or frame.captured_at.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        async with self._lock:
            if self._phase == "idle":
                self._pre_roll.append(frame.payload, captured_at=frame.captured_at)
                if not await self._wake_detect(frame.payload):
                    return False
                self._phase = "connecting"
                initial_audio = self._pre_roll.take()
                try:
                    session_id = await self._start_session(initial_audio)
                except BaseException:
                    self._phase = "idle"
                    self._session_id = None
                    self._last_activity = None
                    self._last_response_activity = None
                    self._speech_started_at = None
                    self._pre_roll.clear()
                    raise
                self._session_id = session_id
                self._last_activity = frame.captured_at
                self._last_response_activity = frame.captured_at
                self._speech_started_at = None
                self._idle_timeout = self._base_idle_timeout
                self._speech_active = False
                self._phase = "active"
                if self._max_session_seconds is not None:
                    self._duration_task = asyncio.create_task(
                        self._expire_session(session_id),
                        name="cube-session-duration-limit",
                    )
                return True
            if self._phase != "active" or self._session_id is None:
                return False
            was_speech_active = self._speech_active
            speech_active = not suppress_speech and self._speech_detect(frame.payload)
            if not suppress_speech and speech_active != was_speech_active:
                _LOGGER.info(
                    "voice local speech candidate %s session_id=%s",
                    "started" if speech_active else "ended",
                    self._session_id,
                )
            if speech_active and not self._speech_active:
                self._speech_started_at = frame.captured_at
            elif not speech_active:
                self._speech_started_at = None
            self._speech_active = speech_active
            candidate_is_bounded = (
                self._speech_started_at is None
                or frame.captured_at - self._speech_started_at
                <= timedelta(seconds=MAX_CANDIDATE_SPEECH_SECONDS)
            )
            if self._speech_active and self._speech_updates_idle and candidate_is_bounded:
                self._last_activity = frame.captured_at
            await self._send_audio(self._session_id, frame)
            return False

    async def interrupt(self) -> bool:
        """Flush current speaker output immediately after local interruption detection."""

        async with self._lock:
            if self._phase != "active":
                return False
            await self._flush_playback()
            return True

    def note_output_activity(self, when: datetime) -> None:
        """Keep a session alive while the speaker is actively playing a reply."""

        if when.tzinfo is None or when.utcoffset() is None:
            raise ValueError("playback activity time must be timezone-aware")
        if (
            self._phase == "active"
            and self._last_activity is not None
            and when > self._last_activity
        ):
            self._last_activity = when

    def note_input_activity(self) -> None:
        """Recognized speech or an explicit wake extends the listening window."""

        self.note_output_activity(datetime.now(timezone.utc))

    def note_response_activity(self, expectation: str | None = None) -> None:
        """Start the next listening window after speaker playback drains."""

        durations = {
            "none": self._base_idle_timeout,
            "open_answer": max(self._base_idle_timeout, 10.0),
            "confirmation": max(self._base_idle_timeout, 8.0),
            "closing": min(self._base_idle_timeout, 0.5),
        }
        self._idle_timeout = durations.get(expectation, self._base_idle_timeout)
        now = datetime.now(timezone.utc)
        self._last_response_activity = now
        self._last_activity = now
        self._speech_active = False
        self._speech_started_at = None

    def restore_response_deadline(self) -> None:
        """Undo rejected-turn activity without extending the listening window."""

        if self._last_response_activity is not None:
            self._last_activity = self._last_response_activity
        self._speech_active = False
        self._speech_started_at = None

    async def close_if_idle(self, *, now: datetime | None = None) -> bool:
        """Close active provider session after speech inactivity reaches the limit."""

        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("idle-check time must be timezone-aware")
        async with self._lock:
            if (
                self._phase != "active"
                or self._session_id is None
                or self._last_activity is None
                or (
                    self._speech_active
                    and self._speech_started_at is not None
                    and current - self._speech_started_at
                    < timedelta(seconds=MAX_CANDIDATE_SPEECH_SECONDS)
                )
                or current - self._last_activity < timedelta(seconds=self._idle_timeout)
            ):
                return False
            return await self._close_active_locked(self._session_id)

    async def close(self) -> bool:
        """Close active provider session and clear volatile pre-roll audio."""

        async with self._lock:
            if self._phase != "active" or self._session_id is None:
                self._cancel_duration_task()
                self._phase = "idle"
                self._pre_roll.clear()
                return False
            return await self._close_active_locked(self._session_id)

    async def _expire_session(self, session_id: UUID) -> None:
        """Stop provider session at hard cap, even when no audio frame arrives."""

        assert self._max_session_seconds is not None
        try:
            await asyncio.sleep(self._max_session_seconds)
            async with self._lock:
                if self._phase != "active" or self._session_id != session_id:
                    return
                await self._close_active_locked(session_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            _LOGGER.warning("maximum-duration session close failed (%s)", type(error).__name__)

    async def _close_active_locked(self, session_id: UUID) -> bool:
        self._phase = "closing"
        self._cancel_duration_task()
        try:
            await self._stop_session(session_id)
        finally:
            self._phase = "idle"
            self._session_id = None
            self._last_activity = None
            self._last_response_activity = None
            self._speech_active = False
            self._speech_started_at = None
            self._pre_roll.clear()
        return True

    def _cancel_duration_task(self) -> None:
        task = self._duration_task
        self._duration_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
