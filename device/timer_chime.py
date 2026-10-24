"""Optional low-amplitude local chime for delivered timer events."""

from __future__ import annotations

import math
import struct
from collections.abc import Callable
from typing import Protocol

from contracts.models import TimerFiredEvent
from device.alsa import AlsaPlayback, PcmFormat


class PcmPlayer(Protocol):
    async def start(self) -> None: ...

    async def write_frame(self, payload: bytes) -> None: ...

    async def drain_and_close(self) -> None: ...

    async def abort(self) -> None: ...


PlayerFactory = Callable[[str, PcmFormat], PcmPlayer]


class AlsaTimerChime:
    """Play one quiet, short chime; caller must choose and configure the PCM."""

    MAX_AMPLITUDE = 0.1
    NOTES = ((880, 4), (0, 2), (1175, 4), (0, 2), (1480, 6))

    def __init__(
        self,
        pcm: str,
        *,
        audio_format: PcmFormat = PcmFormat(),
        amplitude: float = 0.05,
        player_factory: PlayerFactory = AlsaPlayback,
    ) -> None:
        if not pcm.strip():
            raise ValueError("timer chime PCM name must not be blank")
        if not math.isfinite(amplitude) or not 0 < amplitude <= self.MAX_AMPLITUDE:
            raise ValueError("timer chime amplitude must be above zero and at most 0.1")
        frame_samples = audio_format.frame_bytes // (audio_format.channels * 2)
        if (
            frame_samples <= 0
            or frame_samples * audio_format.channels * 2 != audio_format.frame_bytes
        ):
            raise ValueError("audio format must contain whole PCM16 samples per frame")
        self._pcm = pcm
        self._format = audio_format
        self._amplitude = amplitude
        self._player_factory = player_factory

    async def __call__(self, _event: TimerFiredEvent) -> None:
        await self.play()

    async def play(self) -> None:
        """Play one chime without requiring a timer event."""

        player = self._player_factory(self._pcm, self._format)
        try:
            await player.start()
            for frame in timer_chime_frames(
                self._format,
                amplitude=self._amplitude,
                notes=self.NOTES,
            ):
                await player.write_frame(frame)
            await player.drain_and_close()
        except BaseException:
            await player.abort()
            raise


def timer_chime_frames(
    audio_format: PcmFormat = PcmFormat(),
    *,
    amplitude: float = 0.05,
    notes: tuple[tuple[int, int], ...] = AlsaTimerChime.NOTES,
) -> tuple[bytes, ...]:
    """Generate bounded PCM16 frames; frequency zero means silence."""

    if not math.isfinite(amplitude) or not 0 < amplitude <= AlsaTimerChime.MAX_AMPLITUDE:
        raise ValueError("timer chime amplitude must be above zero and at most 0.1")
    channels = audio_format.channels
    frame_samples = audio_format.frame_bytes // (channels * 2)
    if frame_samples <= 0 or frame_samples * channels * 2 != audio_format.frame_bytes:
        raise ValueError("audio format must contain whole PCM16 samples per frame")
    if not notes or any(frequency < 0 or frame_count < 1 for frequency, frame_count in notes):
        raise ValueError("timer chime notes must have nonnegative frequency and positive duration")

    peak = round(amplitude * 32_767)
    frames: list[bytes] = []
    for frequency, frame_count in notes:
        if frequency >= audio_format.sample_rate_hz / 2:
            raise ValueError("timer chime tone must be below the Nyquist frequency")
        samples = bytearray()
        segment_samples = frame_samples * frame_count
        edge_samples = max(1, audio_format.sample_rate_hz // 200)
        for sample_index in range(segment_samples):
            if frequency == 0:
                sample = 0
            else:
                envelope = min(
                    1.0,
                    sample_index / edge_samples,
                    (segment_samples - 1 - sample_index) / edge_samples,
                )
                phase = 2.0 * math.pi * frequency * sample_index / audio_format.sample_rate_hz
                sample = round(peak * max(0.0, envelope) * math.sin(phase))
            for _channel in range(channels):
                samples.extend(struct.pack("<h", sample))
        segment = bytes(samples)
        for offset in range(0, len(segment), audio_format.frame_bytes):
            frames.append(segment[offset : offset + audio_format.frame_bytes])
    return tuple(frames)
