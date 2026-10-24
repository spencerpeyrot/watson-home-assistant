"""Async raw-PCM wrappers around the Pi's ALSA command-line tools."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from asyncio.subprocess import Process

DEFAULT_PLAYBACK_DRAIN_TIMEOUT_SECONDS = 30.0


class AudioDeviceError(RuntimeError):
    """Sanitized ALSA capture or playback failure."""


@dataclass(frozen=True)
class PcmFormat:
    sample_rate_hz: int = 24_000
    channels: int = 1
    frame_duration_ms: int = 20

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0 or self.channels <= 0 or self.frame_duration_ms <= 0:
            raise ValueError("PCM format values must be positive")
        if self.frame_duration_ms > 100:
            raise ValueError("PCM frame duration must not exceed 100 ms")
        if self.frame_bytes_numerator % 1000:
            raise ValueError("PCM frame duration must align to whole samples")

    @property
    def frame_bytes_numerator(self) -> int:
        return self.sample_rate_hz * self.channels * 2 * self.frame_duration_ms

    @property
    def frame_bytes(self) -> int:
        return self.frame_bytes_numerator // 1000


class AlsaCapture:
    """Read fixed-size PCM16 frames from an `arecord` subprocess."""

    def __init__(self, pcm: str, audio_format: PcmFormat = PcmFormat()) -> None:
        if not pcm.strip():
            raise ValueError("capture PCM name must not be blank")
        self.pcm = pcm
        self.audio_format = audio_format
        self._process: Process | None = None

    async def start(self) -> None:
        if self._process is not None:
            raise AudioDeviceError("capture stream already started")
        try:
            self._process = await asyncio.create_subprocess_exec(
                "arecord",
                "--quiet",
                "--device",
                self.pcm,
                "--file-type",
                "raw",
                "--format",
                "S16_LE",
                "--rate",
                str(self.audio_format.sample_rate_hz),
                "--channels",
                str(self.audio_format.channels),
                "-",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as error:
            raise AudioDeviceError(f"arecord could not start ({type(error).__name__})") from None
        if self._process.stdout is None:
            await self.close()
            raise AudioDeviceError("arecord did not provide a capture stream")

    async def read_frame(self) -> bytes:
        process = self._process
        if process is None or process.stdout is None:
            raise AudioDeviceError("capture stream is not running")
        try:
            return await process.stdout.readexactly(self.audio_format.frame_bytes)
        except asyncio.IncompleteReadError:
            status = await process.wait()
            raise AudioDeviceError(f"arecord capture stream ended (exit {status})") from None
        except OSError as error:
            raise AudioDeviceError(f"capture read failed ({type(error).__name__})") from None

    async def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()


class AlsaPlayback:
    """Write PCM16 frames to `aplay`; `abort` avoids intentional queue draining."""

    def __init__(self, pcm: str, audio_format: PcmFormat = PcmFormat()) -> None:
        if not pcm.strip():
            raise ValueError("playback PCM name must not be blank")
        self.pcm = pcm
        self.audio_format = audio_format
        self._process: Process | None = None

    async def start(self) -> None:
        if self._process is not None:
            raise AudioDeviceError("playback stream already started")
        try:
            self._process = await asyncio.create_subprocess_exec(
                "aplay",
                "--quiet",
                "--device",
                self.pcm,
                "--file-type",
                "raw",
                "--format",
                "S16_LE",
                "--rate",
                str(self.audio_format.sample_rate_hz),
                "--channels",
                str(self.audio_format.channels),
                "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as error:
            raise AudioDeviceError(f"aplay could not start ({type(error).__name__})") from None
        if self._process.stdin is None:
            await self.abort()
            raise AudioDeviceError("aplay did not provide a playback stream")

    async def write_frame(self, payload: bytes) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise AudioDeviceError("playback stream is not running")
        if len(payload) != self.audio_format.frame_bytes:
            raise ValueError(f"PCM frame must contain exactly {self.audio_format.frame_bytes} bytes")
        if process.returncode is not None:
            raise AudioDeviceError("aplay playback stream stopped")
        try:
            process.stdin.write(payload)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as error:
            raise AudioDeviceError(f"playback write failed ({type(error).__name__})") from None

    async def drain_and_close(
        self, *, timeout_seconds: float = DEFAULT_PLAYBACK_DRAIN_TIMEOUT_SECONDS
    ) -> None:
        """Finish queued audio normally; not suitable for interruption/reconnect."""

        if timeout_seconds <= 0:
            raise ValueError("playback drain timeout must be positive")
        process, self._process = self._process, None
        if process is None:
            return
        try:
            async with asyncio.timeout(timeout_seconds):
                if process.stdin is not None:
                    process.stdin.close()
                    # Propagate EOF to aplay before waiting for ALSA's hardware
                    # drain. StreamWriter.close() alone only schedules closure.
                    await process.stdin.wait_closed()
                await process.wait()
        except asyncio.TimeoutError:
            process.terminate()
            await process.wait()
            raise AudioDeviceError("aplay playback drain timed out") from None
        if process.returncode != 0:
            raise AudioDeviceError(f"aplay playback failed (exit {process.returncode})")

    async def abort(self) -> None:
        """Stop playback without intentionally draining queued frames."""

        process, self._process = self._process, None
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
