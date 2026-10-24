"""Keep microphone capture continuous across wake and provider startup."""

from __future__ import annotations

import asyncio
from collections import deque

from device.alsa import AudioDeviceError, PcmFormat
from device.audio_runtime import CaptureStream


class StartupBufferedCapture:
    """Retain at most 20 seconds of post-wake PCM in memory, never on disk.

    Idle buffering stays at 500 ms. After wake, the larger bounded FIFO covers
    provider startup. Buffered frames drain at capture rate to preserve order
    without overflowing the runtime's low-latency transport queue.
    """

    def __init__(self, capture: CaptureStream, audio_format: PcmFormat) -> None:
        self._capture = capture
        self._format = audio_format
        self._frames: deque[bytes] = deque()
        self._capacity = 500 // audio_format.frame_duration_ms
        self._condition = asyncio.Condition()
        self._reader: asyncio.Task[None] | None = None
        self._error: Exception | None = None
        self._next_read = 0.0

    async def start(self) -> None:
        if self._reader is not None:
            return
        await self._capture.start()
        self._reader = asyncio.create_task(self._pump(), name="cube-continuous-mic")

    def begin_session(self) -> None:
        """Expand the FIFO before provider startup begins."""

        self._capacity = 20_000 // self._format.frame_duration_ms

    async def _pump(self) -> None:
        try:
            while True:
                payload = await self._capture.read_frame()
                async with self._condition:
                    if len(self._frames) >= self._capacity:
                        if self._capacity > 500 // self._format.frame_duration_ms:
                            raise AudioDeviceError("voice startup audio buffer exceeded 20 seconds")
                        self._frames.popleft()
                    self._frames.append(payload)
                    self._condition.notify_all()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            async with self._condition:
                self._error = error
                self._condition.notify_all()

    async def read_frame(self) -> bytes:
        loop = asyncio.get_running_loop()
        delay = self._next_read - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        async with self._condition:
            while not self._frames:
                if self._error is not None:
                    raise self._error
                if self._reader is None:
                    raise AudioDeviceError("continuous capture is closed")
                await self._condition.wait()
            if self._error is not None:
                raise self._error
            payload = self._frames.popleft()
        self._next_read = loop.time() + self._format.frame_duration_ms / 1000
        return payload

    async def close(self) -> None:
        reader, self._reader = self._reader, None
        if reader is not None:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        await self._capture.close()
        async with self._condition:
            self._frames.clear()
            self._condition.notify_all()
