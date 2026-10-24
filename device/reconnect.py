"""Cancellable reconnect loop with bounded exponential backoff."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from device.alsa import AudioDeviceError
from device.transport import DeviceTransportError


@dataclass(frozen=True)
class RetryNotice:
    """Safe retry status; never includes exception text or credentials."""

    attempts: int
    consecutive_failures: int
    delay_seconds: float
    error_type: str


RetryCallback = Callable[[RetryNotice], None]


class ReconnectSupervisor:
    """Retry one fully-cleaned connection attempt until stopped.

    The attempt callback owns session, socket, and audio cleanup. It should
    return only when its connection ends. Retryable device/network errors are
    reduced to their exception type before an optional status callback sees
    them; unexpected programming errors still escape.
    """

    def __init__(
        self,
        attempt: Callable[[], Awaitable[None]],
        *,
        initial_delay_seconds: float = 0.5,
        maximum_delay_seconds: float = 30.0,
        reset_after_seconds: float = 30.0,
        on_retry: RetryCallback | None = None,
    ) -> None:
        if initial_delay_seconds <= 0:
            raise ValueError("initial retry delay must be positive")
        if maximum_delay_seconds < initial_delay_seconds:
            raise ValueError("maximum retry delay must be at least the initial delay")
        if reset_after_seconds <= 0:
            raise ValueError("retry reset duration must be positive")
        self._attempt = attempt
        self._initial_delay = initial_delay_seconds
        self._maximum_delay = maximum_delay_seconds
        self._reset_after = reset_after_seconds
        self._on_retry = on_retry
        self._stop_event = asyncio.Event()
        self._attempts = 0
        self._consecutive_failures = 0

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        """Stop retrying and cancel any active attempt cooperatively."""

        self._stop_event.set()

    async def run(self) -> None:
        """Run attempts until stop; cancel active attempt when stop is set."""

        while not self._stop_event.is_set():
            self._attempts += 1
            started_at = asyncio.get_running_loop().time()
            attempt_task = asyncio.create_task(self._attempt(), name="cube-device-connect")
            stop_task = asyncio.create_task(self._stop_event.wait(), name="cube-device-stop")
            error_type = "ConnectionEnded"
            try:
                done, _ = await asyncio.wait(
                    (attempt_task, stop_task), return_when=asyncio.FIRST_COMPLETED
                )
                if stop_task in done:
                    attempt_task.cancel()
                    await asyncio.gather(attempt_task, return_exceptions=True)
                    return
                try:
                    await attempt_task
                except asyncio.CancelledError:
                    raise
                except (AudioDeviceError, DeviceTransportError, ConnectionError, TimeoutError) as error:
                    error_type = type(error).__name__
                elapsed = asyncio.get_running_loop().time() - started_at
            finally:
                if not attempt_task.done():
                    attempt_task.cancel()
                if not stop_task.done():
                    stop_task.cancel()
                await asyncio.gather(attempt_task, stop_task, return_exceptions=True)

            if self._stop_event.is_set():
                return
            if elapsed >= self._reset_after:
                self._consecutive_failures = 0
            self._consecutive_failures += 1
            exponent = min(self._consecutive_failures - 1, 30)
            delay = min(self._initial_delay * (2**exponent), self._maximum_delay)
            if self._on_retry is not None:
                self._on_retry(
                    RetryNotice(
                        attempts=self._attempts,
                        consecutive_failures=self._consecutive_failures,
                        delay_seconds=delay,
                        error_type=error_type,
                    )
                )
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
            except TimeoutError:
                pass
