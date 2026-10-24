from __future__ import annotations

import asyncio
import inspect
import unittest

from device.alsa import (
    DEFAULT_PLAYBACK_DRAIN_TIMEOUT_SECONDS,
    AlsaPlayback,
    AudioDeviceError,
)


class FakeStdin:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.closed = asyncio.Event()

    def close(self) -> None:
        self.calls.append("stdin_close")

    async def wait_closed(self) -> None:
        self.calls.append("stdin_wait_closed")
        self.closed.set()


class FakeProcess:
    def __init__(self, *, exit_on_eof: bool = True) -> None:
        self.calls: list[str] = []
        self.stdin = FakeStdin(self.calls)
        self.exit_on_eof = exit_on_eof
        self.returncode: int | None = None
        self.terminated = asyncio.Event()

    async def wait(self) -> int:
        self.calls.append("process_wait")
        if self.returncode is not None:
            return self.returncode
        if self.exit_on_eof:
            await self.stdin.closed.wait()
            self.returncode = 0
        else:
            await self.terminated.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.calls.append("process_terminate")
        self.returncode = -15
        self.terminated.set()


class AlsaPlaybackDrainTests(unittest.IsolatedAsyncioTestCase):
    def test_drain_default_allows_slow_speaker_output_to_finish(self) -> None:
        default = inspect.signature(AlsaPlayback.drain_and_close).parameters[
            "timeout_seconds"
        ].default

        self.assertEqual(default, DEFAULT_PLAYBACK_DRAIN_TIMEOUT_SECONDS)
        self.assertEqual(default, 30.0)

    async def test_drain_sends_stdin_eof_before_waiting_for_aplay(self) -> None:
        process = FakeProcess()
        playback = AlsaPlayback("unused")
        playback._process = process  # type: ignore[assignment]

        await playback.drain_and_close()

        self.assertEqual(
            process.calls,
            ["stdin_close", "stdin_wait_closed", "process_wait"],
        )
        self.assertEqual(process.returncode, 0)
        self.assertIsNone(playback._process)

    async def test_drain_timeout_terminates_stuck_aplay(self) -> None:
        process = FakeProcess(exit_on_eof=False)
        playback = AlsaPlayback("unused")
        playback._process = process  # type: ignore[assignment]

        with self.assertRaisesRegex(AudioDeviceError, "drain timed out"):
            await playback.drain_and_close(timeout_seconds=0.01)

        self.assertTrue(process.terminated.is_set())
        self.assertEqual(process.returncode, -15)
        self.assertIsNone(playback._process)


if __name__ == "__main__":
    unittest.main()
