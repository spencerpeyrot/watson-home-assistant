from __future__ import annotations

import asyncio
import unittest

from device.reconnect import ReconnectSupervisor, RetryNotice
from device.transport import DeviceTransportError


class ReconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_with_capped_backoff_and_never_exposes_error_text(self) -> None:
        notices: list[RetryNotice] = []
        attempts = 0
        supervisor: ReconnectSupervisor

        async def attempt() -> None:
            nonlocal attempts
            attempts += 1
            raise DeviceTransportError("secret token must not escape")

        def on_retry(notice: RetryNotice) -> None:
            notices.append(notice)
            if len(notices) == 3:
                supervisor.stop()

        supervisor = ReconnectSupervisor(
            attempt,
            initial_delay_seconds=0.001,
            maximum_delay_seconds=0.002,
            on_retry=on_retry,
        )
        await asyncio.wait_for(supervisor.run(), timeout=1)

        self.assertEqual(attempts, 3)
        self.assertEqual([item.delay_seconds for item in notices], [0.001, 0.002, 0.002])
        self.assertTrue(all(item.error_type == "DeviceTransportError" for item in notices))
        self.assertTrue(all("secret" not in repr(item) for item in notices))

    async def test_external_cancellation_cleans_up_active_attempt(self) -> None:
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def attempt() -> None:
            started.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.set()

        supervisor = ReconnectSupervisor(attempt)
        task = asyncio.create_task(supervisor.run())
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(cleaned.is_set())


if __name__ == "__main__":
    unittest.main()
