from __future__ import annotations

import asyncio
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import UUID

from contracts.models import TimerFiredEvent
from service.postgres import ClaimedTimer
from service.timers import TimerDeliveryWorker


TIMER_ID = UUID("00000000-0000-4000-8000-000000000901")
OWNER_ID = UUID("00000000-0000-4000-8000-000000000902")
WORKER_ID = UUID("00000000-0000-4000-8000-000000000903")


def claimed_timer(attempt_count: int = 1) -> ClaimedTimer:
    now = datetime.now(timezone.utc)
    event = TimerFiredEvent(
        event_id=TIMER_ID,
        timer_id=TIMER_ID,
        device_id="cube-01",
        due_at=now - timedelta(seconds=1),
        fired_at=now,
        message="Tea is ready.",
        attempt_count=attempt_count,
    )
    return ClaimedTimer(
        owner_id=OWNER_ID,
        worker_id=WORKER_ID,
        event=event,
        lease_expires_at=now + timedelta(seconds=30),
    )


class FakeTimerStore:
    def __init__(self, timer: ClaimedTimer | None) -> None:
        self.timer = timer
        self.acked: list[tuple[UUID, UUID, UUID]] = []
        self.retried: list[tuple[UUID, UUID, UUID, datetime, int]] = []

    async def claim_due_timers(self, **_kwargs):
        if self.timer is None:
            return ()
        timer, self.timer = self.timer, None
        return (timer,)

    async def renew_timer_lease(self, **_kwargs):
        return True

    async def ack_timer_delivery(self, **kwargs):
        self.acked.append(
            (kwargs["owner_id"], kwargs["timer_id"], kwargs["worker_id"])
        )
        return True

    async def retry_timer_delivery(self, **kwargs):
        self.retried.append(
            (
                kwargs["owner_id"],
                kwargs["timer_id"],
                kwargs["worker_id"],
                kwargs["retry_after"],
                kwargs["max_attempts"],
            )
        )
        return True


class TimerDeliveryWorkerTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def fake_trace():
        class Trace:
            def __init__(self):
                self.updates = []
                self.ended = None

            def update(self, **values):
                self.updates.append(values)

            @contextmanager
            def activate(self):
                yield self

            def end(self):
                self.ended = self.updates[-1]["status"]

            def event(self, *_args, **_kwargs):
                pass

        return Trace()

    async def test_success_acks_stable_deduplicable_event(self) -> None:
        store = FakeTimerStore(claimed_timer())
        received: list[TimerFiredEvent] = []

        async def deliver(event: TimerFiredEvent) -> None:
            received.append(event)

        worker = TimerDeliveryWorker(store=store, deliver=deliver, worker_id=WORKER_ID)
        trace = self.fake_trace()
        with patch("service.timers.start_session", return_value=trace) as start_trace:
            self.assertTrue(await worker.run_once())
        self.assertEqual(start_trace.call_args.kwargs["delivery_attempt"], 1)
        self.assertEqual(trace.ended, "succeeded")
        self.assertEqual(received[0].event_id, TIMER_ID)
        self.assertEqual(received[0].timer_id, TIMER_ID)
        self.assertEqual(store.acked, [(OWNER_ID, TIMER_ID, WORKER_ID)])
        self.assertEqual(store.retried, [])

    async def test_delivery_error_requests_bounded_retry(self) -> None:
        store = FakeTimerStore(claimed_timer(attempt_count=2))

        async def deliver(_event: TimerFiredEvent) -> None:
            raise OSError("transport down")

        worker = TimerDeliveryWorker(
            store=store,
            deliver=deliver,
            worker_id=WORKER_ID,
            retry_base_seconds=0.01,
            retry_max_seconds=0.02,
            max_attempts=3,
        )
        before = datetime.now(timezone.utc)
        trace = self.fake_trace()
        with patch("service.timers.start_session", return_value=trace):
            self.assertTrue(await worker.run_once())
        self.assertEqual(trace.ended, "failed")
        after = datetime.now(timezone.utc)
        self.assertEqual(store.acked, [])
        self.assertEqual(len(store.retried), 1)
        _, timer_id, worker_id, retry_at, max_attempts = store.retried[0]
        self.assertEqual((timer_id, worker_id, max_attempts), (TIMER_ID, WORKER_ID, 3))
        self.assertGreaterEqual(retry_at, before)
        self.assertLessEqual(retry_at, after + timedelta(milliseconds=100))

    async def test_stalled_delivery_times_out_and_retries(self) -> None:
        store = FakeTimerStore(claimed_timer())

        async def deliver(_event: TimerFiredEvent) -> None:
            await asyncio.Event().wait()

        worker = TimerDeliveryWorker(
            store=store, deliver=deliver, worker_id=WORKER_ID,
            delivery_timeout_seconds=0.001, retry_base_seconds=0.01,
            retry_max_seconds=0.02,
        )
        trace = self.fake_trace()
        with patch("service.timers.start_session", return_value=trace):
            self.assertTrue(await worker.run_once())
        self.assertEqual(trace.updates[-1]["status"], "timed_out")
        self.assertEqual(trace.updates[-1]["failure_stage"], "delivery")
        self.assertEqual(trace.updates[-1]["error_category"], "timeout")
        self.assertEqual(trace.updates[-1]["timeout_ms"], 1)
        self.assertEqual(len(store.retried), 1)
        self.assertEqual(store.acked, [])

    async def test_empty_queue_returns_false(self) -> None:
        worker = TimerDeliveryWorker(
            store=FakeTimerStore(None), deliver=lambda _event: asyncio.sleep(0), worker_id=WORKER_ID
        )
        self.assertFalse(await worker.run_once())


if __name__ == "__main__":
    unittest.main()
