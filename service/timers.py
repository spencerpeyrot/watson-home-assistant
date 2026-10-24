"""Durable timer delivery with leases and receiver-deduplicable events."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from contracts.models import TimerFiredEvent
from service.postgres import ClaimedTimer, PostgresStore, TimerLeaseLost
from voice.observability import start_session


TimerDelivery = Callable[[TimerFiredEvent], Awaitable[None]]
_LOGGER = logging.getLogger(__name__)


class TimerDeliveryWorker:
    """Deliver due timers at least once; receivers deduplicate by stable event ID."""

    def __init__(
        self,
        *,
        store: PostgresStore,
        deliver: TimerDelivery,
        worker_id: UUID,
        lease_seconds: int = 30,
        delivery_timeout_seconds: float = 30.0,
        poll_seconds: float = 0.5,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 300.0,
        max_attempts: int = 8,
    ) -> None:
        if (
            lease_seconds < 1
            or delivery_timeout_seconds <= 0
            or poll_seconds <= 0
            or retry_base_seconds <= 0
            or retry_max_seconds < retry_base_seconds
            or max_attempts < 1
        ):
            raise ValueError("timer worker limits are invalid")
        self._store = store
        self._deliver = deliver
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._delivery_timeout_seconds = delivery_timeout_seconds
        self._poll_seconds = poll_seconds
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds
        self._max_attempts = max_attempts

    async def run_once(self) -> bool:
        """Deliver one due timer; return false when none are available."""

        claimed = await self._store.claim_due_timers(
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
            max_attempts=self._max_attempts,
            limit=1,
        )
        if not claimed:
            return False
        await self._deliver_one(claimed[0])
        return True

    async def run(self, stop_event: asyncio.Event) -> None:
        """Poll until stopped; expired leases recover delivery after process loss."""

        while not stop_event.is_set():
            try:
                processed = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                _LOGGER.warning("timer delivery pass failed (%s)", type(error).__name__)
                processed = False
            if processed:
                continue
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._poll_seconds)
            except TimeoutError:
                pass

    async def _deliver_one(self, timer: ClaimedTimer) -> None:
        event = timer.event
        trace = start_session(
            uuid4(), "timer.delivery", event_id=event.event_id, timer_id=event.timer_id,
            delivery_attempt=event.attempt_count,
            delivery_delay_ms=max(0, round((event.fired_at - event.due_at).total_seconds() * 1000)),
        )
        trace.update(status="running")
        delivery = heartbeat = None
        try:
            with trace.activate():
                delivery = asyncio.create_task(self._deliver(event))
                heartbeat = asyncio.create_task(self._heartbeat(timer, delivery))
            async with asyncio.timeout(self._delivery_timeout_seconds):
                done, _ = await asyncio.wait((delivery, heartbeat), return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                await heartbeat
            await delivery
        except TimerLeaseLost as error:
            trace.update(status="abandoned", error_type=type(error).__name__, error_category="lease_lost")
            return
        except asyncio.CancelledError:
            trace.update(status="cancelled", error_category="cancelled")
            raise
        except Exception as error:
            retry_seconds = min(
                self._retry_max_seconds,
                self._retry_base_seconds * (2 ** (event.attempt_count - 1)),
            )
            timeout = isinstance(error, TimeoutError)
            trace.update(status="timed_out" if timeout else "failed",
                         failure_stage="delivery", error_type=type(error).__name__,
                         error_category="timeout" if timeout else "delivery_failed",
                         timeout_ms=round(self._delivery_timeout_seconds * 1000) if timeout else None,
                         retry_after_ms=round(retry_seconds * 1000))
            try:
                await self._store.retry_timer_delivery(
                    owner_id=timer.owner_id,
                    timer_id=event.timer_id,
                    worker_id=self._worker_id,
                    retry_after=datetime.now(timezone.utc) + timedelta(seconds=retry_seconds),
                    max_attempts=self._max_attempts,
                )
            except Exception as retry_error:
                trace.event("timer.retry_persist_failed", status="failed",
                            error_type=type(retry_error).__name__)
                raise
            return
        else:
            try:
                await self._store.ack_timer_delivery(
                    owner_id=timer.owner_id,
                    timer_id=event.timer_id,
                    worker_id=self._worker_id,
                )
            except Exception as error:
                trace.update(status="failed", failure_stage="acknowledge",
                             error_type=type(error).__name__, error_category="ack_failed")
                raise
            trace.update(status="succeeded", stage="acknowledged")
        finally:
            for task in (delivery, heartbeat):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*(task for task in (delivery, heartbeat) if task is not None),
                                 return_exceptions=True)
            trace.end()

    async def _heartbeat(
        self, timer: ClaimedTimer, delivery: asyncio.Task[None]
    ) -> None:
        interval = max(0.1, self._lease_seconds / 3)
        while not delivery.done():
            await asyncio.sleep(interval)
            if delivery.done():
                return
            renewed = await self._store.renew_timer_lease(
                timer_id=timer.event.timer_id,
                worker_id=self._worker_id,
                lease_seconds=self._lease_seconds,
            )
            if not renewed:
                delivery.cancel()
                raise TimerLeaseLost("timer delivery lease could not be renewed")
