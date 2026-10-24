from datetime import datetime, timezone
from decimal import Decimal
from unittest import IsolatedAsyncioTestCase
from uuid import uuid4

from contracts.models import UsageRecord
from service.live_provider import LiveSessionClose
from service.live_usage import live_usage_id, persist_live_session_usage


class FakeUsageStore:
    def __init__(self):
        self.calls = []

    async def record_usage_once(self, **kwargs):
        self.calls.append(kwargs)
        return UsageRecord(
            usage_id=kwargs["usage_id"],
            session_id=kwargs["session_id"],
            job_id=kwargs.get("job_id"),
            provider=kwargs["provider"],
            measurement=kwargs["measurement"],
            observed_at=kwargs.get("observed_at") or datetime.now(timezone.utc),
        )


class LiveUsagePersistenceTests(IsolatedAsyncioTestCase):
    async def test_records_confirmed_seconds_without_inventing_cost(self):
        store = FakeUsageStore()
        session_id = uuid4()
        closed = LiveSessionClose(
            provider_session_id="provider-session-42",
            connected_seconds=Decimal("17.25"),
            final_event={"type": "session.closed", "usage": {"seconds": 17.25}},
        )
        observed_at = datetime(2026, 10, 7, tzinfo=timezone.utc)

        record = await persist_live_session_usage(
            store,
            owner_id=uuid4(),
            session_id=session_id,
            closed=closed,
            observed_at=observed_at,
        )

        self.assertIsNotNone(record)
        self.assertEqual(record.usage_id, live_usage_id("provider-session-42"))
        self.assertEqual(record.session_id, session_id)
        self.assertEqual(record.provider, "openai-live")
        self.assertEqual(record.measurement.connected_seconds, Decimal("17.25"))
        self.assertIsNone(record.measurement.cost_usd)
        self.assertIsNone(record.measurement.cost_kind)
        self.assertEqual(store.calls[0]["observed_at"], observed_at)

    async def test_missing_final_usage_is_not_recorded(self):
        store = FakeUsageStore()
        closed = LiveSessionClose(
            provider_session_id="provider-session-43",
            connected_seconds=None,
            final_event={"type": "session.closed"},
        )

        record = await persist_live_session_usage(
            store,
            owner_id=uuid4(),
            session_id=uuid4(),
            closed=closed,
        )

        self.assertIsNone(record)
        self.assertEqual(store.calls, [])

    async def test_rejects_unconfirmed_close_result(self):
        store = FakeUsageStore()
        closed = LiveSessionClose(
            provider_session_id="provider-session-44",
            connected_seconds=Decimal("4"),
            final_event={"type": "session.usage.updated", "usage": {"seconds": 4}},
        )

        with self.assertRaisesRegex(ValueError, "session.closed"):
            await persist_live_session_usage(
                store,
                owner_id=uuid4(),
                session_id=uuid4(),
                closed=closed,
            )
        self.assertEqual(store.calls, [])

    async def test_rejects_seconds_that_disagree_with_final_event(self):
        store = FakeUsageStore()
        closed = LiveSessionClose(
            provider_session_id="provider-session-45",
            connected_seconds=Decimal("4"),
            final_event={"type": "session.closed", "usage": {"seconds": 5}},
        )

        with self.assertRaisesRegex(ValueError, "do not match"):
            await persist_live_session_usage(
                store,
                owner_id=uuid4(),
                session_id=uuid4(),
                closed=closed,
            )
        self.assertEqual(store.calls, [])

    def test_provider_session_id_is_stable_and_required(self):
        self.assertEqual(live_usage_id("session-1"), live_usage_id("session-1"))
        with self.assertRaisesRegex(ValueError, "must not be blank"):
            live_usage_id("  ")
