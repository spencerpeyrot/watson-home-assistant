"""Persist confirmed GPT-Live session usage without inferring provider cost."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from uuid import NAMESPACE_URL, UUID, uuid5

from contracts.models import UsageMeasurement, UsageRecord
from service.live_provider import LiveSessionClose
from service.postgres import PostgresStore


def live_usage_id(provider_session_id: str) -> UUID:
    """Return stable usage identity for one provider session."""

    if not provider_session_id.strip():
        raise ValueError("provider session ID must not be blank")
    return uuid5(NAMESPACE_URL, f"cube-assistant:openai-live-usage:{provider_session_id}")


async def persist_live_session_usage(
    store: PostgresStore,
    *,
    owner_id: UUID,
    session_id: UUID,
    closed: LiveSessionClose,
    observed_at: datetime | None = None,
) -> UsageRecord | None:
    """Store final provider seconds once; return None when final usage is absent.

    This function accepts only a close result backed by `session.closed`. It
    records no dollar cost because the provider event does not provide one.
    """

    if closed.final_event.get("type") != "session.closed":
        raise ValueError("Live usage requires a confirmed session.closed event")
    if not closed.provider_session_id.strip():
        raise ValueError("provider session ID must not be blank")
    usage = closed.final_event.get("usage")
    raw_seconds = usage.get("seconds") if isinstance(usage, dict) else None
    final_seconds = None
    if not isinstance(raw_seconds, bool) and raw_seconds is not None:
        try:
            parsed_seconds = Decimal(str(raw_seconds))
        except (InvalidOperation, TypeError, ValueError):
            parsed_seconds = None
        if parsed_seconds is not None and parsed_seconds.is_finite() and parsed_seconds >= 0:
            final_seconds = parsed_seconds
    if final_seconds != closed.connected_seconds:
        raise ValueError("Live close seconds do not match final provider usage")
    if closed.connected_seconds is None:
        return None
    return await store.record_usage_once(
        owner_id=owner_id,
        session_id=session_id,
        usage_id=live_usage_id(closed.provider_session_id),
        provider="openai-live",
        measurement=UsageMeasurement(connected_seconds=closed.connected_seconds),
        observed_at=observed_at,
    )
