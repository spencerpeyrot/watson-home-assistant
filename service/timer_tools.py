"""Owner-scoped, idempotent timer tools for simple voice tasks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Protocol
from uuid import UUID

from langchain_core.tools import StructuredTool
from pydantic import AwareDatetime, BaseModel, Field

from contracts.models import TimerCreateRequest, TimerRecord


class TimerRepository(Protocol):
    async def create_timer(
        self,
        *,
        owner_id: UUID,
        device_id: str,
        request: TimerCreateRequest,
    ) -> TimerRecord: ...

    async def list_timers(
        self, *, owner_id: UUID, limit: int = 100
    ) -> tuple[TimerRecord, ...]: ...

    async def cancel_timer(self, *, owner_id: UUID, timer_id: UUID) -> bool: ...


class CreateTimerArgs(BaseModel):
    due_at: AwareDatetime = Field(description="Absolute ISO 8601 date-time with timezone offset")
    message: str = Field(min_length=1, max_length=500)


class ListTimersArgs(BaseModel):
    pass


class CancelTimerArgs(BaseModel):
    timer_id: UUID


class TimerTools:
    """Bind timer operations to authenticated owner and current task context."""

    def __init__(
        self,
        repository: TimerRepository,
        *,
        owner_id: UUID,
        device_id: str,
        conversation_id: UUID,
        task_idempotency_key: str,
        context_revision: int,
    ) -> None:
        self._repository = repository
        self._owner_id = owner_id
        self._device_id = device_id
        self._conversation_id = conversation_id
        self._task_key_digest = hashlib.sha256(
            task_idempotency_key.encode("utf-8")
        ).hexdigest()[:32]
        self._context_revision = context_revision

    def model_tools(self) -> tuple[StructuredTool, ...]:
        async def create_timer(*, due_at: AwareDatetime, message: str) -> str:
            return await self._create_timer(due_at=due_at, message=message)

        async def list_timers() -> str:
            return await self._list_timers()

        async def cancel_timer(*, timer_id: UUID) -> str:
            return await self._cancel_timer(timer_id=timer_id)

        return (
            StructuredTool.from_function(
                coroutine=create_timer,
                name="create_timer",
                description="Create a reminder timer at a clear, future date and time.",
                args_schema=CreateTimerArgs,
            ),
            StructuredTool.from_function(
                coroutine=list_timers,
                name="list_timers",
                description=(
                    "List recent timers for this owner so a specific timer can be identified."
                ),
                args_schema=ListTimersArgs,
            ),
            StructuredTool.from_function(
                coroutine=cancel_timer,
                name="cancel_timer",
                description="Cancel one specific, active timer by timer ID.",
                args_schema=CancelTimerArgs,
            ),
        )

    async def _create_timer(self, *, due_at: AwareDatetime, message: str) -> str:
        normalized_message = message.strip()
        now = datetime.now(timezone.utc)
        normalized_due = due_at.astimezone(timezone.utc)
        if not normalized_message:
            return _json(
                {
                    "status": "rejected",
                    "error_code": "INVALID_REQUEST",
                    "reason": "message is blank",
                }
            )
        if normalized_due <= now:
            return _json(
                {
                    "status": "rejected",
                    "error_code": "INVALID_REQUEST",
                    "reason": "due_at must be in the future",
                }
            )
        digest = hashlib.sha256(normalized_message.encode("utf-8")).hexdigest()
        request = TimerCreateRequest(
            conversation_id=self._conversation_id,
            due_at=normalized_due,
            message=normalized_message,
            idempotency_key=(
                f"task/{self._task_key_digest}/revision/{self._context_revision}/timer/{digest}"
            ),
        )
        record = await self._repository.create_timer(
            owner_id=self._owner_id,
            device_id=self._device_id,
            request=request,
        )
        return _json(
            {
                "status": record.status,
                "timer_id": str(record.timer_id),
                "due_at": record.due_at.isoformat(),
                "message": record.message,
            }
        )

    async def _list_timers(self) -> str:
        records: Sequence[TimerRecord] = await self._repository.list_timers(
            owner_id=self._owner_id,
            limit=10,
        )
        return _json(
            {
                "timers": [
                    {
                        "timer_id": str(record.timer_id),
                        "due_at": record.due_at.isoformat(),
                        "message": record.message,
                        "status": record.status,
                    }
                    for record in records
                ]
            }
        )

    async def _cancel_timer(self, *, timer_id: UUID) -> str:
        cancelled = await self._repository.cancel_timer(
            owner_id=self._owner_id,
            timer_id=timer_id,
        )
        return _json(
            {
                "status": "cancelled" if cancelled else "not_active",
                "timer_id": str(timer_id),
            }
        )


def _json(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
