"""Explicit, owner-scoped conversational memory writes."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Protocol
from uuid import UUID

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from contracts.models import MemoryRecord, MemoryWriteRequest


class MemoryRepository(Protocol):
    async def write_memory(
        self,
        *,
        owner_id: UUID,
        request: MemoryWriteRequest,
        idempotency_key: str,
    ) -> tuple[MemoryRecord, bool]: ...


class SaveMemoryArgs(BaseModel):
    category: Literal["preference", "fact"] = Field(
        description="Whether saved content describes a user preference or a fact."
    )
    content: str = Field(
        min_length=1,
        max_length=500,
        description="One concise fact or preference the user explicitly asked Cube to remember.",
    )


class MemoryTools:
    """Bind one explicit save to its authenticated owner and source turn."""

    def __init__(
        self,
        repository: MemoryRepository,
        *,
        owner_id: UUID,
        source_turn_id: UUID,
        task_idempotency_key: str,
        context_revision: int,
    ) -> None:
        self._repository = repository
        self._owner_id = owner_id
        self._source_turn_id = source_turn_id
        self._task_idempotency_key = task_idempotency_key
        self._context_revision = context_revision

    def model_tool(self) -> StructuredTool:
        async def save_memory(
            *, category: Literal["preference", "fact"], content: str
        ) -> str:
            normalized_content = content.strip()
            if not normalized_content:
                return _json(
                    {
                        "status": "rejected",
                        "error_code": "INVALID_REQUEST",
                        "reason": "content is blank",
                    }
                )
            request = MemoryWriteRequest(
                category=category,
                content=normalized_content,
                source_turn_id=self._source_turn_id,
            )
            material = json.dumps(
                {
                    "task_idempotency_key": self._task_idempotency_key,
                    "context_revision": self._context_revision,
                    "category": category,
                    "content": normalized_content,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            idempotency_key = "voice-memory/" + hashlib.sha256(material).hexdigest()
            record, replayed = await self._repository.write_memory(
                owner_id=self._owner_id,
                request=request,
                idempotency_key=idempotency_key,
            )
            return _json(
                {
                    "status": "saved",
                    "memory_id": str(record.memory_id),
                    "category": record.category,
                    "content": record.content,
                    "replayed": replayed,
                }
            )

        return StructuredTool.from_function(
            coroutine=save_memory,
            name="save_memory",
            description=(
                "Save one fact or preference only after the user directly asks Cube "
                "to remember it. Never save inferred content or credentials."
            ),
            args_schema=SaveMemoryArgs,
        )


def _json(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
