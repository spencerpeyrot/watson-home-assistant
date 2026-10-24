"""Secure PostgreSQL checkpointer lifecycle for LangGraph workers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


@asynccontextmanager
async def postgres_checkpointer(database_url: str) -> AsyncIterator[AsyncPostgresSaver]:
    """Open saver with strict MessagePack types and ensure schema is ready."""

    if not database_url.strip():
        raise ValueError("database URL must not be blank")
    safe_serializer = JsonPlusSerializer(allowed_msgpack_modules=None)
    async with AsyncPostgresSaver.from_conn_string(
        database_url, serde=safe_serializer
    ) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
