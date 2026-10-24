"""Durable, at-most-once local tool execution for cloud-bridged requests."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from contracts.models import LocalToolRequest, ToolResult


ToolExecutor = Callable[[LocalToolRequest], Awaitable[ToolResult]]
ToolStatus = Literal["succeeded", "failed", "denied", "needs_reconciliation"]


class LocalToolInbox:
    """Persist request identity and result before acknowledging the gateway.

    A duplicate completed action replays its result. A process crash after the
    action starts but before the result is committed becomes reconciliation;
    the device never repeats a possibly completed external write.
    """

    def __init__(self, database_path: str | Path, *, device_id: str) -> None:
        if not device_id.strip():
            raise ValueError("device ID must not be blank")
        self._database_path = str(database_path)
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        self._device_id = device_id
        self._lock = asyncio.Lock()
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS local_tool_inbox ("
                "action_id TEXT PRIMARY KEY, request_json TEXT NOT NULL, "
                "state TEXT NOT NULL CHECK (state IN ('executing', 'completed')), "
                "result_json TEXT)"
            )

    async def execute_once(
        self, request: LocalToolRequest, execute: ToolExecutor
    ) -> ToolResult:
        """Run once, persist outcome, and return the same result on retry."""

        if request.device_id != self._device_id:
            return make_tool_result(request, status="denied", error_code="DEVICE_SCOPE_REJECTED")
        request_json = json.dumps(
            request.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        async with self._lock:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT request_json, state, result_json FROM local_tool_inbox "
                    "WHERE action_id = ?",
                    (str(request.action_id),),
                ).fetchone()
                if row is not None:
                    if row[0] != request_json:
                        return make_tool_result(
                            request, status="denied", error_code="IDEMPOTENCY_CONFLICT"
                        )
                    if row[2] is not None:
                        return ToolResult.model_validate_json(row[2])
                    uncertain = make_tool_result(
                        request,
                        status="needs_reconciliation",
                        error_code="DEVICE_RESTART_DURING_ACTION",
                    )
                    connection.execute(
                        "UPDATE local_tool_inbox SET state = 'completed', result_json = ? "
                        "WHERE action_id = ? AND result_json IS NULL",
                        (uncertain.model_dump_json(), str(request.action_id)),
                    )
                    return uncertain
                if request.expires_at <= datetime.now(timezone.utc):
                    expired = make_tool_result(
                        request, status="denied", error_code="LOCAL_TOOL_REQUEST_EXPIRED"
                    )
                    connection.execute(
                        "INSERT INTO local_tool_inbox "
                        "(action_id, request_json, state, result_json) "
                        "VALUES (?, ?, 'completed', ?)",
                        (str(request.action_id), request_json, expired.model_dump_json()),
                    )
                    return expired
                connection.execute(
                    "INSERT INTO local_tool_inbox (action_id, request_json, state) "
                    "VALUES (?, ?, 'executing')",
                    (str(request.action_id), request_json),
                )

        try:
            result = await execute(request)
        except asyncio.CancelledError:
            raise
        except Exception:
            result = make_tool_result(
                request,
                status="needs_reconciliation",
                error_code="LOCAL_TOOL_EXECUTION_UNCERTAIN",
            )
        if not _same_request(request, result):
            result = make_tool_result(
                request,
                status="needs_reconciliation",
                error_code="LOCAL_TOOL_RESULT_MISMATCH",
            )

        async with self._lock:
            with self._connection() as connection:
                connection.execute(
                    "UPDATE local_tool_inbox SET state = 'completed', result_json = ? "
                    "WHERE action_id = ? AND result_json IS NULL",
                    (result.model_dump_json(), str(request.action_id)),
                )
                row = connection.execute(
                    "SELECT result_json FROM local_tool_inbox WHERE action_id = ?",
                    (str(request.action_id),),
                ).fetchone()
                if row is None or row[0] is None:
                    raise RuntimeError("local tool result was not persisted")
                return ToolResult.model_validate_json(row[0])

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._database_path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


def _same_request(request: LocalToolRequest, result: ToolResult) -> bool:
    return all(
        getattr(request, field) == getattr(result, field)
        for field in (
            "action_id",
            "trace_id",
            "device_id",
            "conversation_id",
            "session_id",
            "delegation_id",
            "context_revision",
            "job_id",
            "turn_id",
        )
    )


def make_tool_result(
    request: LocalToolRequest,
    *,
    status: ToolStatus,
    error_code: str | None = None,
    verified: bool = False,
    observed_state: dict[str, object] | None = None,
    retryable: bool = False,
) -> ToolResult:
    return ToolResult(
        action_id=request.action_id,
        trace_id=request.trace_id,
        device_id=request.device_id,
        conversation_id=request.conversation_id,
        session_id=request.session_id,
        delegation_id=request.delegation_id,
        context_revision=request.context_revision,
        job_id=request.job_id,
        turn_id=request.turn_id,
        status=status,
        verified=verified,
        observed_state=observed_state,
        error_code=error_code,
        retryable=retryable,
        completed_at=datetime.now(timezone.utc),
    )
