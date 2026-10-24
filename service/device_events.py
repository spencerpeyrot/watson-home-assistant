"""Authenticated device event sockets with acknowledgement-bound delivery."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from uuid import UUID

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from contracts.models import LocalToolRequest, TimerEventAck, TimerFiredEvent, ToolResult


class DeviceEventUnavailable(RuntimeError):
    """Device is disconnected or did not acknowledge an event in time."""


@dataclass
class _DeviceConnection:
    device_id: str
    websocket: WebSocket
    pending: dict[UUID, _PendingDelivery] = field(default_factory=dict)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class _PendingDelivery:
    expected_type: str
    request: TimerFiredEvent | LocalToolRequest
    future: asyncio.Future[TimerEventAck | ToolResult]


class DeviceEventHub:
    """Route timers and correlated local actions over a device WebSocket."""

    MAX_ENVELOPE_BYTES = 16_384

    def __init__(self, *, acknowledgement_timeout_seconds: float = 15.0) -> None:
        if acknowledgement_timeout_seconds <= 0:
            raise ValueError("event acknowledgement timeout must be positive")
        self._acknowledgement_timeout_seconds = acknowledgement_timeout_seconds
        self._lock = asyncio.Lock()
        self._connections: dict[str, _DeviceConnection] = {}

    async def serve(self, websocket: WebSocket, *, device_id: str) -> None:
        """Serve an already-authenticated, device-scoped socket."""

        connection = _DeviceConnection(device_id=device_id, websocket=websocket)
        await websocket.accept()
        async with self._lock:
            replaced = self._connections.get(device_id)
            self._connections[device_id] = connection
            if replaced is not None:
                self._fail_pending(replaced, "device event socket was replaced")
        if replaced is not None:
            await _close_quietly(replaced.websocket, code=1012)

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                raw = message.get("text")
                if not isinstance(raw, str):
                    await _close_quietly(websocket, code=1008)
                    return
                try:
                    response_type, response = _parse_device_response(raw)
                except ValueError:
                    await _close_quietly(websocket, code=1008)
                    return
                correlation_id = (
                    response.event_id
                    if isinstance(response, TimerEventAck)
                    else response.action_id
                )
                pending = connection.pending.get(correlation_id)
                if pending is None or pending.future.done():
                    continue
                if response_type != pending.expected_type or not _matches_request(
                    pending.request, response
                ):
                    pending.future.set_exception(
                        DeviceEventUnavailable("device response did not match request")
                    )
                    await _close_quietly(websocket, code=1008)
                    return
                pending.future.set_result(response)
        finally:
            async with self._lock:
                if self._connections.get(device_id) is connection:
                    del self._connections[device_id]
                self._fail_pending(connection, "device event socket disconnected")
            await _close_quietly(websocket, code=1000)

    async def deliver(self, event: TimerFiredEvent) -> None:
        """Send timer and wait for matching device acknowledgement."""

        response = await self._deliver(
            event,
            response_type="timer_ack",
            envelope_type="timer_fired",
            correlation_id=event.event_id,
        )
        if not isinstance(response, TimerEventAck):
            raise DeviceEventUnavailable("device returned the wrong timer response")

    async def deliver_local_tool(self, request: LocalToolRequest) -> ToolResult:
        """Send an action to its Pi and wait for the fully correlated result."""

        response = await self._deliver(
            request,
            response_type="local_tool_result",
            envelope_type="local_tool_request",
            correlation_id=request.action_id,
        )
        if not isinstance(response, ToolResult):
            raise DeviceEventUnavailable("device returned the wrong local-tool response")
        return response

    async def _deliver(
        self,
        request: TimerFiredEvent | LocalToolRequest,
        *,
        response_type: str,
        envelope_type: str,
        correlation_id: UUID,
    ) -> TimerEventAck | ToolResult:
        """Send one request and correlate its response to the exact message."""

        loop = asyncio.get_running_loop()
        async with self._lock:
            connection = self._connections.get(request.device_id)
            if connection is None:
                raise DeviceEventUnavailable("device is not connected")
            if connection.pending:
                raise DeviceEventUnavailable("device already has an event in flight")
            future: asyncio.Future[TimerEventAck | ToolResult] = loop.create_future()
            connection.pending[correlation_id] = _PendingDelivery(
                expected_type=response_type,
                request=request,
                future=future,
            )

        envelope = {
            "type": envelope_type,
            "payload": request.model_dump(mode="json"),
        }
        serialized = json.dumps(envelope, separators=(",", ":"))
        if len(serialized.encode("utf-8")) > self.MAX_ENVELOPE_BYTES:
            async with self._lock:
                connection.pending.pop(correlation_id, None)
            future.cancel()
            raise DeviceEventUnavailable("device request exceeded message size limit")
        try:
            async with connection.send_lock:
                await connection.websocket.send_text(serialized)
            response = await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self._acknowledgement_timeout_seconds,
            )
            return response
        except TimeoutError:
            raise DeviceEventUnavailable("device response timed out") from None
        except Exception as error:
            raise DeviceEventUnavailable(
                f"device request failed ({type(error).__name__})"
            ) from None
        finally:
            async with self._lock:
                pending = connection.pending.get(correlation_id)
                if pending is not None and pending.future is future:
                    del connection.pending[correlation_id]
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                future.exception()

    @staticmethod
    def _fail_pending(connection: _DeviceConnection, reason: str) -> None:
        for pending in connection.pending.values():
            if not pending.future.done():
                pending.future.set_exception(DeviceEventUnavailable(reason))


def _parse_device_response(raw: str) -> tuple[str, TimerEventAck | ToolResult]:
    try:
        if len(raw.encode("utf-8")) > DeviceEventHub.MAX_ENVELOPE_BYTES:
            raise ValueError
        envelope = json.loads(raw)
        if not isinstance(envelope, dict) or set(envelope) != {"type", "payload"}:
            raise ValueError
        if envelope["type"] == "timer_ack":
            return "timer_ack", TimerEventAck.model_validate(envelope["payload"])
        if envelope["type"] == "local_tool_result":
            return "local_tool_result", ToolResult.model_validate(envelope["payload"])
        raise ValueError
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("device response did not match the contract") from None


def _matches_request(
    request: TimerFiredEvent | LocalToolRequest,
    response: TimerEventAck | ToolResult,
) -> bool:
    if isinstance(request, TimerFiredEvent) and isinstance(response, TimerEventAck):
        return response.event_id == request.event_id
    if isinstance(request, LocalToolRequest) and isinstance(response, ToolResult):
        return all(
            getattr(request, field) == getattr(response, field)
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
    return False


async def _close_quietly(websocket: WebSocket, *, code: int) -> None:
    if websocket.application_state == WebSocketState.CONNECTED:
        try:
            await websocket.close(code=code)
        except Exception:
            pass
