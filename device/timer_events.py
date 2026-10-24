"""Authenticated timer-event transport and durable receiver deduplication."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import UUID, uuid4

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

from contracts.models import LocalToolRequest, TimerEventAck, TimerFiredEvent, ToolResult
from device.local_tools import LocalToolInbox, ToolExecutor, make_tool_result
from device.reconnect import ReconnectSupervisor, RetryCallback
from device.transport import DeviceTransportError
from voice.observability import start_session


TimerNotification = Callable[[TimerFiredEvent], Awaitable[None]]
def timer_event_websocket_url(gateway_url: str, device_id: str) -> str:
    """Build the TLS-protected device event socket URL."""

    if not gateway_url.strip() or not device_id.strip():
        raise ValueError("gateway URL and device ID must not be blank")
    parsed = urlsplit(gateway_url.rstrip("/"))
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.hostname:
        raise ValueError("gateway URL must be an absolute HTTP(S)/WS(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("gateway URL must not include embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("gateway URL must not include a query or fragment")
    if device_id in {".", ".."} or "/" in device_id or "\\" in device_id:
        raise ValueError("device ID must be one URL path segment")
    local_host = parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme in {"http", "ws"} and not local_host:
        raise ValueError("unencrypted device transport is allowed only on loopback")
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}[
        parsed.scheme
    ]
    path = f"{parsed.path.rstrip('/')}/v1/devices/{quote(device_id, safe='')}/events"
    return urlunsplit((scheme, parsed.netloc, path, "", ""))


class TimerEventInbox:
    """Persist event IDs before presentation and acknowledge only afterward.

    A failed notification stays pending, so a gateway retry attempts it again.
    A completed event survives process restart and duplicate delivery.
    """

    def __init__(self, database_path: str | Path) -> None:
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
        self._lock = asyncio.Lock()
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS timer_event_inbox ("
                "event_id TEXT PRIMARY KEY, state TEXT NOT NULL "
                "CHECK (state IN ('pending', 'delivered')), "
                "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )

    async def present_once(
        self, event: TimerFiredEvent, notify: TimerNotification
    ) -> bool:
        """Run callback once after success; pending events retry after failure."""

        async with self._lock:
            with self._connection() as connection:
                connection.execute(
                    "INSERT INTO timer_event_inbox (event_id, state) VALUES (?, 'pending') "
                    "ON CONFLICT(event_id) DO NOTHING",
                    (str(event.event_id),),
                )
                row = connection.execute(
                    "SELECT state FROM timer_event_inbox WHERE event_id = ?",
                    (str(event.event_id),),
                ).fetchone()
            if row is not None and row[0] == "delivered":
                return False

            await notify(event)

            with self._connection() as connection:
                connection.execute(
                    "UPDATE timer_event_inbox SET state = 'delivered', "
                    "updated_at = CURRENT_TIMESTAMP WHERE event_id = ?",
                    (str(event.event_id),),
                )
            return True

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


class TimerEventSocket:
    """Single authenticated connection; caller owns reconnect and notification."""

    MAX_MESSAGE_BYTES = 16_384

    def __init__(self, connection: ClientConnection, *, device_id: str) -> None:
        self._connection = connection
        self._device_id = device_id

    @classmethod
    async def open(
        cls, gateway_url: str, device_id: str, device_token: str
    ) -> TimerEventSocket:
        if len(device_token) < 32:
            raise ValueError("device token must contain at least 32 characters")
        url = timer_event_websocket_url(gateway_url, device_id)
        try:
            connection = await connect(
                url,
                additional_headers={"Authorization": f"Bearer {device_token}"},
                open_timeout=10,
                ping_interval=20,
                ping_timeout=20,
                max_size=16_384,
                max_queue=8,
            )
        except (OSError, TimeoutError, WebSocketException) as error:
            raise DeviceTransportError(
                f"timer event connection failed ({type(error).__name__})"
            ) from None
        return cls(connection, device_id=device_id)

    async def receive(self) -> TimerFiredEvent | LocalToolRequest:
        """Read and validate one timer or local-tool request for this device."""

        try:
            raw = await self._connection.recv()
            if not isinstance(raw, str) or len(raw.encode("utf-8")) > self.MAX_MESSAGE_BYTES:
                raise ValueError
            envelope = json.loads(raw)
            if not isinstance(envelope, dict) or set(envelope) != {"type", "payload"}:
                raise ValueError
            if envelope["type"] == "timer_fired":
                event = TimerFiredEvent.model_validate(envelope["payload"])
                if event.device_id != self._device_id:
                    raise ValueError
                return event
            if envelope["type"] == "local_tool_request":
                request = LocalToolRequest.model_validate(envelope["payload"])
                if request.device_id != self._device_id:
                    raise ValueError
                return request
            raise ValueError
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise DeviceTransportError(
                f"gateway timer event did not match contract ({type(error).__name__})"
            ) from None
        except (WebSocketException, OSError) as error:
            raise DeviceTransportError(
                f"timer event receive failed ({type(error).__name__})"
            ) from None

    async def acknowledge(self, event_id: UUID) -> None:
        """Confirm durable local presentation to release the server timer lease."""

        envelope = {
            "type": "timer_ack",
            "payload": TimerEventAck(event_id=event_id).model_dump(mode="json"),
        }
        try:
            await self._connection.send(json.dumps(envelope, separators=(",", ":")))
        except (WebSocketException, OSError) as error:
            raise DeviceTransportError(
                f"timer event acknowledgement failed ({type(error).__name__})"
            ) from None

    async def send_tool_result(self, result: ToolResult) -> None:
        """Return a correlated local-tool result to the gateway."""

        if result.device_id != self._device_id:
            raise ValueError("local-tool result belongs to another device")
        envelope = {
            "type": "local_tool_result",
            "payload": result.model_dump(mode="json"),
        }
        serialized = json.dumps(envelope, separators=(",", ":"))
        if len(serialized.encode("utf-8")) > self.MAX_MESSAGE_BYTES:
            raise DeviceTransportError("local-tool result exceeded message size limit")
        try:
            await self._connection.send(serialized)
        except (WebSocketException, OSError) as error:
            raise DeviceTransportError(
                f"local-tool result send failed ({type(error).__name__})"
            ) from None

    async def process_next(
        self,
        inbox: TimerEventInbox,
        notify: TimerNotification,
        *,
        local_tool_inbox: LocalToolInbox | None = None,
        execute_tool: ToolExecutor | None = None,
    ) -> TimerFiredEvent | ToolResult:
        """Process one timer or action, then acknowledge its durable outcome."""

        message = await self.receive()
        if isinstance(message, LocalToolRequest):
            trace = start_session(
                uuid4(), "timer.receiver", action_id=message.action_id,
                request_trace_id=message.trace_id, voice_session_id=message.session_id,
                turn_id=message.turn_id, job_id=message.job_id,
                delegation_id=message.delegation_id, tool_name=message.tool_name,
            )
        else:
            trace = start_session(
                uuid4(), "timer.receiver", event_id=message.event_id,
                timer_id=message.timer_id, delivery_attempt=message.attempt_count,
                delivery_delay_ms=max(0, round((message.fired_at - message.due_at).total_seconds() * 1000)),
            )
        trace.update(status="running")
        stage = "execute"
        try:
            with trace.activate():
                if isinstance(message, LocalToolRequest):
                    if (local_tool_inbox is None) != (execute_tool is None):
                        raise ValueError("local-tool inbox and executor must be supplied together")
                    if local_tool_inbox is None or execute_tool is None:
                        result = make_tool_result(
                            message, status="denied", error_code="LOCAL_TOOL_NOT_CONFIGURED",
                        )
                    else:
                        with trace.operation("device.local_tool", action_id=message.action_id,
                                             tool_name=message.tool_name) as span:
                            result = await local_tool_inbox.execute_once(message, execute_tool)
                            span.update(
                                status="failed" if result.status in {"failed", "needs_reconciliation"}
                                else "rejected" if result.status == "denied" else "succeeded",
                                tool_status=result.status, error_code=result.error_code,
                                verified=result.verified, retryable=result.retryable,
                            )
                    stage = "send_tool_result"
                    with trace.operation("device.tool_result_send", action_id=message.action_id):
                        await self.send_tool_result(result)
                    trace.update(
                        status="failed" if result.status in {"failed", "needs_reconciliation"}
                        else "rejected" if result.status == "denied" else "succeeded",
                        stage="result_sent", tool_status=result.status,
                        error_code=result.error_code, verified=result.verified,
                    )
                    return result
                stage = "present"
                with trace.operation("timer.present", event_id=message.event_id, timer_id=message.timer_id):
                    presented = await inbox.present_once(message, notify)
                stage = "acknowledge"
                with trace.operation("timer.acknowledge", event_id=message.event_id):
                    await self.acknowledge(message.event_id)
                trace.update(status="succeeded", stage="acknowledged",
                             reason="duplicate_event" if not presented else "presented")
        except asyncio.CancelledError:
            trace.update(status="cancelled", failure_stage=stage, error_category="cancelled")
            raise
        except Exception as error:
            trace.update(status="failed", failure_stage=stage, error_type=type(error).__name__,
                         cause_type=type(error).__name__, error_category="device_event_failed")
            raise DeviceTransportError(
                f"timer notification callback failed ({type(error).__name__})"
            ) from None
        finally:
            trace.end()
        return message

    async def close(self) -> None:
        await self._connection.close()


async def run_timer_event_client(
    *,
    gateway_url: str,
    device_id: str,
    device_token: str,
    inbox: TimerEventInbox,
    notify: TimerNotification,
    stop_event: asyncio.Event,
    local_tool_inbox: LocalToolInbox | None = None,
    execute_tool: ToolExecutor | None = None,
    on_retry: RetryCallback | None = None,
) -> None:
    """Keep event socket connected; reconnect after network or callback failure."""

    async def attempt() -> None:
        trace = start_session(uuid4(), "timer.connection", component="timer-client")
        trace.update(status="running")
        socket = None
        stage = "connect"
        try:
            socket = await TimerEventSocket.open(gateway_url, device_id, device_token)
            stage = "receive_event"
            while not stop_event.is_set():
                receive = asyncio.create_task(
                    socket.process_next(
                        inbox,
                        notify,
                        local_tool_inbox=local_tool_inbox,
                        execute_tool=execute_tool,
                    ),
                    name="cube-device-event",
                )
                stopping = asyncio.create_task(
                    stop_event.wait(), name="cube-timer-stop"
                )
                try:
                    done, _ = await asyncio.wait(
                        (receive, stopping), return_when=asyncio.FIRST_COMPLETED
                    )
                    if stopping in done:
                        receive.cancel()
                        await asyncio.gather(receive, return_exceptions=True)
                        return
                    await receive
                finally:
                    for task in (receive, stopping):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(receive, stopping, return_exceptions=True)
            trace.update(status="succeeded")
        except asyncio.CancelledError:
            trace.update(status="cancelled", failure_stage=stage, error_category="cancelled")
            raise
        except Exception as error:
            trace.update(status="failed", failure_stage=stage, error_type=type(error).__name__,
                         cause_type=type(error).__name__, error_category="device_event_transport_failed")
            raise
        finally:
            if socket is not None:
                try:
                    await socket.close()
                except Exception as error:
                    trace.event("timer.socket_close_failed", status="failed",
                                error_type=type(error).__name__, error_category="cleanup_failed")
            trace.end()

    supervisor = ReconnectSupervisor(attempt, on_retry=on_retry)
    running = asyncio.create_task(supervisor.run(), name="cube-timer-reconnect")
    stopping = asyncio.create_task(stop_event.wait(), name="cube-timer-client-stop")
    try:
        done, _ = await asyncio.wait(
            (running, stopping), return_when=asyncio.FIRST_COMPLETED
        )
        if running in done:
            await running
    finally:
        supervisor.stop()
        if not running.done():
            await running
        if not stopping.done():
            stopping.cancel()
        await asyncio.gather(stopping, return_exceptions=True)
