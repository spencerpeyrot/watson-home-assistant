from __future__ import annotations

import asyncio
import json
import os
import stat
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from starlette.websockets import WebSocketState

from contracts.models import TimerEventAck, TimerFiredEvent
from device.timer_events import (
    TimerEventInbox,
    TimerEventSocket,
    timer_event_websocket_url,
)
from service.auth import (
    AuthenticatedPrincipal,
    BearerCredential,
    CredentialAuthenticator,
)
from service.durable_api import create_durable_app
from service.device_events import DeviceEventHub, DeviceEventUnavailable


EVENT_ID = UUID("00000000-0000-4000-8000-000000000a01")
EVENT = TimerFiredEvent(
    event_id=EVENT_ID,
    timer_id=EVENT_ID,
    device_id="cube-01",
    due_at=datetime.now(timezone.utc),
    fired_at=datetime.now(timezone.utc),
    message="Tea is ready.",
    attempt_count=1,
)


class FakeClientConnection:
    def __init__(self, messages: list[str | bytes]) -> None:
        self.messages = list(messages)
        self.sent: list[str | bytes] = []

    async def recv(self) -> str | bytes:
        return self.messages.pop(0)

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def close(self) -> None:
        return None


class FakeServerWebSocket:
    def __init__(self) -> None:
        self.application_state = WebSocketState.CONNECTING
        self.accepted = asyncio.Event()
        self.receiving = asyncio.Event()
        self.sent_event = asyncio.Event()
        self.sent: list[str] = []
        self.inbound: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.close_code: int | None = None

    async def accept(self) -> None:
        self.application_state = WebSocketState.CONNECTED
        self.accepted.set()

    async def receive(self) -> dict[str, object]:
        self.receiving.set()
        return await self.inbound.get()

    async def send_text(self, message: str) -> None:
        self.sent.append(message)
        self.sent_event.set()

    async def close(self, *, code: int) -> None:
        self.close_code = code
        self.application_state = WebSocketState.DISCONNECTED


class TimerEventTransportTests(unittest.IsolatedAsyncioTestCase):
    def test_inbox_database_is_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "timer-inbox.sqlite3"
            database_path.touch(mode=0o644)
            os.chmod(database_path, 0o644)

            TimerEventInbox(database_path)

            self.assertEqual(stat.S_IMODE(database_path.stat().st_mode), 0o600)

    def test_socket_url_quotes_device_and_requires_tls_remotely(self) -> None:
        self.assertEqual(
            timer_event_websocket_url("https://gateway.example/base/", "cube 01"),
            "wss://gateway.example/base/v1/devices/cube%2001/events",
        )
        self.assertEqual(
            timer_event_websocket_url("http://127.0.0.1:8080", "cube-01"),
            "ws://127.0.0.1:8080/v1/devices/cube-01/events",
        )
        with self.assertRaises(ValueError):
            timer_event_websocket_url("http://gateway.example", "cube-01")
        with self.assertRaises(ValueError):
            timer_event_websocket_url("https://owner:secret@gateway.example", "cube-01")
        with self.assertRaises(ValueError):
            timer_event_websocket_url("https://gateway.example", "../admin")

    async def test_inbox_retries_pending_and_deduplicates_completed_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inbox = TimerEventInbox(Path(temporary) / "timer-inbox.sqlite3")
            calls = 0

            async def fail_first(_event: TimerFiredEvent) -> None:
                nonlocal calls
                calls += 1
                raise OSError("notification unavailable")

            with self.assertRaises(OSError):
                await inbox.present_once(EVENT, fail_first)

            async def notify(_event: TimerFiredEvent) -> None:
                nonlocal calls
                calls += 1

            await inbox.present_once(EVENT, notify)
            await inbox.present_once(EVENT, notify)

            with closing(
                sqlite3.connect(Path(temporary) / "timer-inbox.sqlite3")
            ) as connection:
                with connection:
                    state = connection.execute(
                        "SELECT state FROM timer_event_inbox WHERE event_id = ?",
                        (str(EVENT_ID),),
                    ).fetchone()[0]
            self.assertEqual(calls, 2)
            self.assertEqual(state, "delivered")

    async def test_device_socket_presents_then_acknowledges_and_duplicate_skips(self) -> None:
        wire_event = json.dumps(
            {"type": "timer_fired", "payload": EVENT.model_dump(mode="json")}
        )
        connection = FakeClientConnection([wire_event, wire_event])
        socket = TimerEventSocket(connection, device_id="cube-01")  # type: ignore[arg-type]
        notifications: list[UUID] = []

        async def notify(event: TimerFiredEvent) -> None:
            notifications.append(event.event_id)

        with tempfile.TemporaryDirectory() as temporary:
            inbox = TimerEventInbox(Path(temporary) / "timer-inbox.sqlite3")
            await socket.process_next(inbox, notify)
            await socket.process_next(inbox, notify)

        acknowledgements = [json.loads(message) for message in connection.sent]
        self.assertEqual(notifications, [EVENT_ID])
        self.assertEqual(len(acknowledgements), 2)
        self.assertTrue(
            all(
                TimerEventAck.model_validate(item["payload"]).event_id == EVENT_ID
                for item in acknowledgements
            )
        )

    async def test_gateway_waits_for_matching_device_ack(self) -> None:
        hub = DeviceEventHub(acknowledgement_timeout_seconds=1)
        websocket = FakeServerWebSocket()
        serving = asyncio.create_task(hub.serve(websocket, device_id="cube-01"))  # type: ignore[arg-type]
        await asyncio.wait_for(websocket.receiving.wait(), timeout=1)
        delivery = asyncio.create_task(hub.deliver(EVENT))
        await asyncio.wait_for(websocket.sent_event.wait(), timeout=1)
        self.assertEqual(len(websocket.sent), 1)
        envelope = json.loads(websocket.sent[0])
        self.assertEqual(envelope["type"], "timer_fired")
        self.assertEqual(envelope["payload"]["event_id"], str(EVENT_ID))
        self.assertFalse(delivery.done())

        ack = TimerEventAck(event_id=EVENT_ID)
        await websocket.inbound.put(
            {
                "type": "websocket.receive",
                "text": json.dumps({"type": "timer_ack", "payload": ack.model_dump(mode="json")}),
            }
        )
        await asyncio.wait_for(delivery, timeout=1)
        await websocket.inbound.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(serving, timeout=1)

    async def test_gateway_rejects_delivery_without_connected_device(self) -> None:
        hub = DeviceEventHub()
        with self.assertRaises(DeviceEventUnavailable):
            await hub.deliver(EVENT)


class TimerEventRouteAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        owner_id = UUID("00000000-0000-4000-8000-000000000a02")
        self.authenticator = CredentialAuthenticator(
            [
                BearerCredential(
                    token="o" * 48,
                    principal=AuthenticatedPrincipal(role="owner", owner_id=owner_id),
                ),
                BearerCredential(
                    token="d" * 48,
                    principal=AuthenticatedPrincipal(
                        role="device", owner_id=owner_id, device_id="cube-01"
                    ),
                ),
            ]
        )
        self.app = create_durable_app(
            store=object(),  # type: ignore[arg-type]
            authenticate=self.authenticator.authenticate_http,
            authenticate_websocket=self.authenticator.authenticate_websocket,
            device_event_hub=DeviceEventHub(),
        )

    def test_owner_and_wrong_device_cannot_open_device_event_socket(self) -> None:
        with TestClient(self.app) as client:
            for device_id, token in (("cube-01", "o" * 48), ("cube-02", "d" * 48)):
                with self.subTest(device_id=device_id):
                    with self.assertRaises(WebSocketDisconnect):
                        with client.websocket_connect(
                            f"/v1/devices/{device_id}/events",
                            headers={"Authorization": f"Bearer {token}"},
                        ):
                            pass

    def test_authenticated_device_socket_rejects_malformed_ack(self) -> None:
        with TestClient(self.app) as client:
            with client.websocket_connect(
                "/v1/devices/cube-01/events",
                headers={"Authorization": f"Bearer {'d' * 48}"},
            ) as websocket:
                websocket.send_json({"type": "wrong"})
                with self.assertRaises(WebSocketDisconnect) as caught:
                    websocket.receive_json()
        self.assertEqual(caught.exception.code, 1008)


if __name__ == "__main__":
    unittest.main()
