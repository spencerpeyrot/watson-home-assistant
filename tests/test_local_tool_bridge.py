from __future__ import annotations

import asyncio
import json
import os
import stat
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import httpx
from starlette.websockets import WebSocketState

from contracts.models import LocalToolRequest, TimerFiredEvent, ToolResult
from device.home_assistant import PiHomeAssistantExecutor
from device.local_tools import LocalToolInbox, make_tool_result
from device.timer_events import TimerEventInbox, TimerEventSocket
from service.device_events import DeviceEventHub, DeviceEventUnavailable
from service.home_assistant import HomeAssistantClient, HomeAssistantLocalBridge
from service.tool_policy import ActionClaim, LocalToolContext, ToolPolicyRegistry


DEVICE_ID = "cube-01"
ACTION_ID = UUID("00000000-0000-4000-8000-000000000a11")
TRACE_ID = UUID("00000000-0000-4000-8000-000000000a12")
CONVERSATION_ID = UUID("00000000-0000-4000-8000-000000000a13")
SESSION_ID = UUID("00000000-0000-4000-8000-000000000a14")
TURN_ID = UUID("00000000-0000-4000-8000-000000000a15")
JOB_ID = UUID("00000000-0000-4000-8000-000000000a16")


def make_request(*, expires_in: int = 60) -> LocalToolRequest:
    now = datetime.now(timezone.utc)
    return LocalToolRequest(
        action_id=ACTION_ID,
        trace_id=TRACE_ID,
        device_id=DEVICE_ID,
        conversation_id=CONVERSATION_ID,
        session_id=SESSION_ID,
        delegation_id="live-delegation-bridge",
        job_id=JOB_ID,
        turn_id=TURN_ID,
        context_revision=2,
        tool_name="set_light_state",
        payload={"entity_id": "light.living_room", "state": "on"},
        issued_at=now - timedelta(seconds=5),
        expires_at=now + timedelta(seconds=expires_in),
    )


def verified_result(request: LocalToolRequest) -> ToolResult:
    return make_tool_result(
        request,
        status="succeeded",
        verified=True,
        observed_state={"entity_id": "light.living_room", "state": "on"},
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
        self.sent_event = asyncio.Event()
        self.sent: list[str] = []
        self.inbound: asyncio.Queue[dict[str, object]] = asyncio.Queue()

    async def accept(self) -> None:
        self.application_state = WebSocketState.CONNECTED
        self.accepted.set()

    async def send_text(self, text: str) -> None:
        self.sent.append(text)
        self.sent_event.set()

    async def receive(self) -> dict[str, object]:
        return await self.inbound.get()

    async def close(self, code: int = 1000) -> None:
        self.application_state = WebSocketState.DISCONNECTED


class LocalToolInboxTests(unittest.IsolatedAsyncioTestCase):
    async def test_database_is_private_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "tools.sqlite"
            database_path.touch(mode=0o644)
            os.chmod(database_path, 0o644)

            LocalToolInbox(database_path, device_id=DEVICE_ID)

            self.assertEqual(stat.S_IMODE(database_path.stat().st_mode), 0o600)
            target = Path(directory) / "target.sqlite"
            target.touch()
            os.chmod(target, 0o644)
            link = Path(directory) / "link.sqlite"
            link.symlink_to(target)
            with self.assertRaises(OSError):
                LocalToolInbox(link, device_id=DEVICE_ID)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

    async def test_completed_action_replays_persisted_result_without_reexecution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inbox = LocalToolInbox(Path(directory) / "tools.sqlite", device_id=DEVICE_ID)
            request = make_request().model_copy(
                update={"action_id": UUID("00000000-0000-4000-8000-000000000a17")}
            )
            calls = 0

            async def execute(item: LocalToolRequest) -> ToolResult:
                nonlocal calls
                calls += 1
                return verified_result(item)

            first = await inbox.execute_once(request, execute)
            second = await inbox.execute_once(request, execute)

            self.assertEqual(first, second)
            self.assertEqual(first.status, "succeeded")
            self.assertEqual(calls, 1)

    async def test_interrupted_action_becomes_uncertain_and_never_reexecutes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inbox = LocalToolInbox(Path(directory) / "tools.sqlite", device_id=DEVICE_ID)
            request = make_request()
            calls = 0

            async def interrupted(_item: LocalToolRequest) -> ToolResult:
                nonlocal calls
                calls += 1
                raise asyncio.CancelledError

            with self.assertRaises(asyncio.CancelledError):
                await inbox.execute_once(request, interrupted)

            async def must_not_run(_item: LocalToolRequest) -> ToolResult:
                nonlocal calls
                calls += 1
                return verified_result(request)

            result = await inbox.execute_once(request, must_not_run)
            self.assertEqual(result.status, "needs_reconciliation")
            self.assertEqual(result.error_code, "DEVICE_RESTART_DURING_ACTION")
            self.assertEqual(calls, 1)

    async def test_expired_request_and_action_id_conflict_do_not_execute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inbox = LocalToolInbox(Path(directory) / "tools.sqlite", device_id=DEVICE_ID)
            expired = make_request(expires_in=-1)
            calls = 0

            async def execute(item: LocalToolRequest) -> ToolResult:
                nonlocal calls
                calls += 1
                return verified_result(item)

            result = await inbox.execute_once(expired, execute)
            self.assertEqual(result.status, "denied")
            self.assertEqual(result.error_code, "LOCAL_TOOL_REQUEST_EXPIRED")
            request = make_request().model_copy(
                update={"action_id": UUID("00000000-0000-4000-8000-000000000a18")}
            )
            await inbox.execute_once(request, execute)
            changed = request.model_copy(
                update={"payload": {"entity_id": "light.office", "state": "off"}}
            )
            conflict = await inbox.execute_once(changed, execute)
            self.assertEqual(conflict.status, "denied")
            self.assertEqual(conflict.error_code, "IDEMPOTENCY_CONFLICT")
            self.assertEqual(calls, 1)


class LocalToolTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_device_event_socket_executes_and_returns_local_tool_result(self) -> None:
        request = make_request()
        connection = FakeClientConnection(
            [
                json.dumps(
                    {
                        "type": "local_tool_request",
                        "payload": request.model_dump(mode="json"),
                    }
                )
            ]
        )
        socket = TimerEventSocket(connection, device_id=DEVICE_ID)  # type: ignore[arg-type]

        async def notify(_event: TimerFiredEvent) -> None:
            raise AssertionError("timer callback must not run for tool request")

        async def execute(item: LocalToolRequest) -> ToolResult:
            return verified_result(item)

        with tempfile.TemporaryDirectory() as directory:
            result = await socket.process_next(
                TimerEventInbox(Path(directory) / "timers.sqlite"),
                notify,
                local_tool_inbox=LocalToolInbox(
                    Path(directory) / "tools.sqlite", device_id=DEVICE_ID
                ),
                execute_tool=execute,
            )

        self.assertIsInstance(result, ToolResult)
        assert isinstance(result, ToolResult)
        self.assertEqual(result.action_id, request.action_id)
        sent = json.loads(connection.sent[0])
        self.assertEqual(sent["type"], "local_tool_result")
        self.assertEqual(sent["payload"]["status"], "succeeded")

    async def test_gateway_correlates_result_to_exact_local_tool_request(self) -> None:
        hub = DeviceEventHub(acknowledgement_timeout_seconds=1)
        websocket = FakeServerWebSocket()
        request = make_request()
        serving = asyncio.create_task(hub.serve(websocket, device_id=DEVICE_ID))  # type: ignore[arg-type]
        await websocket.accepted.wait()
        delivery = asyncio.create_task(hub.deliver_local_tool(request))
        await websocket.sent_event.wait()
        outbound = json.loads(websocket.sent[0])
        self.assertEqual(outbound["type"], "local_tool_request")
        response = verified_result(request)
        await websocket.inbound.put(
            {
                "type": "websocket.receive",
                "text": json.dumps(
                    {
                        "type": "local_tool_result",
                        "payload": response.model_dump(mode="json"),
                    }
                ),
            }
        )

        actual = await delivery
        self.assertEqual(actual, response)
        await websocket.inbound.put({"type": "websocket.disconnect"})
        await serving

    async def test_gateway_rejects_local_tool_result_with_wrong_revision(self) -> None:
        hub = DeviceEventHub(acknowledgement_timeout_seconds=1)
        websocket = FakeServerWebSocket()
        request = make_request()
        serving = asyncio.create_task(hub.serve(websocket, device_id=DEVICE_ID))  # type: ignore[arg-type]
        await websocket.accepted.wait()
        delivery = asyncio.create_task(hub.deliver_local_tool(request))
        await websocket.sent_event.wait()
        bad_result = verified_result(request).model_copy(update={"context_revision": 3})
        await websocket.inbound.put(
            {
                "type": "websocket.receive",
                "text": json.dumps(
                    {
                        "type": "local_tool_result",
                        "payload": bad_result.model_dump(mode="json"),
                    }
                ),
            }
        )

        with self.assertRaises(DeviceEventUnavailable):
            await delivery
        await serving


class PiHomeAssistantExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_executor_verifies_light_state_after_write(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json=[{"entity_id": "light.living_room"}])
            return httpx.Response(
                200,
                json={"entity_id": "light.living_room", "state": "on"},
            )

        client = HomeAssistantClient(
            "http://127.0.0.1:8123",
            "a" * 40,
            allowed_light_entities=frozenset({"light.living_room"}),
            transport=httpx.MockTransport(respond),
        )
        try:
            result = await PiHomeAssistantExecutor(client)(make_request())
        finally:
            await client.close()

        self.assertEqual(result.status, "succeeded")
        self.assertTrue(result.verified)
        self.assertEqual(result.observed_state["observed_state"], "on")


class FakeActionLedger:
    def __init__(self) -> None:
        self.claims: list[dict[str, object]] = []
        self.completed: list[dict[str, object]] = []
        self.uncertain: list[UUID] = []

    async def claim_action(self, **values):
        self.claims.append(values)
        return ActionClaim(outcome="execute", action_id=values["action_id"], state="claimed")

    async def complete_action(self, **values) -> None:
        self.completed.append(values)

    async def mark_action_uncertain(self, *, owner_id: UUID, action_id: UUID) -> None:
        self.uncertain.append(action_id)


class HomeAssistantCloudBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_scope_and_unlisted_light_never_reach_ledger_or_device(self) -> None:
        delivered: list[LocalToolRequest] = []

        async def deliver(item: LocalToolRequest) -> ToolResult:
            delivered.append(item)
            return verified_result(item)

        registry = ToolPolicyRegistry()
        HomeAssistantLocalBridge(
            allowed_light_entities=frozenset({"light.living_room"}),
            deliver=deliver,
        ).register_tools(registry)
        ledger = FakeActionLedger()
        now = datetime.now(timezone.utc)
        context = LocalToolContext(
            device_id=DEVICE_ID,
            conversation_id=CONVERSATION_ID,
            session_id=SESSION_ID,
            delegation_id="live-delegation-bridge",
            turn_id=TURN_ID,
            expires_at=now + timedelta(seconds=30),
        )
        expired_context = LocalToolContext(
            device_id=DEVICE_ID,
            conversation_id=CONVERSATION_ID,
            session_id=SESSION_ID,
            delegation_id="live-delegation-bridge",
            turn_id=TURN_ID,
            expires_at=now - timedelta(seconds=1),
        )
        common = {
            "owner_id": UUID("00000000-0000-4000-8000-000000000a19"),
            "tool_name": "set_light_state",
            "payload": {"entity_id": "light.living_room", "state": "on"},
            "action_id": ACTION_ID,
            "trace_id": TRACE_ID,
            "context_revision": 2,
            "idempotency_key": "action/bridge/denied",
            "ledger": ledger,
            "job_id": JOB_ID,
            "now": now,
        }

        missing_scope = await registry.dispatch(**common)
        expired_scope = await registry.dispatch(
            **{**common, "local_context": expired_context}
        )
        unlisted_light = await registry.dispatch(
            **{
                **common,
                "payload": {"entity_id": "light.office", "state": "off"},
                "local_context": context,
            }
        )

        self.assertEqual(missing_scope.status, "denied")
        self.assertEqual(expired_scope.status, "denied")
        self.assertEqual(unlisted_light.status, "denied")
        self.assertEqual(ledger.claims, [])
        self.assertEqual(delivered, [])

    async def test_policy_and_ledger_dispatch_to_pilocal_tool_with_exact_context(self) -> None:
        request = make_request()
        received: list[LocalToolRequest] = []

        async def deliver(item: LocalToolRequest) -> ToolResult:
            received.append(item)
            return verified_result(item)

        registry = ToolPolicyRegistry()
        HomeAssistantLocalBridge(
            allowed_light_entities=frozenset({"light.living_room"}),
            deliver=deliver,
        ).register_tools(registry)
        ledger = FakeActionLedger()
        issued_at = datetime.now(timezone.utc)
        local_context = LocalToolContext(
            device_id=DEVICE_ID,
            conversation_id=CONVERSATION_ID,
            session_id=SESSION_ID,
            delegation_id="live-delegation-bridge",
            turn_id=TURN_ID,
            expires_at=issued_at + timedelta(seconds=30),
        )

        result = await registry.dispatch(
            owner_id=UUID("00000000-0000-4000-8000-000000000a19"),
            tool_name="set_light_state",
            payload={"entity_id": "light.living_room", "state": "on"},
            action_id=ACTION_ID,
            trace_id=TRACE_ID,
            context_revision=2,
            idempotency_key="action/bridge/1",
            ledger=ledger,
            job_id=JOB_ID,
            local_context=local_context,
            now=issued_at,
        )

        self.assertEqual(result.status, "succeeded")
        self.assertTrue(result.verified)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].action_id, ACTION_ID)
        self.assertEqual(received[0].trace_id, TRACE_ID)
        self.assertEqual(received[0].session_id, SESSION_ID)
        self.assertEqual(received[0].context_revision, 2)
        self.assertEqual(len(ledger.claims), 1)
        self.assertEqual(len(ledger.completed), 1)
        self.assertEqual(ledger.uncertain, [])

    async def test_transport_loss_marks_claim_uncertain_and_never_completes(self) -> None:
        async def disconnected(_item: LocalToolRequest) -> ToolResult:
            raise DeviceEventUnavailable("device disconnected")

        registry = ToolPolicyRegistry()
        HomeAssistantLocalBridge(
            allowed_light_entities=frozenset({"light.living_room"}),
            deliver=disconnected,
        ).register_tools(registry)
        ledger = FakeActionLedger()
        now = datetime.now(timezone.utc)

        with self.assertRaises(DeviceEventUnavailable):
            await registry.dispatch(
                owner_id=UUID("00000000-0000-4000-8000-000000000a19"),
                tool_name="set_light_state",
                payload={"entity_id": "light.living_room", "state": "off"},
                action_id=ACTION_ID,
                trace_id=TRACE_ID,
                context_revision=2,
                idempotency_key="action/bridge/2",
                ledger=ledger,
                job_id=JOB_ID,
                local_context=LocalToolContext(
                    device_id=DEVICE_ID,
                    conversation_id=CONVERSATION_ID,
                    session_id=SESSION_ID,
                    delegation_id="live-delegation-bridge",
                    turn_id=TURN_ID,
                    expires_at=now + timedelta(seconds=30),
                ),
                now=now,
            )

        self.assertEqual(ledger.uncertain, [ACTION_ID])
        self.assertEqual(ledger.completed, [])


if __name__ == "__main__":
    unittest.main()
