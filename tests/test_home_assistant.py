from __future__ import annotations

import json
import unittest
from typing import Any
from uuid import UUID

import httpx
from pydantic import BaseModel

from service.home_assistant import HomeAssistantClient
from service.tool_policy import ActionClaim, ToolOutcome, ToolPolicyRegistry


OWNER = UUID("00000000-0000-4000-8000-000000000991")
ACTION = UUID("00000000-0000-4000-8000-000000000992")
TRACE = UUID("00000000-0000-4000-8000-000000000993")


class FakeLedger:
    def __init__(self) -> None:
        self.claims = 0
        self.completed: list[Any] = []
        self.uncertain: list[UUID] = []

    async def claim_action(self, **values) -> ActionClaim:
        self.claims += 1
        return ActionClaim("execute", values["action_id"], "executing")

    async def complete_action(self, *, owner_id, action_id, result, verified) -> None:
        assert verified
        self.completed.append((owner_id, action_id, result))

    async def mark_action_uncertain(self, *, owner_id, action_id) -> None:
        self.uncertain.append(action_id)


class HomeAssistantToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_entity_allowlist_and_verified_light_write(self) -> None:
        calls: list[tuple[str, str, Any]] = []
        states = {"light.desk": "off"}

        async def handle(request: httpx.Request) -> httpx.Response:
            calls.append((request.method, request.url.path, request.content))
            if request.method == "GET":
                entity_id = request.url.path.rsplit("/", 1)[-1]
                if entity_id not in states:
                    return httpx.Response(404)
                return httpx.Response(200, json={"entity_id": entity_id, "state": states[entity_id]})
            if request.method == "POST":
                payload = json.loads(request.content)
                states[payload["entity_id"]] = "on" if request.url.path.endswith("turn_on") else "off"
                return httpx.Response(200, json=[])
            return httpx.Response(405)

        client = HomeAssistantClient(
            "https://ha.example",
            "h" * 40,
            allowed_light_entities=frozenset({"light.desk"}),
            transport=httpx.MockTransport(handle),
        )
        registry = ToolPolicyRegistry()
        client.register_tools(registry)
        ledger = FakeLedger()
        try:
            rejected = await registry.dispatch(
                owner_id=OWNER,
                tool_name="set_light_state",
                payload={"entity_id": "light.kitchen", "state": "on"},
                action_id=ACTION,
                trace_id=TRACE,
                context_revision=0,
                idempotency_key="denied-light",
                ledger=ledger,
            )
            self.assertEqual(rejected.status, "denied")
            self.assertEqual(ledger.claims, 0)
            self.assertEqual(calls, [])

            read = await registry.dispatch(
                owner_id=OWNER,
                tool_name="get_light_state",
                payload={"entity_id": "light.desk"},
                action_id=ACTION,
                trace_id=TRACE,
                context_revision=0,
                idempotency_key="read-light",
                ledger=None,
            )
            self.assertEqual(read.status, "succeeded")
            self.assertEqual(read.result["observed_state"], "off")

            written = await registry.dispatch(
                owner_id=OWNER,
                tool_name="set_light_state",
                payload={"entity_id": "light.desk", "state": "on"},
                action_id=ACTION,
                trace_id=TRACE,
                context_revision=0,
                idempotency_key="turn-on-desk",
                ledger=ledger,
            )
            self.assertEqual(written.status, "succeeded")
            self.assertTrue(written.verified)
            self.assertEqual(written.result["observed_state"], "on")
            self.assertEqual(ledger.claims, 1)
            self.assertEqual(len(ledger.completed), 1)
            self.assertEqual([item[1] for item in calls], [
                "/api/states/light.desk",
                "/api/services/light/turn_on",
                "/api/states/light.desk",
            ])
        finally:
            await client.close()

    async def test_unverified_state_leaves_write_uncertain(self) -> None:
        async def handle(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json=[])
            return httpx.Response(200, json={"entity_id": "light.desk", "state": "off"})

        client = HomeAssistantClient(
            "http://localhost:8123",
            "h" * 40,
            allowed_light_entities=frozenset({"light.desk"}),
            transport=httpx.MockTransport(handle),
        )
        registry = ToolPolicyRegistry()
        client.register_tools(registry)
        ledger = FakeLedger()
        try:
            result = await registry.dispatch(
                owner_id=OWNER,
                tool_name="set_light_state",
                payload={"entity_id": "light.desk", "state": "on"},
                action_id=ACTION,
                trace_id=TRACE,
                context_revision=0,
                idempotency_key="unverified-light",
                ledger=ledger,
            )
            self.assertEqual(result.status, "needs_reconciliation")
            self.assertFalse(result.verified)
            self.assertEqual(ledger.uncertain, [ACTION])
            self.assertEqual(ledger.completed, [])
        finally:
            await client.close()

    def test_rejects_cleartext_remote_url_and_non_light_catalog(self) -> None:
        with self.assertRaisesRegex(ValueError, "unencrypted"):
            HomeAssistantClient(
                "http://ha.example",
                "h" * 40,
                allowed_light_entities=frozenset({"light.desk"}),
            )
        with self.assertRaisesRegex(ValueError, "allowlist"):
            HomeAssistantClient(
                "https://ha.example",
                "h" * 40,
                allowed_light_entities=frozenset({"switch.desk"}),
            )


if __name__ == "__main__":
    unittest.main()
