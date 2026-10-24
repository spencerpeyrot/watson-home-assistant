from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from uuid import UUID

import httpx

from contracts.models import LocalToolRequest, ToolResult
from service.internal_tools import WorkerDeviceToolClient, WorkerToolBridgeError


class WorkerDeviceToolClientTests(unittest.IsolatedAsyncioTestCase):
    def request(self) -> LocalToolRequest:
        now = datetime.now(timezone.utc)
        return LocalToolRequest(
            action_id=UUID("00000000-0000-4000-8000-000000000721"),
            trace_id=UUID("00000000-0000-4000-8000-000000000722"),
            device_id="cube-01",
            conversation_id=UUID("00000000-0000-4000-8000-000000000723"),
            session_id=UUID("00000000-0000-4000-8000-000000000724"),
            delegation_id="delegation-internal-client",
            job_id=UUID("00000000-0000-4000-8000-000000000725"),
            turn_id=UUID("00000000-0000-4000-8000-000000000726"),
            context_revision=0,
            tool_name="set_light_state",
            payload={"entity_id": "light.desk", "state": "on"},
            issued_at=now,
            expires_at=now + timedelta(seconds=15),
        )

    def result(self, request: LocalToolRequest, **overrides) -> ToolResult:
        values = {
            "action_id": request.action_id,
            "trace_id": request.trace_id,
            "device_id": request.device_id,
            "conversation_id": request.conversation_id,
            "session_id": request.session_id,
            "delegation_id": request.delegation_id,
            "context_revision": request.context_revision,
            "job_id": request.job_id,
            "turn_id": request.turn_id,
            "status": "succeeded",
            "verified": True,
            "retryable": False,
        }
        values.update(overrides)
        return ToolResult(**values)

    async def test_posts_worker_auth_and_requires_correlated_result(self):
        calls = []
        request = self.request()

        async def handle(http_request: httpx.Request) -> httpx.Response:
            calls.append(http_request)
            result = self.result(request)
            return httpx.Response(200, json=result.model_dump(mode="json"))

        client = WorkerDeviceToolClient(
            "http://127.0.0.1:8000",
            "w" * 48,
            transport=httpx.MockTransport(handle),
        )
        try:
            result = await client.deliver(request)
        finally:
            await client.close()

        self.assertTrue(result.verified)
        self.assertEqual(calls[0].url.path, "/internal/v1/local-tools")
        self.assertEqual(calls[0].headers["authorization"], "Bearer " + "w" * 48)

    async def test_rejects_mismatched_result_and_missing_job(self):
        request = self.request()

        async def handle(_http_request: httpx.Request) -> httpx.Response:
            result = self.result(
                request,
                session_id=UUID("00000000-0000-4000-8000-000000000799"),
            )
            return httpx.Response(200, json=result.model_dump(mode="json"))

        client = WorkerDeviceToolClient(
            "https://api.example",
            "w" * 48,
            transport=httpx.MockTransport(handle),
        )
        try:
            with self.assertRaisesRegex(WorkerToolBridgeError, "did not match"):
                await client.deliver(request)
            with self.assertRaisesRegex(WorkerToolBridgeError, "no durable job"):
                await client.deliver(request.model_copy(update={"job_id": None}))
        finally:
            await client.close()

    def test_rejects_remote_cleartext_and_embedded_credentials(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            WorkerDeviceToolClient("http://api.example", "w" * 48)
        with self.assertRaisesRegex(ValueError, "credentials"):
            WorkerDeviceToolClient("https://user@api.example", "w" * 48)
        with self.assertRaisesRegex(ValueError, "origin"):
            WorkerDeviceToolClient("https://api.example/prefix", "w" * 48)


if __name__ == "__main__":
    unittest.main()
