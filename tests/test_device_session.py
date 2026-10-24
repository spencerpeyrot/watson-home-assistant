from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from uuid import UUID

import httpx

from device.session import GatewaySessionClient
from device.transport import DeviceTransportError


CONVERSATION = UUID("00000000-0000-4000-8000-000000000030")
SESSION = UUID("00000000-0000-4000-8000-000000000031")


class DeviceSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_close_uses_device_scope_and_contract_payload(self) -> None:
        seen: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.method == "POST":
                return httpx.Response(
                    201,
                    json={
                        "schema_version": 1,
                        "session_id": str(SESSION),
                        "conversation_id": str(CONVERSATION),
                        "state": "ready",
                        "opened_at": datetime.now(timezone.utc).isoformat(),
                        "idle_close_seconds": 30,
                    },
                )
            return httpx.Response(204)

        client = GatewaySessionClient(
            "http://127.0.0.1:8080",
            "d" * 40,
            transport=httpx.MockTransport(handler),
        )
        try:
            session = await client.create_session(
                conversation_id=CONVERSATION,
                locale="en-US",
                timezone="America/New_York",
            )
            await client.close_session(session.session_id)
        finally:
            await client.close()

        self.assertEqual(session.session_id, SESSION)
        self.assertEqual([request.method for request in seen], ["POST", "DELETE"])
        self.assertEqual(seen[0].url.path, "/v1/sessions")
        self.assertEqual(seen[0].headers["Authorization"], f"Bearer {'d' * 40}")
        self.assertEqual(seen[1].url.path, f"/v1/sessions/{SESSION}")
        self.assertEqual(
            json.loads(seen[0].read()),
            {
                "schema_version": 1,
                "conversation_id": str(CONVERSATION),
                "locale": "en-US",
                "timezone": "America/New_York",
            },
        )

    async def test_rejects_insecure_remote_endpoint_and_short_token(self) -> None:
        with self.assertRaises(ValueError):
            GatewaySessionClient("http://gateway.example", "d" * 40)
        with self.assertRaises(ValueError):
            GatewaySessionClient("https://gateway.example", "short")

    async def test_http_error_does_not_echo_response_body(self) -> None:
        client = GatewaySessionClient(
            "https://gateway.example",
            "d" * 40,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(403, text="sensitive gateway detail")
            ),
        )
        try:
            with self.assertRaises(DeviceTransportError) as context:
                await client.create_session(
                    conversation_id=CONVERSATION,
                    locale="en-US",
                    timezone="America/New_York",
                )
        finally:
            await client.close()
        self.assertNotIn("sensitive gateway detail", str(context.exception))


if __name__ == "__main__":
    unittest.main()
