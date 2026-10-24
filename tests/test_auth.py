from __future__ import annotations

import unittest
from uuid import UUID

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from service.auth import (
    AuthenticationRejected,
    AuthenticatedPrincipal,
    BearerCredential,
    CredentialAuthenticator,
    authenticator_from_environment,
)
from service.durable_api import create_durable_app


OWNER_ID = UUID("00000000-0000-4000-8000-000000000701")
DEVICE_TOKEN = "d" * 48
OWNER_TOKEN = "o" * 48
OPERATOR_TOKEN = "x" * 48
WORKER_TOKEN = "w" * 48


def request_with_token(token: str) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [(b"authorization", f"Bearer {token}".encode("ascii"))],
        }
    )


class FakeWebSocket:
    def __init__(self, authorization: str):
        self.headers = Headers({"authorization": authorization})


class FakeStore:
    async def healthcheck(self):
        return True


class CredentialAuthenticatorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.auth = CredentialAuthenticator(
            [
                BearerCredential(
                    token=DEVICE_TOKEN,
                    principal=AuthenticatedPrincipal(
                        role="device", owner_id=OWNER_ID, device_id="cube-01"
                    ),
                ),
                BearerCredential(
                    token=OWNER_TOKEN,
                    principal=AuthenticatedPrincipal(role="owner", owner_id=OWNER_ID),
                ),
                BearerCredential(
                    token=OPERATOR_TOKEN,
                    principal=AuthenticatedPrincipal(role="operator", owner_id=None),
                ),
                BearerCredential(
                    token=WORKER_TOKEN,
                    principal=AuthenticatedPrincipal(
                        role="worker", owner_id=OWNER_ID, device_id="cube-01"
                    ),
                ),
            ]
        )

    def test_authenticator_wires_into_durable_http_dependencies(self):
        app = create_durable_app(
            store=FakeStore(),  # type: ignore[arg-type]
            authenticate=self.auth.authenticate_http,
        )
        with TestClient(app) as client:
            missing = client.get("/v1/health")
            owner = client.get(
                "/v1/health", headers={"Authorization": f"Bearer {OWNER_TOKEN}"}
            )
            operator = client.get(
                "/v1/health", headers={"Authorization": f"Bearer {OPERATOR_TOKEN}"}
            )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(owner.status_code, 403)
        self.assertEqual(operator.status_code, 200)

    async def test_http_auth_maps_bearer_to_fixed_roles(self):
        device = await self.auth.authenticate_http(request_with_token(DEVICE_TOKEN))
        owner = await self.auth.authenticate_http(request_with_token(OWNER_TOKEN))
        operator = await self.auth.authenticate_http(request_with_token(OPERATOR_TOKEN))
        worker = await self.auth.authenticate_http(request_with_token(WORKER_TOKEN))

        self.assertEqual((device.role, device.owner_id, device.device_id), ("device", OWNER_ID, "cube-01"))
        self.assertEqual((owner.role, owner.owner_id, owner.device_id), ("owner", OWNER_ID, None))
        self.assertEqual((operator.role, operator.owner_id), ("operator", None))
        self.assertEqual((worker.role, worker.owner_id, worker.device_id), ("worker", OWNER_ID, "cube-01"))

    async def test_http_rejects_missing_bad_and_non_bearer_credentials(self):
        for authorization in ("", "Basic " + OWNER_TOKEN, "Bearer unknown-token"):
            with self.subTest(authorization=authorization[:12]):
                with self.assertRaises(HTTPException) as caught:
                    await self.auth.authenticate_http(
                        Request({"type": "http", "headers": [(b"authorization", authorization.encode())]})
                    )
                self.assertEqual(caught.exception.status_code, 401)
                self.assertNotIn(OWNER_TOKEN, str(caught.exception))

    async def test_websocket_auth_rejects_without_exposing_token(self):
        principal = await self.auth.authenticate_websocket(
            FakeWebSocket(f"Bearer {DEVICE_TOKEN}")  # type: ignore[arg-type]
        )
        self.assertEqual(principal.device_id, "cube-01")

        with self.assertRaises(AuthenticationRejected) as caught:
            await self.auth.authenticate_websocket(
                FakeWebSocket("Bearer wrong-token")  # type: ignore[arg-type]
            )
        self.assertNotIn(DEVICE_TOKEN, str(caught.exception))

    def test_configuration_reads_only_passed_process_environment_mapping(self):
        auth = authenticator_from_environment(
            {
                "CUBE_OWNER_ID": str(OWNER_ID),
                "CUBE_DEVICE_ID": "cube-01",
                "CUBE_OWNER_TOKEN": OWNER_TOKEN,
                "CUBE_DEVICE_TOKEN": DEVICE_TOKEN,
                "CUBE_WORKER_TOKEN": WORKER_TOKEN,
            }
        )
        self.assertIsInstance(auth, CredentialAuthenticator)

    def test_duplicate_tokens_and_unmapped_device_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            CredentialAuthenticator(
                [
                    BearerCredential(
                        token=OWNER_TOKEN,
                        principal=AuthenticatedPrincipal(role="owner", owner_id=OWNER_ID),
                    ),
                    BearerCredential(
                        token=OWNER_TOKEN,
                        principal=AuthenticatedPrincipal(role="owner", owner_id=OWNER_ID),
                    ),
                ]
            )
        with self.assertRaisesRegex(ValueError, "device mappings"):
            CredentialAuthenticator(
                [
                    BearerCredential(
                        token=DEVICE_TOKEN,
                        principal=AuthenticatedPrincipal(role="device", owner_id=OWNER_ID),
                    )
                ]
            )


if __name__ == "__main__":
    unittest.main()
