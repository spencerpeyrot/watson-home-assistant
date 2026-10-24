"""Authenticated gateway session lifecycle for the Pi device."""

from __future__ import annotations

from urllib.parse import urlsplit
from uuid import UUID

import httpx

from contracts.models import SessionCreateRequest, SessionResponse
from device.transport import DeviceTransportError


class GatewaySessionClient:
    """Create and close device-scoped sessions over authenticated HTTPS."""

    def __init__(
        self,
        gateway_url: str,
        device_token: str,
        *,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(gateway_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("gateway URL must be absolute HTTP(S)")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("gateway URL must not include embedded credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("gateway URL must not include a query or fragment")
        if parsed.scheme == "http" and parsed.hostname.lower() not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("unencrypted gateway sessions are allowed only on loopback")
        if len(device_token) < 32:
            raise ValueError("device token must contain at least 32 characters")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._client = httpx.AsyncClient(
            base_url=gateway_url.rstrip("/"),
            headers={"Authorization": f"Bearer {device_token}"},
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    async def create_session(
        self,
        *,
        conversation_id: UUID,
        locale: str,
        timezone: str,
    ) -> SessionResponse:
        request = SessionCreateRequest(
            conversation_id=conversation_id,
            locale=locale,
            timezone=timezone,
        )
        try:
            response = await self._client.post(
                "/v1/sessions",
                json=request.model_dump(mode="json"),
            )
        except httpx.RequestError as error:
            raise DeviceTransportError(
                f"session creation transport failed ({type(error).__name__})"
            ) from None
        if response.status_code != 201:
            raise DeviceTransportError(
                f"session creation rejected (HTTP {response.status_code})"
            )
        try:
            return SessionResponse.model_validate(response.json())
        except (ValueError, TypeError) as error:
            raise DeviceTransportError(
                f"session response invalid ({type(error).__name__})"
            ) from None

    async def close_session(self, session_id: UUID) -> None:
        try:
            response = await self._client.delete(f"/v1/sessions/{session_id}")
        except httpx.RequestError as error:
            raise DeviceTransportError(
                f"session close transport failed ({type(error).__name__})"
            ) from None
        if response.status_code != 204:
            raise DeviceTransportError(
                f"session close rejected (HTTP {response.status_code})"
            )

    async def close(self) -> None:
        await self._client.aclose()
