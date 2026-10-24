"""Worker-only HTTP client for Pi-local, policy-approved tool execution."""

from __future__ import annotations

from urllib.parse import urlsplit

import httpx

from contracts.models import LocalToolRequest, ToolResult


class WorkerToolBridgeError(RuntimeError):
    """Sanitized API bridge failure; never includes credentials or response text."""


class WorkerDeviceToolClient:
    """Send scoped local-tool requests to the authenticated API process."""

    def __init__(
        self,
        api_url: str,
        token: str,
        *,
        timeout_seconds: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(api_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("internal API URL must be absolute HTTP(S)")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("internal API URL must not include credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("internal API URL must not include a query or fragment")
        if parsed.path not in {"", "/"}:
            raise ValueError("internal API URL must contain only an origin")
        if parsed.scheme == "http" and parsed.hostname.lower() not in {
            "localhost", "127.0.0.1", "::1"
        }:
            raise ValueError("unencrypted internal API access is allowed only on loopback")
        if not 32 <= len(token) <= 4_096 or any(char.isspace() for char in token):
            raise ValueError("internal worker token is invalid")
        if timeout_seconds <= 0:
            raise ValueError("internal API timeout must be positive")
        self._client = httpx.AsyncClient(
            base_url=api_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    async def deliver(self, request: LocalToolRequest) -> ToolResult:
        """Send one exact typed request and require its correlated device result."""

        if request.job_id is None:
            raise WorkerToolBridgeError("local tool request has no durable job")
        try:
            response = await self._client.post(
                "/internal/v1/local-tools", json=request.model_dump(mode="json")
            )
        except httpx.RequestError as error:
            raise WorkerToolBridgeError(
                f"internal tool bridge transport failed ({type(error).__name__})"
            ) from None
        if response.status_code != 200:
            raise WorkerToolBridgeError(
                f"internal tool bridge rejected request (HTTP {response.status_code})"
            )
        try:
            result = ToolResult.model_validate(response.json())
        except (ValueError, TypeError):
            raise WorkerToolBridgeError("internal tool bridge returned an invalid result") from None
        if (
            result.action_id != request.action_id
            or result.trace_id != request.trace_id
            or result.job_id != request.job_id
            or result.device_id != request.device_id
            or result.conversation_id != request.conversation_id
            or result.session_id != request.session_id
            or result.delegation_id != request.delegation_id
            or result.turn_id != request.turn_id
            or result.context_revision != request.context_revision
        ):
            raise WorkerToolBridgeError("device result did not match the local-tool request")
        return result

    async def close(self) -> None:
        await self._client.aclose()
