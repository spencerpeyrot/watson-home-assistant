"""Home Assistant client restricted to explicitly configured light entities."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from hashlib import sha256
from typing import Literal
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, create_model

from contracts.models import LocalToolRequest, ToolResult
from service.tool_policy import (
    ToolExecutionContext,
    ToolOutcome,
    ToolPolicyRegistry,
)


class HomeAssistantError(RuntimeError):
    """Sanitized Home Assistant transport or response failure."""


class EntityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    entity_id: str = Field(min_length=1, max_length=255)


class LightStateRequest(EntityRequest):
    state: Literal["on", "off"]


def _allowlisted_payload_models(
    allowed_light_entities: frozenset[str],
) -> tuple[type[EntityRequest], type[LightStateRequest]]:
    """Give the model exact entity enums while retaining strict runtime payloads."""

    allowed = tuple(sorted(allowed_light_entities))
    entity_type = Literal.__getitem__(allowed)
    suffix = sha256("\0".join(allowed).encode("utf-8")).hexdigest()[:10]
    entity_request = create_model(
        f"AllowlistedEntityRequest_{suffix}",
        __base__=EntityRequest,
        entity_id=(entity_type, Field(description="Exact allowlisted light entity ID.")),
    )
    light_state_request = create_model(
        f"AllowlistedLightStateRequest_{suffix}",
        __base__=LightStateRequest,
        entity_id=(entity_type, Field(description="Exact allowlisted light entity ID.")),
    )
    return entity_request, light_state_request


class HomeAssistantClient:
    """Read and control only listed `light.*` entity IDs."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        allowed_light_entities: frozenset[str],
        timeout_seconds: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Home Assistant URL must be absolute HTTP(S)")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Home Assistant URL must not include credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("Home Assistant URL must not include a query or fragment")
        if parsed.scheme == "http" and parsed.hostname.lower() not in {
            "localhost", "127.0.0.1", "::1"
        }:
            raise ValueError("unencrypted Home Assistant access is allowed only on loopback")
        if len(token) < 32 or timeout_seconds <= 0:
            raise ValueError("Home Assistant token or timeout is invalid")
        if not allowed_light_entities or any(
            re.fullmatch(r"light\.[a-z0-9_]+", entity) is None
            or len(entity) > 255
            for entity in allowed_light_entities
        ):
            raise ValueError("allowlist must contain only light entity IDs")
        self.allowed_light_entities = allowed_light_entities
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    def require_allowed_light(self, entity_id: str) -> str:
        if entity_id not in self.allowed_light_entities:
            raise ValueError("entity is not in the configured light allowlist")
        return entity_id

    async def get_light_state(self, entity_id: str) -> str | None:
        entity_id = self.require_allowed_light(entity_id)
        try:
            response = await self._client.get(
                f"/api/states/{quote(entity_id, safe='.')}"
            )
        except httpx.RequestError as error:
            raise HomeAssistantError(f"state read failed ({type(error).__name__})") from None
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise HomeAssistantError(f"state read rejected (HTTP {response.status_code})")
        try:
            payload = response.json()
            if payload.get("entity_id") != entity_id or payload.get("state") not in {"on", "off"}:
                raise ValueError("state shape mismatch")
            return payload["state"]
        except (ValueError, TypeError, AttributeError):
            raise HomeAssistantError("state response was invalid") from None

    async def set_light(self, entity_id: str, state: Literal["on", "off"]) -> ToolOutcome:
        entity_id = self.require_allowed_light(entity_id)
        service = "turn_on" if state == "on" else "turn_off"
        try:
            response = await self._client.post(
                f"/api/services/light/{service}", json={"entity_id": entity_id}
            )
        except httpx.RequestError as error:
            # A lost response after a write is uncertain; the ledger blocks retries.
            raise HomeAssistantError(f"light write outcome uncertain ({type(error).__name__})") from None
        if response.status_code not in {200, 201}:
            raise HomeAssistantError(f"light write rejected (HTTP {response.status_code})")
        observed = await self.get_light_state(entity_id)
        verified = observed == state
        return ToolOutcome(
            verified=verified,
            result={"entity_id": entity_id, "requested_state": state, "observed_state": observed},
        )

    def register_tools(self, registry: ToolPolicyRegistry) -> None:
        """Register exact read-light and automatic set-light tools."""

        entity_request, light_state_request = _allowlisted_payload_models(
            self.allowed_light_entities
        )

        async def read(payload: BaseModel) -> ToolOutcome:
            assert isinstance(payload, EntityRequest)
            state = await self.get_light_state(payload.entity_id)
            return ToolOutcome(
                verified=state is not None,
                result={"entity_id": payload.entity_id, "observed_state": state},
            )

        async def write(payload: BaseModel) -> ToolOutcome:
            assert isinstance(payload, LightStateRequest)
            return await self.set_light(payload.entity_id, payload.state)

        registry.register(
            name="get_light_state",
            description="Read the current state of one explicitly allowlisted light.",
            payload_model=entity_request,
            risk="read_only",
            execute=read,
            resource_key=lambda payload: self.require_allowed_light(payload.entity_id),
        )
        registry.register(
            name="set_light_state",
            description=(
                "Set one explicitly allowlisted light on or off when the user asks. "
                "Use only for a direct, simple light-control request."
            ),
            payload_model=light_state_request,
            risk="automatic",
            execute=write,
            resource_key=lambda payload: self.require_allowed_light(payload.entity_id),
        )

    async def close(self) -> None:
        await self._client.aclose()


LocalToolSender = Callable[[LocalToolRequest], Awaitable[ToolResult]]


class HomeAssistantLocalBridge:
    """Register cloud policy tools that execute on the authenticated Pi."""

    def __init__(
        self,
        *,
        allowed_light_entities: frozenset[str],
        deliver: LocalToolSender,
    ) -> None:
        if not allowed_light_entities or any(
            re.fullmatch(r"light\.[a-z0-9_]+", entity) is None
            or len(entity) > 255
            for entity in allowed_light_entities
        ):
            raise ValueError("allowlist must contain only light entity IDs")
        self._allowed_light_entities = allowed_light_entities
        self._deliver = deliver

    def register_tools(self, registry: ToolPolicyRegistry) -> None:
        """Register narrow Pi-local read and automatic light tools."""

        entity_request, light_state_request = _allowlisted_payload_models(
            self._allowed_light_entities
        )

        async def execute(
            tool_name: str,
            payload: BaseModel,
            context: ToolExecutionContext,
        ) -> ToolOutcome:
            local = context.local
            request = LocalToolRequest(
                action_id=context.action_id,
                trace_id=context.trace_id,
                device_id=local.device_id,
                conversation_id=local.conversation_id,
                session_id=local.session_id,
                delegation_id=local.delegation_id,
                job_id=context.job_id,
                turn_id=local.turn_id,
                context_revision=context.context_revision,
                tool_name=tool_name,
                payload=payload.model_dump(mode="json"),
                issued_at=context.issued_at,
                expires_at=local.expires_at,
            )
            result = await self._deliver(request)
            if not _tool_result_matches(request, result):
                return ToolOutcome(
                    verified=False,
                    result={"status": "needs_reconciliation", "error_code": "INVALID_REQUEST"},
                )
            return ToolOutcome(
                verified=result.status == "succeeded" and result.verified,
                result=result.model_dump(mode="json"),
            )

        registry.register(
            name="get_light_state",
            description="Read the current state of one explicitly allowlisted light.",
            payload_model=entity_request,
            risk="read_only",
            execute_contextual=lambda payload, context: execute(
                "get_light_state", payload, context
            ),
            resource_key=lambda payload: self._require_allowed_light(payload.entity_id),
        )
        registry.register(
            name="set_light_state",
            description=(
                "Set one explicitly allowlisted light on or off when the user asks. "
                "Use only for a direct, simple light-control request."
            ),
            payload_model=light_state_request,
            risk="automatic",
            execute_contextual=lambda payload, context: execute(
                "set_light_state", payload, context
            ),
            resource_key=lambda payload: self._require_allowed_light(payload.entity_id),
        )

    def _require_allowed_light(self, entity_id: str) -> str:
        if entity_id not in self._allowed_light_entities:
            raise ValueError("entity is not in the configured light allowlist")
        return entity_id


def _tool_result_matches(request: LocalToolRequest, result: ToolResult) -> bool:
    return all(
        getattr(request, field) == getattr(result, field)
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
