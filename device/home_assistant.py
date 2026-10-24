"""Pi-local Home Assistant tool adapter for authenticated cloud requests."""

from __future__ import annotations

from contracts.models import LocalToolRequest, ToolResult
from device.local_tools import make_tool_result
from service.home_assistant import (
    EntityRequest,
    HomeAssistantClient,
    HomeAssistantError,
    LightStateRequest,
)


class PiHomeAssistantExecutor:
    """Execute only the narrow Home Assistant tools configured on this Pi."""

    def __init__(self, client: HomeAssistantClient) -> None:
        self._client = client

    async def __call__(self, request: LocalToolRequest) -> ToolResult:
        if request.tool_name == "get_light_state":
            try:
                payload = EntityRequest.model_validate(request.payload)
                state = await self._client.get_light_state(payload.entity_id)
            except ValueError:
                return make_tool_result(
                    request, status="denied", error_code="INVALID_REQUEST"
                )
            except HomeAssistantError:
                return make_tool_result(
                    request,
                    status="failed",
                    error_code="DEPENDENCY_UNAVAILABLE",
                )
            if state is None:
                return make_tool_result(
                    request, status="failed", error_code="RESOURCE_NOT_FOUND"
                )
            return make_tool_result(
                request,
                status="succeeded",
                verified=True,
                observed_state={"entity_id": payload.entity_id, "state": state},
            )

        if request.tool_name == "set_light_state":
            try:
                payload = LightStateRequest.model_validate(request.payload)
                outcome = await self._client.set_light(payload.entity_id, payload.state)
            except ValueError:
                return make_tool_result(
                    request, status="denied", error_code="INVALID_REQUEST"
                )
            except HomeAssistantError:
                return make_tool_result(
                    request,
                    status="needs_reconciliation",
                    error_code="TOOL_WRITE_UNCERTAIN",
                )
            result = outcome.result if isinstance(outcome.result, dict) else None
            return make_tool_result(
                request,
                status=("succeeded" if outcome.verified else "needs_reconciliation"),
                verified=outcome.verified,
                observed_state=result,
                error_code=None if outcome.verified else "TOOL_WRITE_UNVERIFIED",
            )

        return make_tool_result(request, status="denied", error_code="ACTION_DENIED")
