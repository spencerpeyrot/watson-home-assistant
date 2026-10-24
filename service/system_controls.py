"""Cloud policy bridge for narrow Pi speaker and power controls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from contracts.models import LocalToolRequest, ToolResult
from service.tool_policy import (
    ToolExecutionContext,
    ToolOutcome,
    ToolPolicyRegistry,
)


LocalToolSender = Callable[[LocalToolRequest], Awaitable[ToolResult]]


class SpeakerVolumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    direction: Literal["up", "down"]
    percent: int = Field(default=5, ge=0, le=100)


class SpeakerVolumeSetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    target_percent: int = Field(ge=0, le=100)


class GetSpeakerVolumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DevicePowerOffRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LocalSystemControlsBridge:
    """Register direct user-requested controls executed on the authenticated Pi."""

    def __init__(self, *, deliver: LocalToolSender) -> None:
        self._deliver = deliver

    def register_tools(self, registry: ToolPolicyRegistry) -> None:
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
            name="get_speaker_volume",
            description=(
                "Read Cube speaker playback volume as measured percent from 0 to 100. "
                "Use for direct questions about current speaker volume."
            ),
            payload_model=GetSpeakerVolumeRequest,
            risk="read_only",
            execute_contextual=lambda payload, context: execute(
                "get_speaker_volume", payload, context
            ),
            resource_key=lambda _payload: "cube-speaker-volume",
        )
        registry.register(
            name="adjust_speaker_volume",
            description=(
                "Raise or lower Cube speaker playback volume by the requested number of "
                "percentage points. Use only for direct speaker-volume requests; do not "
                "change microphone gain."
            ),
            payload_model=SpeakerVolumeRequest,
            risk="automatic",
            execute_contextual=lambda payload, context: execute(
                "adjust_speaker_volume", payload, context
            ),
            resource_key=lambda _payload: "cube-speaker-volume",
        )
        registry.register(
            name="set_speaker_volume",
            description=(
                "Set Cube speaker playback volume to an absolute percentage from 0 to 100. "
                "Use only for direct requests to set volume to a specific level."
            ),
            payload_model=SpeakerVolumeSetRequest,
            risk="automatic",
            execute_contextual=lambda payload, context: execute(
                "set_speaker_volume", payload, context
            ),
            resource_key=lambda _payload: "cube-speaker-volume",
        )
        registry.register(
            name="power_off_device",
            description=(
                "Gracefully power off Cube only when the user directly asks to turn off "
                "Cube or this device. Never use for mute, sleep, another device, or an "
                "ambiguous request. The voice session ends when shutdown begins."
            ),
            payload_model=DevicePowerOffRequest,
            risk="automatic",
            execute_contextual=lambda payload, context: execute(
                "power_off_device", payload, context
            ),
            resource_key=lambda _payload: "cube-power-state",
        )


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
