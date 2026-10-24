"""Bounded local speaker-volume and graceful-shutdown controls for the Pi."""

from __future__ import annotations

import asyncio
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from contracts.models import LocalToolRequest, ToolResult
from device.local_tools import make_tool_result


_VOLUME_RE = re.compile(r"Front (Left|Right): Playback \d+ \[(\d+)%\]")
POWER_OFF_GRACE_SECONDS = 30


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


class PiSystemControlsExecutor:
    """Execute only fixed ALSA and systemd commands; never invoke a shell."""

    def __init__(
        self,
        *,
        volume_card: str = "seeed2micvoicec",
        timeout_seconds: float = 8.0,
    ) -> None:
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", volume_card) is None:
            raise ValueError("ALSA volume card name is invalid")
        if timeout_seconds <= 0:
            raise ValueError("system-control command timeout must be positive")
        self._volume_card = volume_card
        self._timeout_seconds = timeout_seconds

    async def __call__(self, request: LocalToolRequest) -> ToolResult:
        if request.tool_name == "get_speaker_volume":
            return await self._get_speaker_volume(request)
        if request.tool_name == "adjust_speaker_volume":
            return await self._adjust_speaker_volume(request)
        if request.tool_name == "set_speaker_volume":
            return await self._set_speaker_volume(request)
        if request.tool_name == "power_off_device":
            return await self._power_off_device(request)
        return make_tool_result(request, status="denied", error_code="ACTION_DENIED")

    async def _get_speaker_volume(self, request: LocalToolRequest) -> ToolResult:
        try:
            GetSpeakerVolumeRequest.model_validate(request.payload)
        except ValueError:
            return make_tool_result(request, status="denied", error_code="INVALID_REQUEST")
        volume = await self._read_volume()
        if volume is None:
            return make_tool_result(
                request,
                status="failed",
                error_code="DEVICE_VOLUME_UNAVAILABLE",
                retryable=True,
            )
        return make_tool_result(
            request,
            status="succeeded",
            verified=True,
            observed_state=self._volume_state(volume),
        )

    async def _adjust_speaker_volume(self, request: LocalToolRequest) -> ToolResult:
        try:
            payload = SpeakerVolumeRequest.model_validate(request.payload)
        except ValueError:
            return make_tool_result(request, status="denied", error_code="INVALID_REQUEST")

        before = await self._read_volume()
        if before is None:
            return make_tool_result(
                request,
                status="failed",
                error_code="DEVICE_VOLUME_UNAVAILABLE",
                retryable=True,
            )
        adjustment = f"{payload.percent}%+" if payload.direction == "up" else f"{payload.percent}%-"
        result = await self._run(
            "/usr/bin/amixer",
            "-c",
            self._volume_card,
            "sset",
            "PCM",
            adjustment,
        )
        after = await self._read_volume() if result is not None and result[0] == 0 else None
        if after is None:
            return make_tool_result(
                request,
                status="needs_reconciliation",
                error_code="DEVICE_VOLUME_CHANGE_UNVERIFIED",
                observed_state={"card": self._volume_card, "control": "PCM"},
            )
        changed = before != after
        return make_tool_result(
            request,
            status="succeeded",
            verified=True,
            observed_state={
                "card": self._volume_card,
                "control": "PCM",
                "left_percent": after[0],
                "right_percent": after[1],
                "requested_direction": payload.direction,
                "requested_percent": payload.percent,
                "changed": changed,
                "at_limit": payload.percent > 0 and not changed,
            },
        )

    async def _set_speaker_volume(self, request: LocalToolRequest) -> ToolResult:
        try:
            payload = SpeakerVolumeSetRequest.model_validate(request.payload)
        except ValueError:
            return make_tool_result(request, status="denied", error_code="INVALID_REQUEST")
        before = await self._read_volume()
        if before is None:
            return make_tool_result(
                request,
                status="failed",
                error_code="DEVICE_VOLUME_UNAVAILABLE",
                retryable=True,
            )
        result = await self._run(
            "/usr/bin/amixer",
            "-c",
            self._volume_card,
            "sset",
            "PCM",
            f"{payload.target_percent}%",
        )
        after = await self._read_volume() if result is not None and result[0] == 0 else None
        if after is None:
            return make_tool_result(
                request,
                status="needs_reconciliation",
                error_code="DEVICE_VOLUME_CHANGE_UNVERIFIED",
                observed_state={"card": self._volume_card, "control": "PCM"},
            )
        return make_tool_result(
            request,
            status="succeeded",
            verified=True,
            observed_state={
                **self._volume_state(after),
                "requested_target_percent": payload.target_percent,
                "changed": before != after,
            },
        )

    async def _power_off_device(self, request: LocalToolRequest) -> ToolResult:
        try:
            DevicePowerOffRequest.model_validate(request.payload)
        except ValueError:
            return make_tool_result(request, status="denied", error_code="INVALID_REQUEST")
        unit_name = f"cube-assistant-poweroff-{request.action_id.hex}"
        result = await self._run(
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/systemd-run",
            "--quiet",
            "--collect",
            f"--unit={unit_name}",
            f"--on-active={POWER_OFF_GRACE_SECONDS}s",
            "/usr/bin/systemctl",
            "poweroff",
        )
        if result is None or result[0] != 0:
            return make_tool_result(
                request,
                status="needs_reconciliation",
                error_code="DEVICE_SHUTDOWN_UNCONFIRMED",
            )
        return make_tool_result(
            request,
            status="succeeded",
            verified=True,
            observed_state={
                "power_state": "shutdown_scheduled",
                "grace_seconds": POWER_OFF_GRACE_SECONDS,
            },
        )

    async def _read_volume(self) -> tuple[int, int] | None:
        result = await self._run(
            "/usr/bin/amixer",
            "-c",
            self._volume_card,
            "sget",
            "PCM",
        )
        if result is None or result[0] != 0:
            return None
        channels = {name: int(value) for name, value in _VOLUME_RE.findall(result[1])}
        if set(channels) != {"Left", "Right"}:
            return None
        return channels["Left"], channels["Right"]

    def _volume_state(self, volume: tuple[int, int]) -> dict[str, object]:
        return {
            "card": self._volume_card,
            "control": "PCM",
            "left_percent": volume[0],
            "right_percent": volume[1],
        }

    async def _run(self, *command: str) -> tuple[int, str] | None:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError:
            return None
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=self._timeout_seconds
            )
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
                await process.communicate()
            raise
        except TimeoutError:
            process.kill()
            await process.communicate()
            return None
        return process.returncode or 0, stdout.decode("utf-8", errors="replace")
