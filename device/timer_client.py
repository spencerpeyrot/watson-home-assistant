"""Environment-composed Pi timer event client."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import httpx

from contracts.models import LocalToolRequest, ToolResult
from device.config import device_state_dir
from device.home_assistant import PiHomeAssistantExecutor
from device.local_tools import LocalToolInbox, make_tool_result
from device.system_controls import PiSystemControlsExecutor
from device.timer_events import TimerEventInbox, run_timer_event_client, timer_event_websocket_url
from device.reconnect import RetryNotice
from device.timer_tts import OpenAITimerSpeech, SUPPORTED_VOICES, make_timer_notifier
from service.home_assistant import HomeAssistantClient
from voice.observability import (
    initialize as initialize_observability,
    shutdown as shutdown_observability,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimerClientSettings:
    gateway_url: str
    device_id: str
    device_token: str = field(repr=False)
    playback_pcm: str
    data_dir: Path
    openai_api_key: str = field(default="", repr=False)
    tts_model: str = OpenAITimerSpeech.MODEL
    tts_voice: str = OpenAITimerSpeech.VOICE
    home_assistant_url: str | None = None
    home_assistant_token: str = field(default="", repr=False)
    local_light_entities: frozenset[str] = frozenset()
    volume_card: str = "seeed2micvoicec"

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> TimerClientSettings:
        values = os.environ if environ is None else environ
        gateway_url = values.get("CUBE_GATEWAY_URL", "").strip()
        device_id = values.get("CUBE_DEVICE_ID", "").strip()
        device_token = values.get("CUBE_DEVICE_TOKEN", "")
        playback_pcm = values.get("CUBE_TIMER_PLAYBACK_PCM", "").strip()
        openai_api_key = values.get("OPENAI_API_KEY", "").strip()
        tts_model = values.get("OPENAI_TTS_MODEL", OpenAITimerSpeech.MODEL).strip()
        tts_voice = values.get("OPENAI_TTS_VOICE", OpenAITimerSpeech.VOICE).strip()
        volume_card = values.get("CUBE_VOLUME_ALSA_CARD", "seeed2micvoicec").strip()
        if not gateway_url:
            raise RuntimeError("CUBE_GATEWAY_URL is required")
        if not device_id:
            raise RuntimeError("CUBE_DEVICE_ID is required")
        if len(device_token) < 32 or any(character.isspace() for character in device_token):
            raise RuntimeError(
                "CUBE_DEVICE_TOKEN must be at least 32 characters without whitespace"
            )
        if not playback_pcm:
            raise RuntimeError("CUBE_TIMER_PLAYBACK_PCM is required for audible timer delivery")
        if tts_model != OpenAITimerSpeech.MODEL:
            raise RuntimeError("OPENAI_TTS_MODEL is unsupported")
        if tts_voice not in SUPPORTED_VOICES:
            raise RuntimeError("OPENAI_TTS_VOICE is unsupported")
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", volume_card) is None:
            raise RuntimeError("CUBE_VOLUME_ALSA_CARD is invalid")
        try:
            timer_event_websocket_url(gateway_url, device_id)
        except ValueError:
            raise RuntimeError("CUBE_GATEWAY_URL or CUBE_DEVICE_ID is invalid") from None
        home_assistant_url = values.get("CUBE_HOME_ASSISTANT_URL", "").strip()
        home_assistant_token = values.get("CUBE_HOME_ASSISTANT_TOKEN", "")
        raw_entities = values.get("CUBE_LOCAL_LIGHT_ENTITIES", "")
        entity_parts = [part.strip() for part in raw_entities.split(",") if part.strip()]
        if raw_entities.strip() and len(entity_parts) != len(raw_entities.split(",")):
            raise RuntimeError("CUBE_LOCAL_LIGHT_ENTITIES contains an empty entry")
        if len(entity_parts) != len(set(entity_parts)):
            raise RuntimeError("CUBE_LOCAL_LIGHT_ENTITIES contains duplicates")
        if any(
            re.fullmatch(r"light\.[a-z0-9_]+", entity) is None or len(entity) > 255
            for entity in entity_parts
        ):
            raise RuntimeError("CUBE_LOCAL_LIGHT_ENTITIES must contain light entity IDs")
        ha_values = (home_assistant_url, home_assistant_token, tuple(entity_parts))
        if any(ha_values) and not all(ha_values):
            raise RuntimeError(
                "CUBE_HOME_ASSISTANT_URL, CUBE_HOME_ASSISTANT_TOKEN, and "
                "CUBE_LOCAL_LIGHT_ENTITIES must be configured together"
            )
        if home_assistant_token and (
            len(home_assistant_token) < 32
            or any(character.isspace() for character in home_assistant_token)
        ):
            raise RuntimeError("CUBE_HOME_ASSISTANT_TOKEN is invalid")
        return cls(
            gateway_url=gateway_url,
            device_id=device_id,
            device_token=device_token,
            playback_pcm=playback_pcm,
            data_dir=device_state_dir(values),
            openai_api_key=openai_api_key,
            tts_model=tts_model,
            tts_voice=tts_voice,
            home_assistant_url=home_assistant_url or None,
            home_assistant_token=home_assistant_token,
            local_light_entities=frozenset(entity_parts),
            volume_card=volume_card,
        )


async def run(settings: TimerClientSettings) -> None:
    """Run the authenticated timer socket until SIGINT or SIGTERM."""

    settings.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_number, stop_event.set)
        except (NotImplementedError, RuntimeError):
            continue
    tts_client: httpx.AsyncClient | None = None
    home_assistant: HomeAssistantClient | None = None
    try:
        await asyncio.to_thread(initialize_observability, "timer-client")
        inbox = TimerEventInbox(settings.data_dir / "timer-events.sqlite3")
        if settings.openai_api_key:
            tts_client = httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=5.0))
        else:
            LOGGER.warning("OPENAI_API_KEY is unset; timer events will use chime fallback")
        local_tool_inbox = LocalToolInbox(
            settings.data_dir / "local-tools.sqlite3",
            device_id=settings.device_id,
        )
        system_controls = PiSystemControlsExecutor(volume_card=settings.volume_card)
        home_assistant_executor: PiHomeAssistantExecutor | None = None
        if settings.home_assistant_url is not None:
            home_assistant = HomeAssistantClient(
                settings.home_assistant_url,
                settings.home_assistant_token,
                allowed_light_entities=settings.local_light_entities,
            )
            home_assistant_executor = PiHomeAssistantExecutor(home_assistant)

        async def execute_tool(request: LocalToolRequest) -> ToolResult:
            if request.tool_name in {
                "get_speaker_volume",
                "adjust_speaker_volume",
                "set_speaker_volume",
                "power_off_device",
            }:
                return await system_controls(request)
            if home_assistant_executor is not None:
                return await home_assistant_executor(request)
            return make_tool_result(
                request,
                status="denied",
                error_code="ACTION_DENIED",
            )

        await run_timer_event_client(
            gateway_url=settings.gateway_url,
            device_id=settings.device_id,
            device_token=settings.device_token,
            inbox=inbox,
            notify=make_timer_notifier(
                pcm=settings.playback_pcm,
                api_key=settings.openai_api_key,
                model=settings.tts_model,
                voice=settings.tts_voice,
                client=tts_client,
            ),
            stop_event=stop_event,
            local_tool_inbox=local_tool_inbox,
            execute_tool=execute_tool,
            on_retry=_report_retry,
        )
    finally:
        if tts_client is not None:
            await tts_client.aclose()
        if home_assistant is not None:
            await home_assistant.close()
        await shutdown_observability()


def _report_retry(notice: RetryNotice) -> None:
    print(
        f"timer event reconnect {notice.attempts}; retry in "
        f"{notice.delay_seconds:.1f}s ({notice.error_type})",
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    try:
        settings = TimerClientSettings.from_environment()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
