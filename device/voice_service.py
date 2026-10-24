"""Source-level Pi voice service composition; hardware policies stay injected."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from contracts.models import SessionResponse
from device.alsa import AlsaCapture, PcmFormat
from device.audio_runtime import PiAudioRuntime
from device.config import device_state_dir
from device.job_notifications import DeviceJobCompletionService
from device.led_ring import VoiceIndicator
from device.reconnect import RetryCallback
from device.session import GatewaySessionClient
from device.timer_chime import AlsaTimerChime
from device.startup_capture import StartupBufferedCapture
from device.session_control import SpeechDetector, WakeDetector
from device.voice_runtime import (
    InterruptDetector,
    LocalVoiceRuntime,
    TurnEndDetector,
)
from voice.job_orchestrator import VoiceJobCoordinator
from voice.observability import initialize as initialize_observability, shutdown as shutdown_observability


class PiVoiceService:
    """Own one local voice runtime and its long-lived HTTP clients."""

    def __init__(
        self,
        runtime: LocalVoiceRuntime,
        session_client: GatewaySessionClient,
        job_notifications: DeviceJobCompletionService,
        indicator: VoiceIndicator | None = None,
    ) -> None:
        self.runtime = runtime
        self.job_notifications = job_notifications
        self._session_client = session_client
        self._indicator = indicator
        self._closed = False

    async def run(self, stop_event: asyncio.Event) -> None:
        """Run device runtime, then close long-lived clients on every exit."""

        primary_error: BaseException | None = None
        indicator_stop = asyncio.Event()
        indicator_task: asyncio.Task[None] | None = None
        try:
            await asyncio.to_thread(initialize_observability, "voice")
            if self._indicator is not None:
                indicator_task = asyncio.create_task(
                    self._indicator.run(indicator_stop), name="cube-voice-led-ring"
                )
            await self.runtime.run(stop_event)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if indicator_task is not None:
                indicator_stop.set()
                await asyncio.gather(indicator_task, return_exceptions=True)
            if not self._closed:
                self._closed = True
                cleanup_error: Exception | None = None
                try:
                    await self.job_notifications.close()
                except Exception as error:
                    cleanup_error = error
                try:
                    await self._session_client.close()
                except Exception as error:
                    cleanup_error = cleanup_error or error
                await shutdown_observability()
                if primary_error is None and cleanup_error is not None:
                    raise cleanup_error


async def compose_pi_voice_service(
    *,
    gateway_url: str,
    device_id: str,
    device_token: str,
    conversation_id: UUID,
    locale: str,
    timezone_name: str,
    capture_pcm: str,
    playback_pcm: str,
    data_dir: str | Path | None = None,
    wake_detect: WakeDetector,
    speech_detect: SpeechDetector,
    interrupt_detect: InterruptDetector,
    turn_end_detect: TurnEndDetector,
    audio_format: PcmFormat = PcmFormat(),
    idle_timeout_seconds: float = 3.0,
    max_session_seconds: float | None = None,
    idle_check_interval_seconds: float = 0.1,
    idle_notification_interval_seconds: float = 15.0,
    on_retry: RetryCallback | None = None,
    indicator: VoiceIndicator | None = None,
) -> PiVoiceService:
    """Compose device clients and runtime without reading environment/secrets.

    Caller supplies provider-safe credentials, selected PCM names, and local
    wake/VAD policies. Construction opens no audio streams and makes no request.
    """

    resolved_data_dir = data_dir if data_dir is not None else device_state_dir()
    notifications = DeviceJobCompletionService(
        gateway_url=gateway_url,
        device_token=device_token,
        data_dir=resolved_data_dir,
        notify=AlsaTimerChime(playback_pcm, audio_format=audio_format).play,
    )
    session_client: GatewaySessionClient | None = None
    try:
        session_client = GatewaySessionClient(gateway_url, device_token)
        audio_runtime = PiAudioRuntime(
            capture_pcm=capture_pcm,
            playback_pcm=playback_pcm,
            audio_format=audio_format,
        )

        def session_callbacks(response: SessionResponse) -> VoiceJobCoordinator:
            return VoiceJobCoordinator(
                jobs=notifications.new_session_job_client(),
                device_id=device_id,
                conversation_id=conversation_id,
                session_id=response.session_id,
                locale=locale,
                timezone_name=timezone_name,
                pending_summaries=notifications.pending_summaries,
                mark_spoken=notifications.mark_spoken,
            )

        runtime = LocalVoiceRuntime(
            session_client=session_client,
            runtime=audio_runtime,
            idle_capture_factory=lambda: StartupBufferedCapture(
                AlsaCapture(capture_pcm, audio_format), audio_format
            ),
            gateway_url=gateway_url,
            device_id=device_id,
            device_token=device_token,
            conversation_id=conversation_id,
            locale=locale,
            timezone_name=timezone_name,
            wake_detect=wake_detect,
            speech_detect=speech_detect,
            interrupt_detect=interrupt_detect,
            turn_end_detect=turn_end_detect,
            session_callbacks_factory=session_callbacks,
            idle_timeout_seconds=idle_timeout_seconds,
            max_session_seconds=max_session_seconds,
            idle_check_interval_seconds=idle_check_interval_seconds,
            idle_notification_check=notifications,
            idle_notification_interval_seconds=idle_notification_interval_seconds,
            on_retry=on_retry,
            continuous_capture=True,
            # Pause idle closure during a candidate utterance without letting
            # VAD-only noise restart the follow-up countdown.
            speech_updates_idle=False,
            indicator_state=(indicator.set_state if indicator is not None else None),
            indicator_audio_level=(
                indicator.set_audio_level if indicator is not None else None
            ),
        )
        return PiVoiceService(runtime, session_client, notifications, indicator)
    except BaseException:
        await notifications.close()
        if session_client is not None:
            await session_client.close()
        raise
