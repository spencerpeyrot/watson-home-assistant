"""Environment-composed Pi voice client."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from device.alsa import PcmFormat
from device.config import device_state_dir
from device.led_ring import Pi5VoiceRing
from device.openwakeword_detector import OpenWakeWordDetector
from device.reconnect import RetryNotice
from device.speech_activity import WebRtcSpeechActivity
from device.voice_service import compose_pi_voice_service


LOGGER = logging.getLogger(__name__)
DEFAULT_WAKEWORD_DIR = (
    Path.home() / ".local" / "state" / "cube-assistant" / "openwakeword"
)


@dataclass(frozen=True)
class VoiceClientSettings:
    gateway_url: str
    device_id: str
    device_token: str = field(repr=False)
    conversation_id: UUID
    locale: str
    timezone_name: str
    capture_pcm: str
    playback_pcm: str
    wakeword_model: Path
    wakeword_asset_dir: Path
    wakeword_threshold: float
    wakeword_confirmation_windows: int
    vad_mode: int
    vad_minimum_speech_ms: int
    vad_endpoint_silence_ms: int
    idle_timeout_seconds: float
    data_dir: Path

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> VoiceClientSettings:
        values = os.environ if environ is None else environ
        gateway_url = values.get("CUBE_GATEWAY_URL", "").strip()
        device_id = values.get("CUBE_DEVICE_ID", "").strip()
        device_token = values.get("CUBE_DEVICE_TOKEN", "")
        raw_conversation_id = values.get("CUBE_CONVERSATION_ID", "").strip()
        locale = values.get("CUBE_VOICE_LOCALE", "en-US").strip()
        timezone_name = values.get("CUBE_TIMEZONE", "").strip()
        capture_pcm = values.get(
            "CUBE_VOICE_CAPTURE_PCM", "plughw:CARD=seeed2micvoicec,DEV=0"
        ).strip()
        playback_pcm = values.get(
            "CUBE_VOICE_PLAYBACK_PCM", "plughw:CARD=seeed2micvoicec,DEV=0"
        ).strip()
        wakeword_dir = Path(
            values.get("CUBE_WAKEWORD_ASSET_DIR", str(DEFAULT_WAKEWORD_DIR))
        ).expanduser()
        wakeword_model = Path(
            values.get(
                "CUBE_WAKEWORD_MODEL", str(wakeword_dir / "hey_watson.onnx")
            )
        ).expanduser()

        try:
            conversation_id = UUID(raw_conversation_id)
        except ValueError:
            raise RuntimeError("CUBE_CONVERSATION_ID must be a UUID") from None
        if not gateway_url:
            raise RuntimeError("CUBE_GATEWAY_URL is required")
        if not device_id:
            raise RuntimeError("CUBE_DEVICE_ID is required")
        if len(device_token) < 32 or any(character.isspace() for character in device_token):
            raise RuntimeError(
                "CUBE_DEVICE_TOKEN must be at least 32 characters without whitespace"
            )
        if not locale:
            raise RuntimeError("CUBE_VOICE_LOCALE must not be blank")
        if not timezone_name:
            raise RuntimeError("CUBE_TIMEZONE is required")
        try:
            ZoneInfo(timezone_name)
        except (ValueError, ZoneInfoNotFoundError):
            raise RuntimeError("CUBE_TIMEZONE must be a valid IANA timezone") from None
        if not capture_pcm or not playback_pcm:
            raise RuntimeError("voice capture and playback PCM names are required")

        try:
            threshold = float(values.get("CUBE_WAKEWORD_THRESHOLD", "0.5"))
            confirmation_windows = int(
                values.get("CUBE_WAKEWORD_CONFIRMATION_WINDOWS", "3")
            )
            vad_mode = int(values.get("CUBE_VAD_MODE", "2"))
            minimum_speech_ms = int(
                values.get("CUBE_VAD_MINIMUM_SPEECH_MS", "60")
            )
            endpoint_silence_ms = int(
                values.get("CUBE_VAD_ENDPOINT_SILENCE_MS", "600")
            )
            idle_timeout_seconds = float(
                values.get("CUBE_VOICE_IDLE_TIMEOUT_SECONDS", "3")
            )
        except ValueError:
            raise RuntimeError("voice timing, threshold, or VAD setting is invalid") from None
        if not 0 < threshold <= 1:
            raise RuntimeError("CUBE_WAKEWORD_THRESHOLD must be greater than 0 and at most 1")
        if not 1 <= confirmation_windows <= 10:
            raise RuntimeError(
                "CUBE_WAKEWORD_CONFIRMATION_WINDOWS must be from 1 to 10"
            )
        if vad_mode not in range(4):
            raise RuntimeError("CUBE_VAD_MODE must be from 0 to 3")
        if minimum_speech_ms < 20 or minimum_speech_ms % 20:
            raise RuntimeError("CUBE_VAD_MINIMUM_SPEECH_MS must align to 20 ms frames")
        if endpoint_silence_ms < 20 or endpoint_silence_ms % 20:
            raise RuntimeError("CUBE_VAD_ENDPOINT_SILENCE_MS must align to 20 ms frames")
        if not 1 <= idle_timeout_seconds <= 600:
            raise RuntimeError("CUBE_VOICE_IDLE_TIMEOUT_SECONDS must be from 1 to 600")

        return cls(
            gateway_url=gateway_url,
            device_id=device_id,
            device_token=device_token,
            conversation_id=conversation_id,
            locale=locale,
            timezone_name=timezone_name,
            capture_pcm=capture_pcm,
            playback_pcm=playback_pcm,
            wakeword_model=wakeword_model,
            wakeword_asset_dir=wakeword_dir,
            wakeword_threshold=threshold,
            wakeword_confirmation_windows=confirmation_windows,
            vad_mode=vad_mode,
            vad_minimum_speech_ms=minimum_speech_ms,
            vad_endpoint_silence_ms=endpoint_silence_ms,
            idle_timeout_seconds=idle_timeout_seconds,
            data_dir=device_state_dir(values),
        )


async def run(settings: VoiceClientSettings) -> None:
    """Run the always-on local wake listener until SIGINT or SIGTERM."""

    settings.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_number, stop_event.set)
        except (NotImplementedError, RuntimeError):
            continue

    activity = WebRtcSpeechActivity(
        mode=settings.vad_mode,
        minimum_speech_ms=settings.vad_minimum_speech_ms,
        endpoint_silence_ms=settings.vad_endpoint_silence_ms,
    )
    wake_detector = OpenWakeWordDetector(
        model_path=settings.wakeword_model,
        melspectrogram_model_path=settings.wakeword_asset_dir / "melspectrogram.onnx",
        embedding_model_path=settings.wakeword_asset_dir / "embedding_model.onnx",
        input_sample_rate_hz=PcmFormat().sample_rate_hz,
        input_channels=PcmFormat().channels,
        input_frame_duration_ms=PcmFormat().frame_duration_ms,
        threshold=settings.wakeword_threshold,
        confirmation_windows=settings.wakeword_confirmation_windows,
    )

    def no_interrupt_detection(_payload: bytes) -> bool:
        return False

    try:
        indicator = Pi5VoiceRing()
    except Exception as error:
        LOGGER.warning(
            "voice status ring unavailable (%s: %s)", type(error).__name__, error
        )
        indicator = None

    service = await compose_pi_voice_service(
        gateway_url=settings.gateway_url,
        device_id=settings.device_id,
        device_token=settings.device_token,
        conversation_id=settings.conversation_id,
        locale=settings.locale,
        timezone_name=settings.timezone_name,
        capture_pcm=settings.capture_pcm,
        playback_pcm=settings.playback_pcm,
        data_dir=settings.data_dir,
        wake_detect=wake_detector,
        speech_detect=activity,
        interrupt_detect=no_interrupt_detection,
        turn_end_detect=activity.turn_end_detect,
        audio_format=PcmFormat(),
        idle_timeout_seconds=settings.idle_timeout_seconds,
        on_retry=_report_retry,
        indicator=indicator,
    )
    LOGGER.warning(
        "voice client started; local VAD is uncalibrated and interruption detection is disabled"
    )
    await service.run(stop_event)


def _report_retry(notice: RetryNotice) -> None:
    LOGGER.warning(
        "voice reconnect %s; retry in %.1fs (%s)",
        notice.attempts,
        notice.delay_seconds,
        notice.error_type,
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        settings = VoiceClientSettings.from_environment()
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
