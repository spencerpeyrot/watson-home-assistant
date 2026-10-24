"""OpenAI TTS presentation for durable timer events."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import httpx

from contracts.models import TimerFiredEvent
from device.alsa import AlsaPlayback, AudioDeviceError, PcmFormat
from device.timer_chime import AlsaTimerChime


LOGGER = logging.getLogger(__name__)
SPEECH_ENDPOINT = "https://api.openai.com/v1/audio/speech"
TTS_PCM_FORMAT = PcmFormat(sample_rate_hz=24_000, channels=1, frame_duration_ms=20)
SUPPORTED_VOICES = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "marin",
        "nova",
        "onyx",
        "sage",
        "shimmer",
        "verse",
        "cedar",
    }
)
PlayerFactory = Callable[[str, PcmFormat], AlsaPlayback]


class OpenAITimerSpeech:
    """Speak a timer's message with streamed PCM; use the chime on failure."""

    MODEL = "gpt-4o-mini-tts"
    VOICE = "marin"
    MAX_DRAIN_SECONDS = 120.0

    def __init__(
        self,
        *,
        pcm: str,
        api_key: str,
        model: str = MODEL,
        voice: str = VOICE,
        client: httpx.AsyncClient,
        fallback: AlsaTimerChime,
        player_factory: PlayerFactory = AlsaPlayback,
    ) -> None:
        if not pcm.strip():
            raise ValueError("timer TTS PCM name must not be blank")
        if not api_key.strip():
            raise ValueError("timer TTS API key must not be blank")
        if model != self.MODEL:
            raise ValueError("timer TTS model is unsupported")
        if voice not in SUPPORTED_VOICES:
            raise ValueError("timer TTS voice is unsupported")
        self._pcm = pcm
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._client = client
        self._fallback = fallback
        self._player_factory = player_factory

    async def __call__(self, event: TimerFiredEvent) -> None:
        try:
            await self._speak(event.message)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            LOGGER.warning(
                "Reminder TTS failed (%s); playing fallback chime",
                type(error).__name__,
            )
            await self._fallback(event)

    async def _speak(self, message: str) -> None:
        payload = {
            "model": self._model,
            "voice": self._voice,
            "input": f"Reminder: {message}",
            "instructions": (
                "Speak this reminder clearly and naturally in a masculine voice "
                "with a British English accent."
                if self._voice == "onyx"
                else "Speak this reminder clearly and naturally."
            ),
            "response_format": "pcm",
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with self._client.stream(
            "POST",
            SPEECH_ENDPOINT,
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            player = self._player_factory(self._pcm, TTS_PCM_FORMAT)
            await player.start()
            pending = bytearray()
            streamed_bytes = 0
            frame_bytes = TTS_PCM_FORMAT.frame_bytes
            try:
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    pending.extend(chunk)
                    aligned_bytes = len(pending) - (len(pending) % frame_bytes)
                    for offset in range(0, aligned_bytes, frame_bytes):
                        await player.write_frame(bytes(pending[offset : offset + frame_bytes]))
                    if aligned_bytes:
                        del pending[:aligned_bytes]
                        streamed_bytes += aligned_bytes

                if not pending and not streamed_bytes:
                    raise AudioDeviceError("OpenAI TTS returned empty PCM")
                if pending:
                    if len(pending) % 2:
                        raise AudioDeviceError("OpenAI TTS returned incomplete PCM sample")
                    pending.extend(bytes(frame_bytes - len(pending)))
                    await player.write_frame(bytes(pending))
                await player.drain_and_close(
                    timeout_seconds=self.MAX_DRAIN_SECONDS
                )
            except BaseException:
                await player.abort()
                raise


def make_timer_notifier(
    *,
    pcm: str,
    api_key: str,
    model: str,
    voice: str,
    client: httpx.AsyncClient | None,
) -> AlsaTimerChime | OpenAITimerSpeech:
    """Choose TTS when configured; retain chime-only operation without a key."""

    fallback = AlsaTimerChime(pcm)
    if not api_key.strip() or client is None:
        return fallback
    return OpenAITimerSpeech(
        pcm=pcm,
        api_key=api_key,
        model=model,
        voice=voice,
        client=client,
        fallback=fallback,
    )
