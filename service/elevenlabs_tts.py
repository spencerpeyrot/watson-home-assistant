"""Streaming ElevenLabs text-to-speech adapter for Cube's 24 kHz PCM path."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable
from urllib.parse import quote, urlencode
from typing import Any

from websockets.asyncio.client import connect
from service.elevenlabs_diagnostics import safe_provider_value, transport_diagnostics
from voice.observability import current_observation


class ElevenLabsError(RuntimeError):
    """Sanitized ElevenLabs connection or audio protocol failure."""

    def __init__(self, message: str, **diagnostics: Any) -> None:
        super().__init__(message)
        self.observability = {"provider": "elevenlabs", "operation": "tts", **diagnostics}


class ElevenLabsTTS:
    ENDPOINT = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
    STREAM_IDLE_TIMEOUT_SECONDS = 30.0

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        *,
        model: str = "eleven_flash_v2_5",
        connector: Callable[..., Any] = connect,
        stream_idle_timeout_seconds: float = STREAM_IDLE_TIMEOUT_SECONDS,
    ) -> None:
        if not api_key.strip():
            raise ValueError("ElevenLabs API key must not be blank")
        if not voice_id.strip() or not model.strip():
            raise ValueError("ElevenLabs voice and model must not be blank")
        if stream_idle_timeout_seconds <= 0:
            raise ValueError("ElevenLabs stream idle timeout must be positive")
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._connector = connector
        self._stream_idle_timeout_seconds = stream_idle_timeout_seconds

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text_chunks: AsyncIterable[str]) -> AsyncIterator[bytes]:
        """Send incremental text and yield validated PCM16 at 24 kHz."""

        query = urlencode(
            {
                "model_id": self._model,
                "output_format": "pcm_24000",
                "language_code": "en",
            }
        )
        url = f"{self.ENDPOINT.format(voice_id=quote(self._voice_id, safe=''))}?{query}"
        connection = None
        sender: Any | None = None
        receive: asyncio.Task | None = None
        audio_bytes = 0
        started = time.monotonic()
        try:
            connection = await self._connector(
                url,
                open_timeout=10,
                close_timeout=3,
                max_size=1_048_576,
                max_queue=32,
            )
            await connection.send(
                json.dumps(
                    {
                        "text": " ",
                        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                        "generation_config": {"chunk_length_schedule": [50, 120, 160, 290]},
                        "xi_api_key": self._api_key,
                    }
                )
            )

            async def send_text() -> None:
                pending_text: str | None = None
                async for text in text_chunks:
                    if not isinstance(text, str) or not text:
                        continue
                    if pending_text is not None:
                        await connection.send(json.dumps({"text": pending_text}))
                    pending_text = text
                if pending_text is not None:
                    # ElevenLabs buffers short text turns unless the final chunk
                    # flushes the generation buffer.
                    await connection.send(json.dumps({"text": pending_text, "flush": True}))
                await connection.send(json.dumps({"text": ""}))

            sender = asyncio.create_task(send_text())
            while True:
                receive = asyncio.create_task(connection.recv())
                done, _ = await asyncio.wait((receive, sender), return_when=asyncio.FIRST_COMPLETED)
                if sender in done:
                    await sender
                try:
                    async with asyncio.timeout(self._stream_idle_timeout_seconds):
                        raw = await receive
                except TimeoutError:
                    raise ElevenLabsError(
                        "ElevenLabs audio stream stopped making progress",
                        failure_stage="await_audio", error_category="timeout", retryable=True,
                        model=self._model, timeout_ms=round(self._stream_idle_timeout_seconds * 1000),
                        elapsed_ms=round((time.monotonic() - started) * 1000),
                        audio_bytes=audio_bytes, output_started=audio_bytes > 0,
                    ) from None
                receive = None
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                try:
                    event = json.loads(raw)
                except (TypeError, ValueError, UnicodeDecodeError):
                    raise ElevenLabsError("ElevenLabs returned an invalid audio event",
                                          failure_stage="decode_event",
                                          error_category="invalid_provider_payload", retryable=False) from None
                if not isinstance(event, dict):
                    raise ElevenLabsError("ElevenLabs returned an invalid audio event",
                                          failure_stage="decode_event",
                                          error_category="invalid_provider_payload", retryable=False)
                encoded = event.get("audio")
                if encoded is not None:
                    if not isinstance(encoded, str):
                        raise ElevenLabsError("ElevenLabs returned invalid audio data",
                                              failure_stage="decode_audio",
                                              error_category="invalid_provider_payload", retryable=False)
                    try:
                        pcm = base64.b64decode(encoded, validate=True)
                    except (binascii.Error, ValueError):
                        raise ElevenLabsError("ElevenLabs returned invalid audio data",
                                              failure_stage="decode_audio",
                                              error_category="invalid_provider_payload", retryable=False) from None
                    if not pcm or len(pcm) % 2:
                        raise ElevenLabsError("ElevenLabs audio was not PCM16",
                                              failure_stage="validate_audio",
                                              error_category="invalid_provider_payload", retryable=False)
                    audio_bytes += len(pcm)
                    yield pcm
                if event.get("is_final") is True or event.get("isFinal") is True:
                    await sender
                    return
                if event.get("error"):
                    error = event.get("error")
                    code = error.get("code") if isinstance(error, dict) else None
                    status = error.get("status") if isinstance(error, dict) else None
                    raise ElevenLabsError(
                        "ElevenLabs rejected speech generation", failure_stage="provider_response",
                        error_category="rate_limit" if status == 429 else "provider_rejected",
                        provider_error_code=safe_provider_value(code),
                        provider_request_id=safe_provider_value(error.get("request_id")) if isinstance(error, dict) else None,
                        http_status=status if isinstance(status, int) else None,
                        retryable=status == 429, model=self._model,
                    )
        except TimeoutError as error:
            raise ElevenLabsError("ElevenLabs speech generation timed out", failure_stage="stream",
                                  error_category="timeout", cause_type=type(error).__name__,
                                  retryable=True, model=self._model,
                                  elapsed_ms=round((time.monotonic() - started) * 1000),
                                  audio_bytes=audio_bytes, output_started=audio_bytes > 0) from None
        except ElevenLabsError as error:
            error.observability.update(model=self._model,
                                       elapsed_ms=round((time.monotonic() - started) * 1000),
                                       audio_bytes=audio_bytes, output_started=audio_bytes > 0)
            raise
        except Exception as error:
            details = transport_diagnostics(error)
            raise ElevenLabsError(
                "ElevenLabs speech connection failed", failure_stage="connect_or_stream",
                error_category=details.pop("error_category", "network"),
                retryable=details.pop("retryable", True), model=self._model,
                elapsed_ms=round((time.monotonic() - started) * 1000), **details,
                audio_bytes=audio_bytes, output_started=audio_bytes > 0,
            ) from None
        finally:
            if receive is not None and not receive.done():
                receive.cancel()
                await asyncio.gather(receive, return_exceptions=True)
            if sender is not None and not sender.done():
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            if connection is not None:
                try:
                    await connection.close()
                except Exception as error:
                    current_observation().event(
                        "speech.connection_close_failed", status="failed",
                        error_type=type(error).__name__, error_category="cleanup_failed",
                    )
