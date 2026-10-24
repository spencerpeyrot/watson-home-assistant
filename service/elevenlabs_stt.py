"""Turn-bounded realtime ElevenLabs speech recognition for Cube PCM audio."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode

from websockets.asyncio.client import connect
from service.elevenlabs_diagnostics import safe_provider_value, transport_diagnostics
from voice.observability import current_observation


class ElevenLabsSTTError(RuntimeError):
    """Sanitized ElevenLabs transcription connection or protocol failure."""

    def __init__(self, message: str, **diagnostics: Any) -> None:
        super().__init__(message)
        self.observability = {"provider": "elevenlabs", "operation": "stt", **diagnostics}


class ElevenLabsSTT:
    ENDPOINT = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    CHUNK_BYTES = 4_800  # 100 ms, mono PCM16 at 24 kHz.
    TRANSCRIPT_TIMEOUT_SECONDS = 30

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "scribe_v2_realtime",
        connector: Callable[..., Any] = connect,
    ) -> None:
        if not api_key.strip():
            raise ValueError("ElevenLabs API key must not be blank")
        if model != "scribe_v2_realtime":
            raise ValueError("ElevenLabs STT model must be scribe_v2_realtime")
        self._api_key = api_key
        self._model = model
        self._connector = connector

    @property
    def model(self) -> str:
        return self._model

    async def transcribe(self, pcm_s16le: bytes) -> str:
        """Transcribe one VAD-bounded 24 kHz mono PCM16 turn."""

        started = time.monotonic()
        if not pcm_s16le or len(pcm_s16le) % 2:
            raise ElevenLabsSTTError("input audio was not PCM16", failure_stage="validate_input",
                                     error_category="invalid_local_input", retryable=False,
                                     model=self._model)

        query = urlencode(
            {
                "model_id": self._model,
                "audio_format": "pcm_24000",
                "commit_strategy": "manual",
                "language_code": "eng",
            }
        )
        connection = None
        try:
            connection = await self._connector(
                f"{self.ENDPOINT}?{query}",
                additional_headers={"xi-api-key": self._api_key},
                open_timeout=10,
                close_timeout=3,
                max_size=1_048_576,
                max_queue=32,
            )
            await self._wait_for_session(connection)
            for offset in range(0, len(pcm_s16le), self.CHUNK_BYTES):
                chunk = pcm_s16le[offset : offset + self.CHUNK_BYTES]
                await connection.send(
                    json.dumps(
                        {
                            "message_type": "input_audio_chunk",
                            "audio_base_64": base64.b64encode(chunk).decode("ascii"),
                            "commit": offset + len(chunk) == len(pcm_s16le),
                        }
                    )
                )
            try:
                async with asyncio.timeout(self.TRANSCRIPT_TIMEOUT_SECONDS):
                    while True:
                        event = await self._receive_event(connection)
                        message_type = event.get("message_type")
                        if message_type == "committed_transcript":
                            text = event.get("text")
                            if not isinstance(text, str):
                                raise ElevenLabsSTTError(
                                    "ElevenLabs returned an invalid transcript", failure_stage="decode_transcript",
                                    error_category="invalid_provider_payload", retryable=False
                                )
                            transcript = text.strip()
                            if len(transcript.encode("utf-8")) > 32_768:
                                raise ElevenLabsSTTError("ElevenLabs transcript exceeded its limit",
                                                         failure_stage="validate_transcript",
                                                         error_category="invalid_provider_payload", retryable=False)
                            return transcript
                        if message_type in {"error", "rate_limited"} or event.get("error"):
                            raise _provider_error(event)
            except TimeoutError:
                raise ElevenLabsSTTError("ElevenLabs transcription timed out",
                                         failure_stage="await_transcript", error_category="timeout",
                                         retryable=True) from None
        except TimeoutError as error:
            raise ElevenLabsSTTError(
                "ElevenLabs transcription timed out", failure_stage="transcribe",
                error_category="timeout", cause_type=type(error).__name__, retryable=True,
                model=self._model, elapsed_ms=round((time.monotonic() - started) * 1000),
            ) from None
        except ElevenLabsSTTError as error:
            error.observability.setdefault("model", self._model)
            error.observability.setdefault("elapsed_ms", round((time.monotonic() - started) * 1000))
            raise
        except Exception as error:
            details = transport_diagnostics(error)
            raise ElevenLabsSTTError(
                "ElevenLabs transcription connection failed", failure_stage="connect_or_stream",
                error_category=details.pop("error_category", "network"),
                retryable=details.pop("retryable", True), model=self._model,
                elapsed_ms=round((time.monotonic() - started) * 1000), **details,
            ) from None
        finally:
            if connection is not None:
                try:
                    await connection.close()
                except Exception as error:
                    current_observation().event(
                        "speech.connection_close_failed", status="failed",
                        error_type=type(error).__name__, error_category="cleanup_failed",
                    )

    async def _wait_for_session(self, connection: Any) -> None:
        try:
            async with asyncio.timeout(10):
                while True:
                    event = await self._receive_event(connection)
                    message_type = event.get("message_type")
                    if message_type == "session_started":
                        return
                    if message_type in {"error", "rate_limited"} or event.get("error"):
                        raise _provider_error(event)
        except TimeoutError:
            raise ElevenLabsSTTError("ElevenLabs session start timed out",
                                     failure_stage="wait_session", error_category="timeout",
                                     retryable=True, model=self._model) from None

    @staticmethod
    async def _receive_event(connection: Any) -> dict[str, Any]:
        raw = await connection.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            event = json.loads(raw)
        except (TypeError, ValueError, UnicodeDecodeError):
            raise ElevenLabsSTTError("ElevenLabs returned an invalid transcript event",
                                     failure_stage="decode_event",
                                     error_category="invalid_provider_payload", retryable=False) from None
        if not isinstance(event, dict):
            raise ElevenLabsSTTError("ElevenLabs returned an invalid transcript event",
                                     failure_stage="decode_event",
                                     error_category="invalid_provider_payload", retryable=False)
        return event


def _provider_error(event: dict[str, Any]) -> ElevenLabsSTTError:
    message_type = event.get("message_type")
    code = event.get("error_code", event.get("code"))
    category = "rate_limit" if message_type == "rate_limited" else "provider_rejected"
    return ElevenLabsSTTError(
        "ElevenLabs rejected transcription", failure_stage="provider_response",
        error_category=category,
        provider_error_code=safe_provider_value(code),
        provider_request_id=safe_provider_value(event.get("request_id")),
        retryable=category == "rate_limit",
    )
