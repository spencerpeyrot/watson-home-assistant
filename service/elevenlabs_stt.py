"""Turn-bounded realtime ElevenLabs speech recognition for Cube PCM audio."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode

from websockets.asyncio.client import connect


class ElevenLabsSTTError(RuntimeError):
    """Sanitized ElevenLabs transcription connection or protocol failure."""


class ElevenLabsSTT:
    ENDPOINT = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    CHUNK_BYTES = 4_800  # 100 ms, mono PCM16 at 24 kHz.

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

    async def transcribe(self, pcm_s16le: bytes) -> str:
        """Transcribe one VAD-bounded 24 kHz mono PCM16 turn."""

        if not pcm_s16le or len(pcm_s16le) % 2:
            raise ElevenLabsSTTError("input audio was not PCM16")

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
            while True:
                event = await self._receive_event(connection)
                message_type = event.get("message_type")
                if message_type == "committed_transcript":
                    text = event.get("text")
                    if not isinstance(text, str):
                        raise ElevenLabsSTTError(
                            "ElevenLabs returned an invalid transcript"
                        )
                    transcript = text.strip()
                    if len(transcript.encode("utf-8")) > 32_768:
                        raise ElevenLabsSTTError("ElevenLabs transcript exceeded its limit")
                    return transcript
                if message_type in {"error", "rate_limited"} or event.get("error"):
                    raise ElevenLabsSTTError("ElevenLabs rejected transcription")
        except ElevenLabsSTTError:
            raise
        except Exception as error:
            raise ElevenLabsSTTError(
                f"ElevenLabs transcription connection failed ({type(error).__name__})"
            ) from None
        finally:
            if connection is not None:
                try:
                    await connection.close()
                except Exception:
                    pass

    async def _wait_for_session(self, connection: Any) -> None:
        async with asyncio.timeout(10):
            while True:
                event = await self._receive_event(connection)
                message_type = event.get("message_type")
                if message_type == "session_started":
                    return
                if message_type in {"error", "rate_limited"} or event.get("error"):
                    raise ElevenLabsSTTError("ElevenLabs rejected transcription")

    @staticmethod
    async def _receive_event(connection: Any) -> dict[str, Any]:
        raw = await connection.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            event = json.loads(raw)
        except (TypeError, ValueError, UnicodeDecodeError):
            raise ElevenLabsSTTError("ElevenLabs returned an invalid transcript event") from None
        if not isinstance(event, dict):
            raise ElevenLabsSTTError("ElevenLabs returned an invalid transcript event")
        return event
