"""Streaming ElevenLabs text-to-speech adapter for Cube's 24 kHz PCM path."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
from collections.abc import AsyncIterable, AsyncIterator, Callable
from urllib.parse import quote, urlencode
from typing import Any

from websockets.asyncio.client import connect


class ElevenLabsError(RuntimeError):
    """Sanitized ElevenLabs connection or audio protocol failure."""


class ElevenLabsTTS:
    ENDPOINT = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        *,
        model: str = "eleven_flash_v2_5",
        connector: Callable[..., Any] = connect,
    ) -> None:
        if not api_key.strip():
            raise ValueError("ElevenLabs API key must not be blank")
        if not voice_id.strip() or not model.strip():
            raise ValueError("ElevenLabs voice and model must not be blank")
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._connector = connector

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
                raw = await connection.recv()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                try:
                    event = json.loads(raw)
                except (TypeError, ValueError, UnicodeDecodeError):
                    raise ElevenLabsError("ElevenLabs returned an invalid audio event") from None
                if not isinstance(event, dict):
                    raise ElevenLabsError("ElevenLabs returned an invalid audio event")
                encoded = event.get("audio")
                if encoded is not None:
                    if not isinstance(encoded, str):
                        raise ElevenLabsError("ElevenLabs returned invalid audio data")
                    try:
                        pcm = base64.b64decode(encoded, validate=True)
                    except (binascii.Error, ValueError):
                        raise ElevenLabsError("ElevenLabs returned invalid audio data") from None
                    if not pcm or len(pcm) % 2:
                        raise ElevenLabsError("ElevenLabs audio was not PCM16")
                    yield pcm
                if event.get("is_final") is True or event.get("isFinal") is True:
                    await sender
                    return
                if event.get("error"):
                    raise ElevenLabsError("ElevenLabs rejected speech generation")
        except ElevenLabsError:
            raise
        except Exception as error:
            raise ElevenLabsError(
                f"ElevenLabs speech connection failed ({type(error).__name__})"
            ) from None
        finally:
            if sender is not None and not sender.done():
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            if connection is not None:
                try:
                    await connection.close()
                except Exception:
                    pass
