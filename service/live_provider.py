"""Server-side GPT-Live WebSocket client with explicit session finalization."""

from __future__ import annotations

import asyncio
import base64
import json
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal
from uuid import uuid4

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException


LIVE_SESSIONS_URL = "wss://api.openai.com/v1/live/sessions"
MAX_PCM_FRAME_BYTES = 9_600  # 200 ms, mono PCM16 at 24 kHz.
MAX_CONTEXT_BYTES = 500


class LiveProviderError(RuntimeError):
    """Sanitized GPT-Live connection, protocol, or lifecycle failure."""


class LiveFinalizationUnconfirmed(LiveProviderError):
    """The socket closed without a final session.closed event."""


@dataclass(frozen=True)
class LiveSessionClose:
    provider_session_id: str
    connected_seconds: Decimal | None
    final_event: dict[str, Any]


Connection = ClientConnection
Connector = Callable[..., Awaitable[Connection]]


class OpenAILiveSession:
    """Manage one primary server-side GPT-Live WebSocket session.

    Pass the API key from trusted server configuration. This class never loads
    environment files, logs credentials, or includes provider response bodies
    in raised errors.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-live-1",
        voice: str = "marin",
        instructions: str,
        startup_timeout_seconds: float = 20,
        close_timeout_seconds: float = 15,
        connector: Connector = connect,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key must not be blank")
        if not model.strip() or not voice.strip() or not instructions.strip():
            raise ValueError("Live model, voice, and instructions must not be blank")
        if startup_timeout_seconds <= 0 or close_timeout_seconds <= 0:
            raise ValueError("Live lifecycle timeouts must be positive")
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._startup_timeout = startup_timeout_seconds
        self._close_timeout = close_timeout_seconds
        self._connector = connector
        self._connection: Connection | None = None
        self._provider_session_id: str | None = None
        self._pending: deque[dict[str, Any]] = deque()
        self._closed_event: dict[str, Any] | None = None
        self._latest_usage_seconds: Decimal | None = None

    @property
    def provider_session_id(self) -> str | None:
        return self._provider_session_id

    @property
    def latest_usage_seconds(self) -> Decimal | None:
        """Latest cumulative usage snapshot, if provider sent one."""

        return self._latest_usage_seconds

    async def start(self) -> str:
        """Connect and wait for session.started before allowing audio sends."""

        if self._connection is not None:
            raise LiveProviderError("Live session already started")
        try:
            connection = await self._connector(
                LIVE_SESSIONS_URL,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
                open_timeout=self._startup_timeout,
                close_timeout=5,
                max_size=1_048_576,
                max_queue=32,
            )
            self._connection = connection
            await self._send(
                {
                    "type": "session.start",
                    "event_id": str(uuid4()),
                    "session": {
                        "model": self._model,
                        "instructions": self._instructions,
                        "audio": {
                            "format": {"type": "audio/pcm", "rate": 24_000},
                            "output": {"voice": self._voice},
                        },
                        "delegation": {"type": "client"},
                    },
                }
            )
            async with asyncio.timeout(self._startup_timeout):
                while True:
                    event = await self._receive_wire()
                    event_type = event.get("type")
                    if event_type == "error":
                        raise LiveProviderError("provider rejected session startup")
                    if event_type == "session.started":
                        provider_id = event.get("session", {}).get("id")
                        if not isinstance(provider_id, str) or not provider_id.strip():
                            raise LiveProviderError("provider startup event has no session ID")
                        self._provider_session_id = provider_id
                        return provider_id
                    self._remember(event)
        except LiveProviderError:
            await self._dispose_connection()
            raise
        except asyncio.CancelledError:
            await self._dispose_connection()
            raise
        except Exception as error:
            await self._dispose_connection()
            raise LiveProviderError(
                f"provider session startup failed ({type(error).__name__})"
            ) from None

    async def append_audio(self, pcm_s16le: bytes) -> None:
        """Append one bounded mono PCM16/24 kHz frame to the active session."""

        if self._provider_session_id is None or self._closed_event is not None:
            raise LiveProviderError("Live session is not active")
        if not pcm_s16le or len(pcm_s16le) > MAX_PCM_FRAME_BYTES or len(pcm_s16le) % 2:
            raise ValueError("audio frame must be 1–9600 even-length PCM bytes")
        await self._send(
            {
                "type": "session.input_audio.append",
                "audio": base64.b64encode(pcm_s16le).decode("ascii"),
            }
        )

    async def append_context(
        self,
        content: str,
        *,
        kind: Literal["commentary", "thinking", "instructions"],
        delegation_id: str | None,
    ) -> str:
        """Append concise task context, correlated to its client delegation."""

        if self._provider_session_id is None or self._closed_event is not None:
            raise LiveProviderError("Live session is not active")
        if not content.strip() or len(content.encode("utf-8")) > MAX_CONTEXT_BYTES:
            raise ValueError("Live context must contain 1–500 UTF-8 bytes")
        if delegation_id is not None and not delegation_id.strip():
            raise ValueError("Live delegation ID must not be blank")
        event_id = str(uuid4())
        await self._send(
            {
                "type": f"session.{kind}.append",
                "event_id": event_id,
                "delegation_id": delegation_id,
                "content": content,
            }
        )
        return event_id

    async def receive_event(self) -> dict[str, Any]:
        """Return the next provider event; cumulative usage snapshots replace prior values."""

        if self._pending:
            return self._pending.popleft()
        if self._closed_event is not None:
            raise LiveProviderError("Live session is finalized")
        event = await self._receive_wire()
        self._track(event)
        return event

    async def close(self) -> LiveSessionClose:
        """Request graceful close and wait for final provider usage event."""

        if self._provider_session_id is None:
            await self._dispose_connection()
            raise LiveFinalizationUnconfirmed("Live session never reached session.started")
        if self._closed_event is None:
            try:
                await self._send({"type": "session.close", "event_id": str(uuid4())})
                async with asyncio.timeout(self._close_timeout):
                    while self._closed_event is None:
                        event = await self._receive_wire()
                        if event.get("type") == "error":
                            self._remember(event)
                        else:
                            self._remember(event)
            except asyncio.CancelledError:
                await self._dispose_connection()
                raise
            except (TimeoutError, OSError, WebSocketException, LiveProviderError) as error:
                await self._dispose_connection()
                if isinstance(error, LiveProviderError):
                    raise LiveFinalizationUnconfirmed(
                        "provider close did not produce session.closed"
                    ) from None
                raise LiveFinalizationUnconfirmed(
                    f"provider finalization unconfirmed ({type(error).__name__})"
                ) from None

        final_event = self._closed_event
        assert final_event is not None
        usage = final_event.get("usage")
        raw_seconds = usage.get("seconds") if isinstance(usage, dict) else None
        seconds = self._usage_seconds(raw_seconds)
        result = LiveSessionClose(
            provider_session_id=self._provider_session_id,
            connected_seconds=seconds,
            final_event=final_event,
        )
        await self._dispose_connection()
        return result

    async def abort(self) -> None:
        """Release transport after an unrecoverable provider or client failure."""

        await self._dispose_connection()

    async def _send(self, payload: dict[str, Any]) -> None:
        connection = self._connection
        if connection is None:
            raise LiveProviderError("Live connection is unavailable")
        try:
            await connection.send(json.dumps(payload, separators=(",", ":")))
        except Exception as error:
            raise LiveProviderError(
                f"provider send failed ({type(error).__name__})"
            ) from None

    async def _receive_wire(self) -> dict[str, Any]:
        connection = self._connection
        if connection is None:
            raise LiveProviderError("Live connection is unavailable")
        try:
            raw = await connection.recv()
            if not isinstance(raw, str):
                raise LiveProviderError("provider sent a non-text control event")
            event = json.loads(raw)
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                raise LiveProviderError("provider sent an invalid control event")
            return event
        except LiveProviderError:
            raise
        except Exception as error:
            raise LiveProviderError(
                f"provider receive failed ({type(error).__name__})"
            ) from None

    def _remember(self, event: dict[str, Any]) -> None:
        self._track(event)
        self._pending.append(event)

    def _track(self, event: dict[str, Any]) -> None:
        if event.get("type") == "session.usage.updated":
            usage = event.get("usage")
            seconds = usage.get("seconds") if isinstance(usage, dict) else None
            parsed = self._usage_seconds(seconds)
            if parsed is not None:
                self._latest_usage_seconds = parsed
        if event.get("type") == "session.closed":
            self._closed_event = event

    @staticmethod
    def _usage_seconds(value: Any) -> Decimal | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            seconds = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
        if not seconds.is_finite() or seconds < 0:
            return None
        return seconds

    async def _dispose_connection(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                await connection.close()
            except Exception:
                pass
