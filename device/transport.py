"""Authenticated device WebSocket transport for contract-shaped audio frames."""

from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from uuid import UUID, uuid4

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

from contracts.models import (
    AudioOutputFrameMetadata,
    AudioTurnEnd,
    VoiceContextAppend,
    VoiceSessionContextAppend,
    VoiceSessionEvent,
)
from device.audio_buffer import AudioFrame, AudioFrameBuffer


_LOGGER = logging.getLogger(__name__)


class DeviceTransportError(RuntimeError):
    """Sanitized device transport failure."""


def device_websocket_url(
    gateway_url: str, device_id: str, session_id: UUID | None = None
) -> str:
    """Build authenticated device route URL from an HTTP(S) gateway base."""

    if not gateway_url.strip() or not device_id.strip():
        raise ValueError("gateway URL and device ID must not be blank")
    parsed = urlsplit(gateway_url.rstrip("/"))
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.hostname:
        raise ValueError("gateway URL must be an absolute HTTP(S)/WS(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("gateway URL must not include embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("gateway URL must not include a query or fragment")
    if device_id in {".", ".."} or "/" in device_id or "\\" in device_id:
        raise ValueError("device ID must be one URL path segment")
    local_host = parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme in {"http", "ws"} and not local_host:
        raise ValueError("unencrypted device transport is allowed only on loopback")
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}[parsed.scheme]
    path_prefix = parsed.path.rstrip("/")
    path = f"{path_prefix}/v1/devices/{quote(device_id, safe='')}/connect"
    query = urlencode({"session_id": str(session_id)}) if session_id is not None else ""
    return urlunsplit((scheme, parsed.netloc, path, query, ""))


class DeviceSocket:
    """Small authenticated client; reconnect/session policy belongs to caller."""

    def __init__(self, connection: ClientConnection, *, session_id: UUID | None = None) -> None:
        self._connection = connection
        self._session_id = session_id
        self._next_output_sequence = 0
        self._send_lock = asyncio.Lock()

    @classmethod
    async def open(
        cls,
        gateway_url: str,
        device_id: str,
        device_token: str,
        session_id: UUID | None = None,
    ) -> DeviceSocket:
        if len(device_token) < 32:
            raise ValueError("device token must contain at least 32 characters")
        url = device_websocket_url(gateway_url, device_id, session_id)
        try:
            connection = await connect(
                url,
                additional_headers={"Authorization": f"Bearer {device_token}"},
                open_timeout=10,
                close_timeout=1,
                # The voice socket carries microphone frames continuously. The
                # default 20s ping + 20s pong timeout matches the Pi's ~41s
                # disconnect during active PCM, so avoid a ping-based cutoff.
                ping_interval=None,
                max_size=1_048_576,
                max_queue=32,
            )
        except (OSError, TimeoutError, WebSocketException) as error:
            raise DeviceTransportError(
                f"device gateway connection failed ({type(error).__name__})"
            ) from None
        return cls(connection, session_id=session_id)

    async def send_audio_frame(
        self,
        frame: AudioFrame,
        buffer: AudioFrameBuffer,
    ) -> bool:
        """Send metadata then its binary PCM; reject frames from old sessions."""

        if not await buffer.is_current(frame):
            return False
        if self._session_id is not None and frame.metadata.session_id != self._session_id:
            return False
        envelope = {
            "type": "audio_frame_metadata",
            "payload": frame.metadata.model_dump(mode="json"),
        }
        try:
            async with self._send_lock:
                await self._connection.send(json.dumps(envelope, separators=(",", ":")))
                await self._connection.send(frame.payload)
        except (WebSocketException, OSError) as error:
            close_detail = _websocket_close_detail(error)
            raise DeviceTransportError(
                f"device audio send failed ({type(error).__name__}{close_detail})"
            ) from None
        return True

    async def send_audio_turn_end(
        self, last_audio_sequence: int, *, vad_speech_ms: int = 0
    ) -> None:
        """Commit the just-sent VAD turn after its final PCM frame."""

        if self._session_id is None or last_audio_sequence < 0:
            raise ValueError("audio turn end requires an active session and frame")
        request = AudioTurnEnd(
            session_id=self._session_id,
            turn_id=uuid4(),
            last_audio_sequence=last_audio_sequence,
            vad_speech_ms=vad_speech_ms,
        )
        try:
            async with self._send_lock:
                await self._connection.send(
                    json.dumps(request.model_dump(mode="json"), separators=(",", ":"))
                )
        except (WebSocketException, OSError) as error:
            raise DeviceTransportError(
                f"audio turn-end send failed ({type(error).__name__})"
            ) from None

    async def send_context_append(self, request: VoiceContextAppend) -> None:
        """Send bounded job context for a known client delegation."""

        if self._session_id is None or request.session_id != self._session_id:
            raise ValueError("voice context must match the active socket session")
        envelope = {
            "type": "voice_context_append",
            "payload": request.model_dump(mode="json"),
        }
        try:
            async with self._send_lock:
                await self._connection.send(json.dumps(envelope, separators=(",", ":")))
        except (WebSocketException, OSError) as error:
            raise DeviceTransportError(
                f"voice context send failed ({type(error).__name__})"
            ) from None

    async def send_session_context_append(
        self, request: VoiceSessionContextAppend
    ) -> None:
        """Send bounded context authorized by the active session identity."""

        if self._session_id is None or request.session_id != self._session_id:
            raise ValueError("voice context must match the active socket session")
        envelope = {
            "type": "voice_session_context_append",
            "payload": request.model_dump(mode="json"),
        }
        try:
            async with self._send_lock:
                await self._connection.send(json.dumps(envelope, separators=(",", ":")))
        except (WebSocketException, OSError) as error:
            raise DeviceTransportError(
                f"voice context send failed ({type(error).__name__})"
            ) from None

    async def receive(self) -> str | bytes:
        """Receive one gateway message; caller validates its contract type."""

        try:
            return await self._connection.recv()
        except (WebSocketException, OSError) as error:
            close_detail = _websocket_close_detail(error)
            raise DeviceTransportError(
                f"device gateway receive failed ({type(error).__name__}{close_detail})"
            ) from None

    async def wait_until_ready(self) -> VoiceSessionEvent:
        """Wait for durable lookup and provider startup before capture begins."""

        message = await self.receive_voice_message()
        if isinstance(message, VoiceSessionEvent) and message.type == "session_error":
            _LOGGER.warning("gateway session startup rejected (code=%s)", message.code)
        if not isinstance(message, VoiceSessionEvent) or message.type != "session_ready":
            raise DeviceTransportError("gateway did not confirm voice session readiness")
        _LOGGER.info("gateway voice session ready")
        return message

    async def receive_voice_message(
        self,
    ) -> VoiceSessionEvent | tuple[AudioOutputFrameMetadata, bytes]:
        """Read one typed control event or one metadata-plus-binary playback frame."""

        if self._session_id is None:
            raise ValueError("session ID is required for voice messages")
        raw = await self.receive()
        if not isinstance(raw, str):
            raise DeviceTransportError("gateway sent binary audio without metadata")
        try:
            envelope = json.loads(raw)
            if not isinstance(envelope, dict):
                raise ValueError
            if envelope.get("type") == "audio_output_metadata":
                if set(envelope) != {"type", "payload"}:
                    raise ValueError
                metadata = AudioOutputFrameMetadata.model_validate(envelope["payload"])
                if metadata.session_id != self._session_id:
                    raise ValueError
                if metadata.sequence != self._next_output_sequence:
                    raise ValueError
                binary = await self.receive()
                if not isinstance(binary, bytes) or len(binary) != metadata.payload_bytes:
                    raise ValueError
                self._next_output_sequence += 1
                return metadata, binary
            event = VoiceSessionEvent.model_validate(envelope)
            if event.session_id != self._session_id:
                raise ValueError
            return event
        except (TypeError, ValueError, json.JSONDecodeError):
            raise DeviceTransportError("gateway voice message did not match the contract") from None

    async def close(self) -> None:
        await self._connection.close()


def _websocket_close_detail(error: BaseException) -> str:
    """Expose safe WebSocket close codes to diagnose transport shutdowns."""

    details: list[str] = []
    for label, attribute in (("sent", "sent"), ("received", "rcvd")):
        close = getattr(error, attribute, None)
        code = getattr(close, "code", None)
        reason = getattr(close, "reason", None)
        if code is None:
            continue
        detail = f"{label}_code={code}"
        if isinstance(reason, str) and reason:
            safe_reason = " ".join(reason.split())[:120]
            detail += f" {label}_reason={safe_reason!r}"
        details.append(detail)
    return f"; {'; '.join(details)}" if details else ""
