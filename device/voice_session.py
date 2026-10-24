"""Compose durable session creation, authenticated socket, and Pi audio relay."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol
from uuid import UUID

from contracts.models import SessionResponse
from device.audio_buffer import BufferedAudioFrame
from device.audio_runtime import PiAudioRuntime
from device.transport import DeviceSocket
from device.voice_relay import (
    CaptureFrameHandler,
    VoiceEventHandler,
    run_device_voice_relay,
)


class SessionLifecycle(Protocol):
    async def create_session(
        self, *, conversation_id: UUID, locale: str, timezone: str
    ) -> SessionResponse: ...

    async def close_session(self, session_id: UUID) -> None: ...


class VoiceSessionCallbacks(Protocol):
    """Session-scoped transcript/job hooks used by the local capture runtime."""

    async def on_event(self, event: VoiceSessionEvent, socket: DeviceSocket) -> None: ...

    def mark_audio_finished(self) -> None: ...

    async def close(self) -> None: ...


SocketFactory = Callable[..., Awaitable[DeviceSocket]]
VoiceEventHandlerFactory = Callable[[SessionResponse], VoiceEventHandler]
VoiceSessionCallbacksFactory = Callable[[SessionResponse], VoiceSessionCallbacks]
SessionCreatedHandler = Callable[[SessionResponse], Awaitable[None]]


async def run_gateway_voice_session(
    *,
    session_client: SessionLifecycle,
    runtime: PiAudioRuntime,
    gateway_url: str,
    device_id: str,
    device_token: str,
    conversation_id: UUID,
    locale: str,
    timezone: str,
    capture_preroll: Sequence[BufferedAudioFrame] = (),
    on_event: VoiceEventHandler | None = None,
    on_event_factory: VoiceEventHandlerFactory | None = None,
    session_callbacks_factory: VoiceSessionCallbacksFactory | None = None,
    on_session_created: SessionCreatedHandler | None = None,
    on_capture_frame: CaptureFrameHandler | None = None,
    socket_factory: SocketFactory = DeviceSocket.open,
) -> SessionResponse:
    """Create one durable session, relay its audio, then close gateway state."""

    if sum(
        callback is not None
        for callback in (on_event, on_event_factory, session_callbacks_factory)
    ) > 1:
        raise ValueError("provide only one voice session callback configuration")
    response = await session_client.create_session(
        conversation_id=conversation_id,
        locale=locale,
        timezone=timezone,
    )
    socket: DeviceSocket | None = None
    session_callbacks: VoiceSessionCallbacks | None = None
    primary_error: BaseException | None = None
    try:
        if session_callbacks_factory is not None:
            session_callbacks = session_callbacks_factory(response)
        if on_session_created is not None:
            await on_session_created(response)
        if session_callbacks is not None:
            event_handler = session_callbacks.on_event
        elif on_event_factory is not None:
            event_handler = on_event_factory(response)
        else:
            event_handler = on_event
        socket = await socket_factory(
            gateway_url,
            device_id,
            device_token,
            response.session_id,
        )
        async def relay() -> None:
            await run_device_voice_relay(
                session_id=response.session_id,
                socket=socket,
                runtime=runtime,
                capture_preroll=capture_preroll,
                on_event=event_handler,
                on_session_ready=(
                    session_callbacks.on_event if session_callbacks is not None else None
                ),
                on_capture_frame=on_capture_frame,
            )

        if response.max_session_seconds is None:
            await relay()
        else:
            duration_limit = asyncio.timeout(response.max_session_seconds)
            try:
                async with duration_limit:
                    await relay()
            except TimeoutError:
                if not duration_limit.expired():
                    raise
        return response
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_error: Exception | None = None
        if session_callbacks is not None:
            try:
                await session_callbacks.close()
            except Exception as error:
                cleanup_error = error
        if socket is not None:
            try:
                await socket.close()
            except Exception as error:
                cleanup_error = cleanup_error or error
        try:
            await session_client.close_session(response.session_id)
        except Exception as error:
            cleanup_error = cleanup_error or error
        if primary_error is None and cleanup_error is not None:
            raise cleanup_error
