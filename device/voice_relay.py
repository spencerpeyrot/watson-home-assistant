"""Bind authenticated voice transport to bounded Pi capture and playback."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol
from uuid import UUID

from contracts.models import AudioOutputFrameMetadata, VoiceSessionEvent
from device.audio_buffer import AudioFrame, BufferedAudioFrame
from device.transport import DeviceTransportError, DeviceSocket


class AudioRuntime(Protocol):
    audio_format: Any
    capture_buffer: Any
    playback_buffer: Any

    async def start(
        self, session_id: UUID, *, capture_preroll: Sequence[BufferedAudioFrame] = ()
    ) -> None: ...

    async def next_capture_frame(self) -> AudioFrame | None: ...

    async def queue_playback_frame(self, session_id: UUID, payload: bytes) -> bool: ...

    async def drain_playback(self, *, reopen: bool = False) -> None: ...

    async def wait_failure(self) -> str: ...

    async def stop(self, session_id: UUID | None = None) -> bool: ...


VoiceEventHandler = Callable[[VoiceSessionEvent, DeviceSocket], Awaitable[None]]


@dataclass(frozen=True)
class CaptureFrameDecision:
    """Local capture handling result controlling upstream audio and turn end."""

    turn_ended: bool = False
    mute_audio: bool = False
    vad_speech_ms: int = 0


CaptureFrameHandler = Callable[
    [AudioFrame], Awaitable[CaptureFrameDecision | bool | None]
]
OUTPUT_FRAME_BYTES = 960  # 20 ms at 24 kHz mono PCM16.


async def run_device_voice_relay(
    *,
    session_id: UUID,
    socket: DeviceSocket,
    runtime: AudioRuntime,
    capture_preroll: Sequence[BufferedAudioFrame] = (),
    on_event: VoiceEventHandler | None = None,
    on_session_ready: VoiceEventHandler | None = None,
    on_capture_frame: CaptureFrameHandler | None = None,
) -> None:
    """Stream one ready session; stop all audio and socket work on any exit."""

    audio_format = runtime.audio_format
    if (
        audio_format.sample_rate_hz != 24_000
        or audio_format.channels != 1
        or audio_format.frame_bytes != OUTPUT_FRAME_BYTES
    ):
        raise ValueError("Pi voice relay requires 20 ms mono PCM16 at 24 kHz")

    runtime_started = False
    tasks: set[asyncio.Task[Any]] = set()
    try:
        ready_event = await socket.wait_until_ready()
        if on_session_ready is not None:
            await on_session_ready(ready_event, socket)
        await runtime.start(session_id, capture_preroll=capture_preroll)
        runtime_started = True

        async def send_microphone() -> None:
            while True:
                frame = await runtime.next_capture_frame()
                if frame is None:
                    return
                turn_ended = False
                vad_speech_ms = 0
                outgoing_frame = frame
                if on_capture_frame is not None:
                    decision = await on_capture_frame(frame)
                    if isinstance(decision, CaptureFrameDecision):
                        turn_ended = decision.turn_ended
                        vad_speech_ms = decision.vad_speech_ms
                        if decision.mute_audio:
                            outgoing_frame = replace(frame, payload=bytes(len(frame.payload)))
                    else:
                        turn_ended = bool(decision)
                sent = await socket.send_audio_frame(outgoing_frame, runtime.capture_buffer)
                if sent and turn_ended:
                    await socket.send_audio_turn_end(
                        frame.metadata.sequence, vad_speech_ms=vad_speech_ms
                    )

        async def receive_speaker() -> None:
            pending = bytearray()
            expected_sequence = 0
            while True:
                message = await socket.receive_voice_message()
                if isinstance(message, tuple):
                    metadata, payload = message
                    if not isinstance(metadata, AudioOutputFrameMetadata):
                        raise DeviceTransportError("gateway audio metadata was invalid")
                    if metadata.session_id != session_id or metadata.sequence != expected_sequence:
                        raise DeviceTransportError("gateway audio sequence or session changed")
                    expected_sequence += 1
                    pending.extend(payload)
                    while len(pending) >= OUTPUT_FRAME_BYTES:
                        frame = bytes(pending[:OUTPUT_FRAME_BYTES])
                        del pending[:OUTPUT_FRAME_BYTES]
                        await runtime.queue_playback_frame(session_id, frame)
                    continue

                if not isinstance(message, VoiceSessionEvent) or message.session_id != session_id:
                    raise DeviceTransportError("gateway voice event belonged to another session")
                if message.type == "assistant_response_complete":
                    # Provider completion can precede physical playback completion.
                    # Keep half-duplex input muted until ALSA drains the final PCM.
                    await runtime.drain_playback(reopen=True)
                if on_event is not None:
                    await on_event(message, socket)
                if message.type == "session_error":
                    raise DeviceTransportError("gateway voice session failed")
                if message.type == "session_closed":
                    if pending:
                        tail = bytes(pending)
                        tail += bytes(OUTPUT_FRAME_BYTES - len(tail))
                        await runtime.queue_playback_frame(session_id, tail)
                    await runtime.drain_playback()
                    return

        async def monitor_audio() -> None:
            detail = await runtime.wait_failure()
            raise DeviceTransportError(f"Pi audio runtime failed ({detail})")

        tasks = {
            asyncio.create_task(send_microphone(), name=f"cube-voice-send-{session_id}"),
            asyncio.create_task(receive_speaker(), name=f"cube-voice-play-{session_id}"),
            asyncio.create_task(monitor_audio(), name=f"cube-voice-audio-watch-{session_id}"),
        }
        done, pending_tasks = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        for task in done:
            if task.cancelled():
                continue
            error = task.exception()
            if error is not None:
                raise error
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            if runtime_started:
                await runtime.stop(session_id)
        finally:
            await socket.close()
