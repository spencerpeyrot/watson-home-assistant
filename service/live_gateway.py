"""Authenticated device-to-GPT-Live relay for one durable session."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import UUID

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from contracts.models import (
    AudioFrameMetadata,
    AudioOutputFrameMetadata,
    AudioTurnEnd,
    SessionResponse,
    VoiceContextAppend,
    VoiceSessionContextAppend,
    VoiceSessionEvent,
)
from service.live_provider import LiveProviderError, LiveSessionClose
from service.live_usage import persist_live_session_usage
from service.postgres import PostgresStore
from voice.observability import Observation, start_session
from voice.audio_levels import PcmLevelMeter


MAX_INPUT_FRAME_BYTES = 9_600
MAX_OUTPUT_FRAME_BYTES = 9_600
OUTPUT_FRAME_CHUNK_BYTES = MAX_OUTPUT_FRAME_BYTES
MAX_CLIENT_DELEGATIONS = 512
_LOGGER = logging.getLogger(__name__)


class VoiceProvider(Protocol):
    async def start(self) -> str: ...

    async def append_audio(self, pcm_s16le: bytes) -> None: ...

    async def append_context(
        self,
        content: str,
        *,
        kind: str,
        delegation_id: str | None,
    ) -> str: ...

    async def receive_event(self) -> dict[str, Any]: ...

    async def close(self) -> LiveSessionClose | None: ...

    async def abort(self) -> None: ...


VoiceAuthenticator = Callable[[WebSocket], Awaitable[Any]]
VoiceProviderFactory = Callable[[Any, SessionResponse], VoiceProvider]


class InvalidVoiceFrame(ValueError):
    """Device sent a frame outside the versioned voice transport contract."""


async def serve_voice_session(
    websocket: WebSocket,
    *,
    device_id: str,
    session_id: UUID,
    store: PostgresStore,
    authenticate: VoiceAuthenticator,
    provider_factory: VoiceProviderFactory,
) -> None:
    """Authenticate, claim, relay, and close one device voice session."""

    admission = start_session(session_id, "gateway.admission")
    admission_started = asyncio.get_running_loop().time()

    def finish_admission(**metadata: Any) -> None:
        admission.update(elapsed_ms=round((asyncio.get_running_loop().time() - admission_started) * 1000),
                         **metadata)
        admission.end()

    try:
        principal = await authenticate(websocket)
    except Exception as error:
        finish_admission(status="rejected", failure_stage="authenticate",
                         error_type=type(error).__name__, error_category="authentication_failed")
        await websocket.close(code=1008, reason="authentication rejected")
        return

    owner_id = getattr(principal, "owner_id", None)
    principal_device = getattr(principal, "device_id", None)
    if (
        getattr(principal, "role", None) != "device"
        or owner_id is None
        or not isinstance(principal_device, str)
        or not principal_device.strip()
        or principal_device != device_id
    ):
        finish_admission(status="rejected", failure_stage="device_scope",
                         error_category="device_scope_rejected")
        await websocket.close(code=1008, reason="device scope rejected")
        return

    try:
        session = await store.get_connecting_session_for_device(
            owner_id=owner_id, device_id=device_id, session_id=session_id
        )
    except Exception as error:
        finish_admission(status="failed", failure_stage="session_lookup",
                         error_type=type(error).__name__, error_category="storage_unavailable")
        await websocket.close(code=1013, reason="session lookup unavailable")
        return
    if session is None:
        finish_admission(status="rejected", failure_stage="session_lookup",
                         error_category="session_scope_rejected")
        await websocket.close(code=1008, reason="session scope rejected")
        return
    if _remaining_session_seconds(session) == 0:
        try:
            await store.close_session(
                owner_id=owner_id, session_id=session_id, device_id=device_id
            )
        except Exception as error:
            finish_admission(status="failed", failure_stage="close_expired_session",
                             error_type=type(error).__name__, error_category="storage_unavailable")
            await websocket.close(code=1013, reason="session cleanup unavailable")
            return
        finish_admission(status="rejected", failure_stage="session_duration",
                         error_category="session_limit_reached")
        await websocket.close(code=1008, reason="session duration limit reached")
        return

    await websocket.accept()
    try:
        claimed_start = await store.claim_session_start(
            owner_id=owner_id, device_id=device_id, session_id=session_id
        )
    except Exception as error:
        finish_admission(status="failed", failure_stage="claim_session",
                         error_type=type(error).__name__, error_category="storage_unavailable")
        await websocket.close(code=1013, reason="session claim unavailable")
        return
    if not claimed_start:
        finish_admission(status="rejected", failure_stage="claim_session",
                         error_category="session_already_claimed")
        await websocket.close(code=1008, reason="session already claimed")
        return
    finish_admission(status="succeeded")

    provider: VoiceProvider | None = None
    provider_started = False
    tasks: set[asyncio.Task[Any]] = set()
    session_error_code: str | None = None
    close_reason = "relay_ended"
    cancellation: asyncio.CancelledError | None = None
    trace = start_session(session_id, "voice.gateway")

    try:
        provider = provider_factory(principal, session)
        startup_limit = _remaining_session_seconds(session)
        with trace.operation("provider.start"):
            if startup_limit is None:
                await provider.start()
            else:
                async with asyncio.timeout(startup_limit):
                    await provider.start()
        provider_started = True
        trace.event("provider.ready")
        claimed = await store.mark_session_ready(
            owner_id=owner_id, device_id=device_id, session_id=session_id
        )
        if not claimed:
            raise InvalidVoiceFrame("session is no longer connecting")
        await _send_control(
            websocket,
            VoiceSessionEvent(
                type="session_ready",
                session_id=session_id,
                half_duplex=(
                    True if getattr(provider, "half_duplex", False) is True else None
                ),
            ),
        )

        known_client_delegations: set[str] = set()
        async def relay_device_audio() -> None:
            with trace.activate():
                await _relay_device_audio(
                    websocket,
                    session_id=session_id,
                    provider=provider,
                    known_client_delegations=known_client_delegations,
                    trace=trace,
                )

        async def relay_provider_events() -> None:
            with trace.activate():
                await _relay_provider_events(
                    websocket,
                    session_id=session_id,
                    provider=provider,
                    known_client_delegations=known_client_delegations,
                    trace=trace,
                )

        tasks = {
            asyncio.create_task(
                relay_device_audio(), name=f"cube-live-input-{session_id}"
            ),
            asyncio.create_task(
                relay_provider_events(), name=f"cube-live-output-{session_id}"
            ),
        }
        remaining_seconds = _remaining_session_seconds(session)
        if remaining_seconds is None:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
        else:
            done, pending = await asyncio.wait(
                tasks,
                timeout=remaining_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done and pending:
                # Reaching the configured cap is expected closure, not a provider error.
                close_reason = "max_session_duration"
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            close_reason = task.get_name()
            _LOGGER.info(
                "voice gateway relay task completed task=%s session_id=%s",
                task.get_name(),
                session_id,
            )
            if task.cancelled():
                continue
            error = task.exception()
            if error is not None:
                _LOGGER.warning(
                    "voice gateway task failed task=%s error_type=%s",
                    task.get_name(),
                    type(error).__name__,
                )
                if isinstance(error, InvalidVoiceFrame):
                    session_error_code = "INVALID_REQUEST"
                elif isinstance(error, (TimeoutError, LiveProviderError)):
                    session_error_code = "DEPENDENCY_UNAVAILABLE"
                else:
                    session_error_code = "INTERNAL_ERROR"
                close_reason = f"{task.get_name()}:{session_error_code}"
                break
    except Exception as error:
        close_reason = f"gateway:{type(error).__name__}"
        _LOGGER.warning(
            "voice gateway failed error_type=%s",
            type(error).__name__,
        )
        session_error_code = (
            "DEPENDENCY_TIMEOUT" if isinstance(error, TimeoutError) else "DEPENDENCY_UNAVAILABLE"
        )
    finally:
        _LOGGER.info(
            "voice gateway session closing session_id=%s reason=%s error_code=%s",
            session_id,
            close_reason,
            session_error_code,
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        if session_error_code is not None:
            await _try_send_control(
                websocket,
                VoiceSessionEvent(
                    type="session_error",
                    session_id=session_id,
                    code=session_error_code,
                ),
            )

        if provider is not None:
            if provider_started:
                try:
                    closed = await provider.close()
                except asyncio.CancelledError as error:
                    cancellation = error
                    try:
                        await provider.abort()
                    except Exception as abort_error:
                        trace.event("voice.provider_abort_failed", status="failed",
                                    failure_stage="abort", error_type=type(abort_error).__name__,
                                    error_category="cleanup_failed")
                except Exception as error:
                    trace.event("voice.provider_close_failed", status="failed",
                                failure_stage="provider_close", error_type=type(error).__name__,
                                error_category="cleanup_failed")
                    try:
                        await provider.abort()
                    except Exception as abort_error:
                        trace.event("voice.provider_abort_failed", status="failed",
                                    failure_stage="abort", error_type=type(abort_error).__name__,
                                    error_category="cleanup_failed")
                else:
                    if closed is not None:
                        try:
                            await persist_live_session_usage(
                                store,
                                owner_id=owner_id,
                                session_id=session_id,
                                closed=closed,
                            )
                        except Exception as error:
                            trace.event("voice.usage_persist_failed", status="failed",
                                        failure_stage="persist_usage", error_type=type(error).__name__,
                                        error_category="storage_failed")
            else:
                try:
                    await provider.abort()
                except Exception as error:
                    trace.event("voice.provider_abort_failed", status="failed",
                                failure_stage="abort", error_type=type(error).__name__,
                                error_category="cleanup_failed")

        try:
            await store.close_session(
                owner_id=owner_id,
                session_id=session_id,
                device_id=device_id,
            )
        except Exception as error:
            # Session teardown still closes the transport if persistence is unavailable.
            trace.event("voice.session_close_persist_failed", status="failed",
                        failure_stage="close_session", error_type=type(error).__name__,
                        error_category="storage_failed")

        if websocket.application_state == WebSocketState.CONNECTED:
            await _try_send_control(
                websocket,
                VoiceSessionEvent(type="session_closed", session_id=session_id),
            )
            try:
                await websocket.close(code=1000)
            except Exception as error:
                trace.event("voice.websocket_close_failed", status="failed",
                            failure_stage="websocket_close", error_type=type(error).__name__,
                            error_category="transport_failed")
        trace.event(
            "voice.session_closed",
            reason=close_reason,
            status="cancelled" if cancellation is not None else (
                "failed" if session_error_code else "closed"
            ),
            error_code=session_error_code,
        )
        if cancellation is not None:
            trace.end(status="cancelled")
            raise cancellation
        trace.end(status="failed" if session_error_code else "closed", error_code=session_error_code)


async def _relay_device_audio(
    websocket: WebSocket,
    *,
    session_id: UUID,
    provider: VoiceProvider,
    known_client_delegations: set[str],
    trace: Observation | None = None,
) -> None:
    trace = trace or Observation()
    levels = PcmLevelMeter()
    try:
        expected_sequence = 0
        last_turn_end_sequence = -1
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            raw_text = message.get("text")
            try:
                envelope = json.loads(raw_text) if isinstance(raw_text, str) else None
            except json.JSONDecodeError:
                envelope = None
            if isinstance(envelope, dict) and envelope.get("type") == "voice_audio_turn_end":
                try:
                    turn_end = AudioTurnEnd.model_validate(envelope)
                except Exception:
                    raise InvalidVoiceFrame("audio turn-end control was invalid") from None
                if turn_end.session_id != session_id:
                    raise InvalidVoiceFrame("audio turn end belongs to another session")
                if (
                    expected_sequence == 0
                    or turn_end.last_audio_sequence != expected_sequence - 1
                    or turn_end.last_audio_sequence <= last_turn_end_sequence
                ):
                    raise InvalidVoiceFrame("audio turn end does not follow its final frame")
                last_turn_end_sequence = turn_end.last_audio_sequence
                finish_turn = getattr(provider, "finish_turn", None)
                if callable(finish_turn):
                    with trace.activate():
                        await finish_turn(
                            turn_end.turn_id,
                            vad_speech_ms=turn_end.vad_speech_ms,
                        )
                continue
            if isinstance(envelope, dict) and envelope.get("type") == "voice_session_context_append":
                context = _parse_session_context_append(message)
                if context.session_id != session_id:
                    raise InvalidVoiceFrame("session context update is not bound to this session")
                await provider.append_context(
                    context.content,
                    kind=context.kind,
                    delegation_id=None,
                )
                forwarded = trace.child("context.forwarded", kind=context.kind, chars=len(context.content))
                forwarded.io(input=context.content, output="Accepted by voice provider")
                forwarded.end()
                continue
            if isinstance(envelope, dict) and envelope.get("type") == "voice_context_append":
                context = _parse_context_append(message)
                if (
                    context.session_id != session_id
                    or context.delegation_id not in known_client_delegations
                ):
                    raise InvalidVoiceFrame("context update is not bound to a known delegation")
                await provider.append_context(
                    context.content,
                    kind=context.kind,
                    delegation_id=context.delegation_id,
                )
                trace.event("context.forwarded", delegation_id=context.delegation_id,
                            kind=context.kind, chars=len(context.content))
                continue
            metadata = _parse_input_metadata(message)
            if metadata.session_id != session_id:
                raise InvalidVoiceFrame("audio belongs to another session")
            if metadata.sequence != expected_sequence:
                raise InvalidVoiceFrame("audio sequence is not contiguous")
            if (
                metadata.encoding != "pcm_s16le"
                or metadata.sample_rate_hz != 24_000
                or metadata.channels != 1
                or metadata.frame_duration_ms > 100
            ):
                raise InvalidVoiceFrame("audio format is outside the relay profile")
            byte_numerator = (
                metadata.sample_rate_hz
                * metadata.channels
                * 2
                * metadata.frame_duration_ms
            )
            if byte_numerator % 1000:
                raise InvalidVoiceFrame("audio duration does not align to PCM samples")
            expected_bytes = byte_numerator // 1000
            if not 0 < expected_bytes <= MAX_INPUT_FRAME_BYTES:
                raise InvalidVoiceFrame("audio frame exceeds relay limits")

            binary_message = await websocket.receive()
            if binary_message.get("type") == "websocket.disconnect":
                return
            payload = binary_message.get("bytes")
            if not isinstance(payload, bytes) or len(payload) != expected_bytes:
                raise InvalidVoiceFrame("binary audio length does not match metadata")
            await provider.append_audio(payload)
            levels.add(payload)
            if levels.frames % 100 == 0:
                trace.event("audio.forwarded_levels", **levels.snapshot())
            if expected_sequence == 0:
                trace.event("audio.first_input_forwarded")
            expected_sequence += 1

    finally:
        trace.event("audio.forwarded_levels", status="final", **levels.snapshot())


async def _relay_provider_events(
    websocket: WebSocket,
    *,
    session_id: UUID,
    provider: VoiceProvider,
    known_client_delegations: set[str],
    trace: Observation | None = None,
) -> None:
    trace = trace or Observation()
    output_sequence = 0
    response_audio_started = False
    while True:
        event = await provider.receive_event()
        event_type = event.get("type")
        if event_type == "session.error":
            raise LiveProviderError("voice provider could not complete the turn")
        if event_type == "session.processing":
            response_audio_started = False
            await _send_control(
                websocket,
                VoiceSessionEvent(type="assistant_processing", session_id=session_id),
            )
        elif event_type == "session.output_audio.done":
            expectation = event.get("reply_expectation")
            expectation_fields = (
                {"reply_expectation": expectation}
                if expectation in {"none", "open_answer", "confirmation", "closing"}
                else {}
            )
            await _send_control(
                websocket,
                VoiceSessionEvent(
                    type="assistant_response_complete",
                    session_id=session_id,
                    **expectation_fields,
                ),
            )
            response_audio_started = False
        elif event_type == "session.input_turn.ignored":
            await _send_control(
                websocket,
                VoiceSessionEvent(
                    type="assistant_turn_ignored", session_id=session_id
                ),
            )
        elif event_type == "session.input_transcript.ignored":
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                trace.transcript("user", delta)
                trace.event("voice.turn_ignored", chars=len(delta), reason="provider_ignored")
                _LOGGER.info(
                    "voice provider ignored transcript session_id=%s chars=%d",
                    session_id, len(delta),
                )
        elif event_type in {"session.input_transcript.delta", "session.output_transcript.delta"}:
            delta = event.get("delta")
            if not isinstance(delta, str) or not delta:
                trace.event("provider.transcript_invalid", direction="user" if event_type == "session.input_transcript.delta" else "assistant")
                continue
            start_ms, end_ms = event.get("start_ms"), event.get("end_ms")
            trace.transcript("user" if event_type == "session.input_transcript.delta" else "assistant", delta)
            trace.event("provider.transcript", direction="user" if event_type == "session.input_transcript.delta" else "assistant",
                        chars=len(delta), start_ms=start_ms, end_ms=end_ms)
            payload: dict[str, Any] = {
                "type": "transcript_delta",
                "session_id": session_id,
                "direction": "user" if event_type == "session.input_transcript.delta" else "assistant",
                "delta": delta,
            }
            if isinstance(start_ms, int) and not isinstance(start_ms, bool):
                payload["start_ms"] = start_ms
            if isinstance(end_ms, int) and not isinstance(end_ms, bool):
                payload["end_ms"] = end_ms
            await _send_control(websocket, VoiceSessionEvent.model_validate(payload))
        elif event_type == "session.delegation.created":
            delegation = event.get("delegation")
            if not isinstance(delegation, dict):
                continue
            delegation_id = delegation.get("id")
            target = delegation.get("target")
            offset_ms = event.get("offset_ms")
            if (
                not isinstance(delegation_id, str)
                or not delegation_id.strip()
                or not isinstance(offset_ms, int)
                or isinstance(offset_ms, bool)
            ):
                continue
            if target not in {"client", "responses", "server"}:
                continue
            if target == "client":
                trace.event("provider.delegation", delegation_id=delegation_id, offset_ms=offset_ms)
                if (
                    delegation_id not in known_client_delegations
                    and len(known_client_delegations) >= MAX_CLIENT_DELEGATIONS
                ):
                    raise LiveProviderError("session exceeded its delegation limit")
                known_client_delegations.add(delegation_id)
            await _send_control(
                websocket,
                VoiceSessionEvent(
                    type="delegation_created",
                    session_id=session_id,
                    delegation_id=delegation_id,
                    target=target,
                    offset_ms=offset_ms,
                ),
            )
        elif event_type == "session.output_audio.delta":
            encoded = event.get("delta")
            if not isinstance(encoded, str):
                continue
            try:
                pcm = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError):
                raise LiveProviderError("provider audio payload was invalid") from None
            if not pcm or len(pcm) % 2:
                raise LiveProviderError("provider audio payload was not PCM16")
            if not response_audio_started:
                await _send_control(
                    websocket,
                    VoiceSessionEvent(
                        type="assistant_response_started", session_id=session_id
                    ),
                )
                response_audio_started = True
            for offset in range(0, len(pcm), OUTPUT_FRAME_CHUNK_BYTES):
                chunk = pcm[offset : offset + OUTPUT_FRAME_CHUNK_BYTES]
                metadata = AudioOutputFrameMetadata(
                    session_id=session_id,
                    sequence=output_sequence,
                    encoding="pcm_s16le",
                    payload_bytes=len(chunk),
                )
                await websocket.send_json(
                    {
                        "type": "audio_output_metadata",
                        "payload": metadata.model_dump(mode="json"),
                    }
                )
                await websocket.send_bytes(chunk)
                if output_sequence == 0:
                    trace.event("audio.first_output_forwarded")
                output_sequence += 1
        elif event_type == "session.closed":
            return
        elif event_type in {"error", "session.error"}:
            trace.event("provider.error", error_code="DEPENDENCY_UNAVAILABLE")
            raise LiveProviderError("provider reported a session error")
        # Keep provider acknowledgements, usage snapshots, and raw backend events internal.


def _parse_input_metadata(message: dict[str, Any]) -> AudioFrameMetadata:
    if message.get("type") != "websocket.receive" or message.get("text") is None:
        raise InvalidVoiceFrame("audio metadata must precede its binary payload")
    try:
        if len(message["text"].encode("utf-8")) > 4_096:
            raise ValueError
        envelope = json.loads(message["text"])
        if not isinstance(envelope, dict) or set(envelope) != {"type", "payload"}:
            raise ValueError
        if envelope["type"] != "audio_frame_metadata":
            raise ValueError
        return AudioFrameMetadata.model_validate(envelope["payload"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise InvalidVoiceFrame("audio metadata did not match the device contract") from None


def _parse_context_append(message: dict[str, Any]) -> VoiceContextAppend:
    if message.get("type") != "websocket.receive" or message.get("text") is None:
        raise InvalidVoiceFrame("voice context must use a JSON control message")
    try:
        if len(message["text"].encode("utf-8")) > 4_096:
            raise ValueError
        envelope = json.loads(message["text"])
        if not isinstance(envelope, dict) or set(envelope) != {"type", "payload"}:
            raise ValueError
        if envelope["type"] != "voice_context_append":
            raise ValueError
        return VoiceContextAppend.model_validate(envelope["payload"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise InvalidVoiceFrame("voice context did not match the device contract") from None


def _parse_session_context_append(message: dict[str, Any]) -> VoiceSessionContextAppend:
    if message.get("type") != "websocket.receive" or message.get("text") is None:
        raise InvalidVoiceFrame("voice session context must use a JSON control message")
    try:
        if len(message["text"].encode("utf-8")) > 4_096:
            raise ValueError
        envelope = json.loads(message["text"])
        if not isinstance(envelope, dict) or set(envelope) != {"type", "payload"}:
            raise ValueError
        if envelope["type"] != "voice_session_context_append":
            raise ValueError
        return VoiceSessionContextAppend.model_validate(envelope["payload"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise InvalidVoiceFrame("voice session context did not match the device contract") from None


def _remaining_session_seconds(session: SessionResponse) -> float | None:
    if session.max_session_seconds is None:
        return None
    # Compute against current UTC without trusting a client-provided clock.
    elapsed = (datetime.now(timezone.utc) - session.opened_at).total_seconds()
    return max(0.0, session.max_session_seconds - elapsed)


async def _send_control(websocket: WebSocket, event: VoiceSessionEvent) -> None:
    await websocket.send_json(event.model_dump(mode="json", exclude_none=True))


async def _try_send_control(websocket: WebSocket, event: VoiceSessionEvent) -> None:
    if websocket.application_state != WebSocketState.CONNECTED:
        return
    try:
        await _send_control(websocket, event)
    except Exception:
        pass
