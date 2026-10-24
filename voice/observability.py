"""Optional Langfuse session trees with bounded, redacted text diagnostics."""

from __future__ import annotations

import asyncio
import logging
import os
import hashlib
import json
import re
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from uuid import UUID

_LOGGER = logging.getLogger(__name__)
_client: Any = None
_initialized = False
_flush_tasks: set[asyncio.Task[Any]] = set()
_forced_ids: ContextVar[tuple[int, int] | None] = ContextVar("cube_span_ids", default=None)
_FIELDS = frozenset({
    "session_id", "turn_id", "job_id", "delegation_id", "request_trace_id", "component",
    "direction", "chars", "start_ms", "end_ms", "offset_ms", "accepted",
    "fragments", "reason", "error_type", "error_code", "status", "route",
    "kind", "frames", "capture_queued_ms", "capture_dropped",
    "playback_queued_ms", "playback_dropped", "pending", "elapsed_ms",
    "wake_to_session_ms", "output_frames", "input_frames", "tool_name",
    "model", "input_tokens", "output_tokens", "total_tokens", "attempt",
    "wake_score", "wake_threshold", "wake_history_peak", "wake_history_windows",
    "wake_confirmation_windows", "confidence", "idle_timeout_seconds",
    "expectation", "muted", "acknowledgement_confidence",
    "stage", "explanation", "provider", "acknowledgement",
    "pcm_rms", "pcm_peak", "pcm_clipped_samples", "audio_bytes",
    "audio_duration_ms",
    "vad_speech_ms",
})
_current: ContextVar[Observation | None] = ContextVar("cube_observation", default=None)

_LABELS = {
    "voice.device": "Watson conversation",
    "voice.bridge": "Understand request and deliver answer",
    "voice.gateway": "Live voice connection",
    "job.lifecycle": "Background request",
    "job.worker": "Execute request",
    "job.graph": "Choose route and generate answer",
    "model.invoke": "Model response",
    "tool.invoke": "Run tool",
    "tool.fast_time": "Get current time",
    "clock.lookup": "Get current time",
    "clock.result_sent": "Clock answer delivered to Live",
    "job.submit": "Submit background request",
    "provider.start": "Connect to voice model",
    "provider.ready": "Voice model ready",
    "transcript.received": "Recognized speech fragment",
    "transcript.filtered": "Check request boundary",
    "provider.transcript": "Voice model speech fragment",
    "input.endpoint": "Local speech endpoint",
    "input.speech_candidate": "Local speech detection",
    "audio.input_muted": "Microphone muted during response",
    "voice.transcription_result": "Transcription result",
    "voice.route_classified": "Jev intent decision",
    "voice.route_selected": "Jev routing decision",
    "voice.reply_expectation": "Follow-up listening window",
    "voice.reply_received": "Answer received from Muse",
    "voice.reply_ready": "Answer ready for the speaker",
    "voice.session_closed": "Conversation ended",
    "voice.turn_ignored": "Speech ignored by voice provider",
    "voice.turn_audio": "Audio submitted for transcription",
    "voice.turn_recovery": "Ask for a repeat after empty transcription",
    "delegation.wait": "Wait for complete request",
    "context.sent": "Deliver background result to Live",
    "session_context.sent": "Deliver local result to Live",
    "context.forwarded": "Forward result to voice model",
    "idle.pending": "Wait for answer before closing",
    "playback.first_audible_frame": "First audible reply on speaker",
    "audio.queues": "Microphone and speaker queue health",
    "wake.detected": "Hey Watson detected",
    "wake.session_created": "Wake opened conversation",
    "audio.first_input_forwarded": "First microphone audio sent",
    "audio.first_output_forwarded": "First model audio received",
    "job.result_persisted": "Save completed answer",
    "diagnostics": "Diagnostics (audio, timing, transcript fragments)",
    "audio.input_levels": "Microphone levels captured on Pi",
    "audio.forwarded_levels": "Microphone levels sent to voice model",
    "input.initial_window": "Listen for first question after connection is ready",
    "speech.recognize": "Turn speech into text",
    "voice.route": "Choose how Watson handles the request",
    "muse.wait": "Wait for Muse to answer",
    "speech.generate": "Turn Watson's reply into speech",
    "voice.turn.failed": "Voice turn failed",
    "jev.fallback": "Use Luna decisions after Jev failed",
    "speech.recognize": "Turn speech into text",
    "voice.route": "Choose how Watson handles the request",
    "muse.wait": "Wait for Muse to answer",
    "speech.generate": "Turn Watson's reply into speech",
    "voice.turn.failed": "Voice turn failed",
}
_EXPLANATIONS = {
    "voice.device": "Complete wake-to-close conversation. Input is recognized user speech; output is Watson speech.",
    "voice.bridge": "Assembles the spoken request, runs local lookups or submits background work, then delivers facts to Live.",
    "voice.gateway": "Maintains the live microphone/model/speaker connection and forwards transcripts and results.",
    "job.lifecycle": "One claimed background request attempt, including execution and saving its result.",
    "job.graph": "Routes the request and runs model/tool calls needed to answer it.",
    "model.invoke": "One model call. Input shows messages; output shows generated text and tool calls.",
    "tool.invoke": "One tool call. Input shows arguments; output shows returned data.",
    "clock.lookup": "Read-only current-time lookup using Pi clock and timezone data; no web or model needed.",
    "idle.pending": "True holds the session for an answer; false resumes the normal inactivity window.",
    "input.speech_candidate": "Local speech candidate started or ended; speech text is not stored here.",
    "audio.input_muted": "Microphone frames replaced with silence while the assistant processes or speaks.",
    "voice.transcription_result": "Speech recognition result using safe length and timing only.",
    "voice.route_selected": "Jev's effective route and acknowledgement choice; fallback is recorded separately.",
    "voice.reply_expectation": "Whether the reply invites another turn and the resulting listening window.",
    "voice.route_classified": "Jev's route and acknowledgement choices with confidence, without speech text.",
    "voice.reply_received": "Muse reply text arrived; safe length and elapsed time only.",
    "voice.reply_ready": "Speech generation finished and the follow-up window was set.",
    "voice.session_closed": "Session close reason and inactivity timing.",
    "voice.turn_ignored": "A recognized turn was intentionally ignored; safe length and reason only.",
    "voice.turn_audio": "Per-turn PCM length, VAD speech time and signal levels; audio itself is not retained.",
    "voice.turn_recovery": "One repeat prompt after an invited reply contained sustained VAD speech but no recognized words.",
    "speech.recognize": "ElevenLabs transcription duration and safe failure type.",
    "voice.route": "Routing decision duration, provider and selected route.",
    "muse.wait": "Time spent waiting for Muse text before speech playback.",
    "speech.generate": "ElevenLabs speech generation duration and safe failure type.",
    "voice.turn.failed": "Failed voice stage and sanitized error type; no provider payloads.",
    "jev.fallback": "Jev failed, so OpenAI Decisions handled this classification.",
    "speech.recognize": "ElevenLabs transcription duration and safe failure type.",
    "voice.route": "Jev routing duration; fallback use is recorded separately.",
    "muse.wait": "Time spent waiting for Muse text before generating speech.",
    "speech.generate": "ElevenLabs speech generation duration and safe failure type.",
    "voice.turn.failed": "Failed voice stage and sanitized error type; no provider payloads.",
    "diagnostics": "Low-level measurements; expand when investigating timing, missing speech or audio loss.",
}
_DIAGNOSTICS = frozenset({
    "transcript.received", "transcript.filtered", "provider.transcript",
    "provider.transcript_invalid", "input.endpoint", "input.speech_candidate",
    "audio.input_muted", "voice.transcription_result", "voice.route_selected",
    "voice.route_classified", "voice.reply_received", "voice.reply_ready",
    "voice.turn_ignored", "voice.turn.failed", "voice.reply_expectation",
    "voice.turn_audio", "voice.turn_recovery",
    "voice.session_closed", "audio.queues",
    "audio.first_input_forwarded", "audio.first_output_forwarded",
    "audio.input_levels", "audio.forwarded_levels",
})


def _name(name: str) -> str:
    return "cube. " + _LABELS.get(name, name.replace(".", " / ").replace("_", " "))


def _root_id(session_id: UUID) -> int:
    return int.from_bytes(hashlib.sha256(f"watson-session:{session_id}".encode()).digest()[:8], "big") or 1


def _content(value: Any, depth: int = 0) -> Any:
    """Bound exported values; avoid object reprs and redact credential fields."""
    if depth > 5:
        return "[depth limit]"
    if isinstance(value, str):
        text = value[:4000]
        for key, secret in os.environ.items():
            if any(part in key.upper() for part in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and len(secret) >= 8:
                text = text.replace(secret, "[redacted]")
        text = re.sub(r"(?i)(bearer\s+)[\w.\-/+=]+", r"\1[redacted]", text)
        text = re.sub(r"\b(?:sk|pk)-[A-Za-z0-9_-]{12,}", "[redacted]", text)
        text = re.sub(r"(?i)((?:api[_ -]?key|secret|password|token)\s*[:=]\s*)[^\s,;]+", r"\1[redacted]", text)
        return text + (" [truncated]" if len(value) > 4000 else "")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): "[redacted]" if any(x in str(k).lower() for x in ("key", "token", "secret", "password", "authorization"))
                else _content(v, depth + 1) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        return [_content(v, depth + 1) for v in value[:20]]
    if hasattr(value, "content"):
        return {"role": getattr(value, "type", "message"), "content": _content(value.content, depth + 1),
                "tool_calls": _content(getattr(value, "tool_calls", []), depth + 1)}
    return f"[{type(value).__name__}]"


def _bounded_content(value: Any) -> Any:
    result = _content(value)
    encoded = json.dumps(result, ensure_ascii=False)
    return result if len(encoded) <= 16000 else {"preview": encoded[:16000], "truncated": True}


def _metadata(values: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {
        key: str(value) if isinstance(value, UUID) else value
        for key, value in values.items()
        if key in _FIELDS and isinstance(value, (str, int, float, bool, UUID))
        and (not isinstance(value, str) or len(value) <= 160)
    }


def _get_client() -> Any:
    global _client, _initialized
    if _initialized:
        return _client
    _initialized = True
    if os.environ.get("CUBE_LANGFUSE_ENABLED", "false").lower() != "true":
        return None
    if not all(os.environ.get(key) for key in (
        "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL",
    )):
        _LOGGER.warning("Langfuse disabled: project keys/base URL missing")
        return None
    try:
        from langfuse import Langfuse
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

        class SessionIdGenerator(RandomIdGenerator):
            def generate_trace_id(self) -> int:
                ids = _forced_ids.get()
                return ids[0] if ids else super().generate_trace_id()

            def generate_span_id(self) -> int:
                ids = _forced_ids.get()
                return ids[1] if ids else super().generate_span_id()

        _client = Langfuse(
            debug=False, timeout=3, flush_at=64, flush_interval=2,
            should_export_span=lambda span: span.name.startswith("cube."),
            tracer_provider=TracerProvider(id_generator=SessionIdGenerator()),
        )
    except Exception as error:
        _LOGGER.warning("Langfuse initialization failed (%s)", type(error).__name__)
    return _client


async def _flush_after_session(client: Any, component: str, session_id: UUID) -> None:
    try:
        await asyncio.to_thread(client.flush)
    except Exception as error:
        _LOGGER.warning(
            "Langfuse session flush failed component=%s session_id=%s error_type=%s",
            component, session_id, type(error).__name__,
        )
    else:
        _LOGGER.info(
            "Langfuse session flush completed component=%s session_id=%s",
            component, session_id,
        )


def _schedule_session_flush(component: str, session_id: UUID) -> None:
    if _client is None:
        return
    try:
        task = asyncio.get_running_loop().create_task(
            _flush_after_session(_client, component, session_id),
            name=f"cube-langfuse-flush-{session_id}",
        )
    except RuntimeError:
        return
    _flush_tasks.add(task)
    task.add_done_callback(_flush_tasks.discard)


class Observation:
    """A manual span; SDK failures never escape into the conversation."""

    def __init__(self, raw: Any = None, *, session_id: UUID | None = None, metadata: dict[str, Any] | None = None) -> None:
        self._raw = raw
        self.session_id = session_id
        self._metadata = _metadata({"session_id": session_id, **(metadata or {})})
        self._transcript: list[dict[str, str]] = []
        self._diagnostics: Observation | None = None

    def update(self, **metadata: Any) -> None:
        if self._raw is None:
            return
        try:
            self._metadata.update(_metadata(metadata))
            level = "ERROR" if self._metadata.get("status") == "failed" else (
                "WARNING" if self._metadata.get("status") in {"expired", "abandoned"} else "DEFAULT")
            self._raw.update(metadata=self._metadata, level=level)
        except Exception as error:
            _LOGGER.warning(
                "Langfuse observation update failed stage=%s session_id=%s error_type=%s",
                self._metadata.get("stage", "unknown"), self.session_id,
                type(error).__name__,
            )

    def io(self, *, input: Any = None, output: Any = None) -> None:
        if self._raw is None or os.environ.get("CUBE_LANGFUSE_CAPTURE_CONTENT", "false").lower() != "true":
            return
        try:
            data = {}
            if input is not None:
                data["input"] = _bounded_content(input)
            if output is not None:
                data["output"] = _bounded_content(output)
            self._raw.update(**data)
            if self._metadata.get("component") == "voice.device":
                from opentelemetry.trace import use_span
                with use_span(self._raw._otel_span):
                    _get_client().set_current_trace_io(**data)
        except Exception as error:
            _LOGGER.warning(
                "Langfuse content update failed stage=%s session_id=%s error_type=%s",
                self._metadata.get("stage", "unknown"), self.session_id,
                type(error).__name__,
            )

    def transcript(self, direction: str, delta: str) -> None:
        if os.environ.get("CUBE_LANGFUSE_CAPTURE_CONTENT", "false").lower() != "true":
            return
        role = "user" if direction == "user" else "assistant"
        if self._transcript and self._transcript[-1]["role"] == role:
            self._transcript[-1]["content"] = (self._transcript[-1]["content"] + delta)[-4000:]
        else:
            self._transcript.append({"role": role, "content": delta[:4000]})
            self._transcript = self._transcript[-20:]
        self.io(input=[x for x in self._transcript if x["role"] == "user"],
                output=[x for x in self._transcript if x["role"] == "assistant"])

    @contextmanager
    def activate(self) -> Iterator[Observation]:
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)

    def usage(self, counts: dict[str, int]) -> None:
        if self._raw is None:
            return
        details = {key.removesuffix("_tokens"): value for key, value in counts.items()
                   if key in {"input_tokens", "output_tokens", "total_tokens"}
                   and isinstance(value, int) and not isinstance(value, bool) and value >= 0}
        try:
            self._raw.update(usage_details=details)
        except Exception:
            pass

    def child(self, name: str, *, as_type: str = "span", **metadata: Any) -> Observation:
        if self._raw is None:
            return Observation(session_id=self.session_id)
        if name in _DIAGNOSTICS:
            if self._diagnostics is None:
                self._diagnostics = self.child("diagnostics")
            # Diagnostics children must not route themselves into another group.
            return self._diagnostics._child(name, as_type=as_type, **metadata)
        return self._child(name, as_type=as_type, **metadata)

    def _child(self, name: str, *, as_type: str = "span", **metadata: Any) -> Observation:
        metadata = {"stage": name, "explanation": _EXPLANATIONS.get(name, _LABELS.get(name, name)), **metadata}
        try:
            from langfuse import propagate_attributes

            with propagate_attributes(session_id=str(self.session_id)):
                raw = self._raw.start_observation(
                    name=_name(name),
                    as_type=as_type,
                    **({"model": metadata["model"]} if as_type == "generation" and "model" in metadata else {}),
                    metadata=_metadata({"session_id": self.session_id, **metadata}),
                )
            return Observation(raw, session_id=self.session_id, metadata=metadata)
        except Exception as error:
            _LOGGER.warning(
                "Langfuse child observation creation failed stage=%s session_id=%s error_type=%s",
                name, self.session_id, type(error).__name__,
            )
            return Observation(session_id=self.session_id)

    def event(self, name: str, **metadata: Any) -> None:
        self.child(name, **metadata).end()

    def end(self, **metadata: Any) -> None:
        if self._diagnostics is not None:
            self._diagnostics.end()
            self._diagnostics = None
        if metadata:
            self.update(**metadata)
        raw, self._raw = self._raw, None
        if raw is None:
            return
        try:
            raw.end()
        except Exception as error:
            _LOGGER.warning(
                "Langfuse observation end failed component=%s session_id=%s error_type=%s",
                self._metadata.get("component", "child"), self.session_id,
                type(error).__name__,
            )
        component = self._metadata.get("component")
        if component in {"voice.device", "voice.gateway"} and self.session_id:
            _schedule_session_flush(component, self.session_id)

    @contextmanager
    def operation(self, name: str, **metadata: Any) -> Iterator[Observation]:
        child = self.child(name, **metadata)
        token = _current.set(child)
        try:
            yield child
        except BaseException as error:
            child.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            _current.reset(token)
            child.end()


def start_session(session_id: UUID, component: str, **metadata: Any) -> Observation:
    client = _get_client()
    if client is None:
        return Observation(session_id=session_id)
    try:
        metadata = {"stage": component, "explanation": _EXPLANATIONS.get(component, _LABELS.get(component, component)), **metadata}
        from langfuse import propagate_attributes
        from opentelemetry import context, trace
        parent = _current.get()
        if parent is not None and parent.session_id == session_id and parent._raw is not None:
            result = parent.child(component, component=component, **metadata)
            return result
        # One actual root on the device. Remote components attach to its stable
        # ID through OTel context rather than SDK trace_context, which marks
        # each component AS_ROOT and makes the UI display separate roots.
        root = component == "voice.device"
        ctx = context.Context()
        if not root:
            remote = trace.NonRecordingSpan(trace.SpanContext(
                trace_id=session_id.int, span_id=_root_id(session_id),
                is_remote=True, trace_flags=trace.TraceFlags(1),
            ))
            ctx = trace.set_span_in_context(remote, ctx)
        ctx_token = context.attach(ctx)
        ids_token = _forced_ids.set((session_id.int, _root_id(session_id)) if root else None)
        try:
            with propagate_attributes(session_id=str(session_id), trace_name="Watson conversation"):
                raw = client.start_observation(
                    name=_name(component),
                    metadata=_metadata({"session_id": session_id, "component": component, **metadata}),
                )
        finally:
            _forced_ids.reset(ids_token)
            context.detach(ctx_token)
        if component in {"voice.device", "voice.gateway"}:
            _LOGGER.info(
                "Langfuse session observation started component=%s session_id=%s",
                component, session_id,
            )
        return Observation(raw, session_id=session_id, metadata={"component": component, **metadata})
    except Exception as error:
        _LOGGER.warning(
            "Langfuse session observation creation failed component=%s session_id=%s error_type=%s",
            component, session_id, type(error).__name__,
        )
        return Observation(session_id=session_id)


def current_observation() -> Observation:
    return _current.get() or Observation()


def initialize(component: str) -> None:
    """Record real process startup before accepting audio or jobs."""
    client = _get_client()
    if client is None:
        _LOGGER.info("Langfuse unavailable for component=%s; tracing disabled", component)
        return
    try:
        span = client.start_observation(
            name="cube.observability.startup",
            metadata={"component": component, "status": "configured"},
        )
        span.end()
        client.flush()
        _LOGGER.info("Langfuse startup observation flushed component=%s", component)
    except Exception as error:
        _LOGGER.warning("Langfuse startup observation failed (%s)", type(error).__name__)


async def observed_await(name: str, operation: Awaitable[Any], *, input: Any = None, **metadata: Any) -> Any:
    with current_observation().operation(
        name, as_type="generation" if name == "model.invoke" else "tool", **metadata,
    ) as span:
        span.io(input=input)
        result = await operation
        span.io(output=result)
        usage = getattr(result, "usage_metadata", None)
        if isinstance(usage, dict):
            counts = {key: usage[key] for key in ("input_tokens", "output_tokens", "total_tokens")
                      if isinstance(usage.get(key), int)}
            span.update(**counts)
            span.usage(counts)
        span.update(status="succeeded")
        return result


async def shutdown() -> None:
    """Drain exports off the audio/event loop, only during process shutdown."""
    if _client is None:
        return
    try:
        if _flush_tasks:
            await asyncio.gather(*tuple(_flush_tasks), return_exceptions=True)
        await asyncio.to_thread(_client.shutdown)
    except Exception as error:
        _LOGGER.warning("Langfuse shutdown failed error_type=%s", type(error).__name__)
