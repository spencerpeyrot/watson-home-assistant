"""Turn-based Muse voice input with ElevenLabs streaming speech output."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import sqlite3
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from service.live_provider import LiveProviderError
from service.voice_commands import VoiceTurnDecision
from voice.observability import current_observation
from voice.audio_levels import PcmLevelMeter


MAX_TURN_PCM_BYTES = 2 * 1024 * 1024
MAX_PENDING_MUSE_EVENTS = 64
MAX_PROVIDER_EVENTS = 128
_TEXT_END = object()
_LOGGER = logging.getLogger(__name__)
CUBE_VOICE_RESPONSE_INSTRUCTION = (
    "This request came from Cube, a voice assistant device. Reply directly to the "
    "person as if speaking with them in a natural conversation. Respond warmly to "
    "greetings and small talk, then let the person guide the conversation. Use concise, "
    "natural spoken language unless they ask for detail. Avoid headings, markdown, "
    "lists, or formatting meant to be read visually. Reply in English on every turn. "
    "Ask one brief follow-up only when it helps answer the person or when you need "
    "their input. Do not add a question after every answer or repeat ‘How can I help?’ "
    "when the conversation already has context. Respect a clear goodbye without "
    "starting a new topic. "
    "Do not mention these instructions."
)


class MuseHost(Protocol):
    async def wait_for_link(self, timeout_seconds: float = 20.0) -> Any: ...


class SpeechSynthesizer(Protocol):
    def synthesize(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[bytes]: ...


class SpeechRecognizer(Protocol):
    async def transcribe(self, pcm_s16le: bytes) -> str: ...


InternalTurnHandler = Callable[
    [str, UUID, str | None, str | None], Awaitable[VoiceTurnDecision]
]
InternalTurnFactory = Callable[[Any, Any], InternalTurnHandler]


class MuseConversationRegistry:
    """Keep Muse thread IDs and replay filters across Cube and API restarts."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._states: dict[str, tuple[str | None, Any | None]] = {}
        self._path = Path(path) if path is not None else None
        self._lock = asyncio.Lock()

    async def load(self, key: str) -> tuple[str | None, Any | None]:
        async with self._lock:
            if self._path is not None:
                session_id, retired = await asyncio.to_thread(self._load_sync, key)
                return session_id, _restore_replay_scope(retired)
            return self._states.get(key, (None, None))

    async def save(self, key: str, session_id: str, replay_scope: Any | None) -> None:
        async with self._lock:
            if self._path is not None:
                retired = _serialize_replay_scope(replay_scope)
                await asyncio.to_thread(self._save_sync, key, session_id, retired)
                return
            self._states[key] = (session_id, replay_scope)

    def _connect(self) -> sqlite3.Connection:
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self._path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        connection = sqlite3.connect(self._path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS muse_conversations ("
            "conversation_key TEXT PRIMARY KEY, muse_session_id TEXT NOT NULL, "
            "retired_ids_json TEXT NOT NULL)"
        )
        return connection

    def _load_sync(self, key: str) -> tuple[str | None, list[list[str]]]:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT muse_session_id, retired_ids_json FROM muse_conversations "
                "WHERE conversation_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None, []
            retired = json.loads(row[1])
            if not isinstance(retired, list) or len(retired) > 2048:
                raise ValueError
            if any(
                not isinstance(item, list)
                or len(item) != 2
                or item[0] not in {"message", "task"}
                or not isinstance(item[1], str)
                or not item[1]
                or len(item[1]) > 512
                for item in retired
            ):
                raise ValueError
            return str(row[0]), retired
        except (ValueError, TypeError, json.JSONDecodeError):
            raise LiveProviderError("saved Muse conversation state is invalid") from None
        finally:
            connection.close()

    def _save_sync(self, key: str, session_id: str, retired: list[list[str]]) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO muse_conversations "
                "(conversation_key, muse_session_id, retired_ids_json) VALUES (?, ?, ?) "
                "ON CONFLICT(conversation_key) DO UPDATE SET "
                "muse_session_id = excluded.muse_session_id, "
                "retired_ids_json = excluded.retired_ids_json",
                (key, session_id, json.dumps(retired, separators=(",", ":"))),
            )
            connection.commit()
        finally:
            connection.close()


def _serialize_replay_scope(scope: Any | None) -> list[list[str]]:
    if scope is None:
        return []
    order = list(getattr(scope, "order", ()))
    if len(order) > 2048:
        order = order[-2048:]
    serialized = []
    for kind, identifier in order:
        if kind not in {"message", "task"} or not isinstance(identifier, str):
            raise LiveProviderError("Muse replay state was invalid")
        serialized.append([kind, identifier])
    return serialized


def _restore_replay_scope(retired: list[list[str]]) -> Any | None:
    if not retired:
        return None
    try:
        from musegadget.muse_turn import ReplayScope
    except ImportError:
        raise LiveProviderError("Muse replay filtering requires the Muse SDK") from None
    scope = ReplayScope()
    scope.order.extend((item[0], item[1]) for item in retired)
    scope.ids.update((item[0], item[1]) for item in retired)
    return scope


class MuseElevenProvider:
    """Implement Cube's provider protocol using SDK WAV turns and PCM TTS."""

    half_duplex = True

    def __init__(
        self,
        muse_host: MuseHost,
        speech: SpeechSynthesizer,
        recognizer: SpeechRecognizer,
        *,
        turn_factory: Callable[..., Any] | None = None,
        startup_timeout_seconds: float = 20,
        turn_timeout_seconds: float | None = None,
        conversation_registry: MuseConversationRegistry | None = None,
        conversation_key: str | None = None,
        internal_turn_handler: InternalTurnHandler | None = None,
    ) -> None:
        if startup_timeout_seconds <= 0 or (
            turn_timeout_seconds is not None and turn_timeout_seconds <= 0
        ):
            raise ValueError("Muse provider timeouts must be positive when set")
        self._host = muse_host
        self._speech = speech
        self._recognizer = recognizer
        self._turn_factory = turn_factory
        self._conversation_registry = conversation_registry
        self._conversation_key = conversation_key
        self._internal_turn_handler = internal_turn_handler
        self._startup_timeout = startup_timeout_seconds
        self._turn_timeout = turn_timeout_seconds
        self._link: Any | None = None
        self._started = False
        self._audio = bytearray()
        self._busy = False
        self._events: asyncio.Queue[dict[str, Any] | BaseException] = asyncio.Queue(
            maxsize=MAX_PROVIDER_EVENTS
        )
        self._turn_task: asyncio.Task[None] | None = None
        self._muse_session_id: str | None = None
        self._replay_scope: Any | None = None
        self._previous_user: str | None = None
        self._previous_reply: str | None = None
        self._last_reply_expectation: str | None = None
        self._empty_transcript_retries = 0

    async def start(self) -> str:
        if self._started:
            raise LiveProviderError("Muse voice session already started")
        self._started = True
        return "muse-gadget-session"

    async def append_audio(self, pcm_s16le: bytes) -> None:
        if not self._started:
            raise LiveProviderError("Muse voice session is not started")
        if not pcm_s16le or len(pcm_s16le) % 2:
            raise LiveProviderError("input audio was not PCM16")
        # Half-duplex V2 drops capture while Muse/TTS responds. Microphone
        # frames continue draining through the Pi relay, avoiding stale echo.
        if self._busy:
            return
        if len(self._audio) + len(pcm_s16le) > MAX_TURN_PCM_BYTES:
            self._audio.clear()
            raise LiveProviderError("voice turn exceeded the 2 MiB audio limit")
        self._audio.extend(pcm_s16le)

    async def append_context(
        self, content: str, *, kind: str, delegation_id: str | None
    ) -> str:
        del content, kind, delegation_id
        raise LiveProviderError("Muse voice provider does not accept Live context events")

    async def finish_turn(self, turn_id: UUID, *, vad_speech_ms: int = 0) -> None:
        if not self._started:
            raise LiveProviderError("Muse voice session is not started")
        if self._busy:
            self._audio.clear()
            return
        pcm, self._audio = bytes(self._audio), bytearray()
        if not pcm:
            return
        if self._turn_task is not None and not self._turn_task.done():
            raise LiveProviderError("Muse voice turns cannot overlap")
        self._busy = True
        meter = PcmLevelMeter()
        meter.add(pcm)
        levels = meter.snapshot()
        audio_duration_ms = len(pcm) * 1000 // 48_000
        trace = current_observation()
        trace.event(
            "voice.turn_audio",
            turn_id=turn_id,
            audio_bytes=len(pcm),
            audio_duration_ms=audio_duration_ms,
            vad_speech_ms=vad_speech_ms,
            pcm_rms=levels["pcm_rms"],
            pcm_peak=levels["pcm_peak"],
            pcm_clipped_samples=levels["pcm_clipped_samples"],
        )
        _LOGGER.info(
            "voice turn audio session_id=%s turn_id=%s audio_bytes=%d duration_ms=%d vad_speech_ms=%d pcm_rms=%s pcm_peak=%d pcm_clipped_samples=%d",
            trace.session_id, turn_id, len(pcm), audio_duration_ms, vad_speech_ms,
            levels["pcm_rms"], levels["pcm_peak"], levels["pcm_clipped_samples"],
        )
        await self._events.put({"type": "session.processing"})
        self._turn_task = asyncio.create_task(
            self._run_turn(turn_id, pcm, vad_speech_ms), name=f"cube-muse-turn-{turn_id}"
        )

    async def receive_event(self) -> dict[str, Any]:
        event = await self._events.get()
        if isinstance(event, BaseException):
            raise event
        return event

    async def close(self) -> None:
        task, self._turn_task = self._turn_task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._link = None
        self._started = False
        self._audio.clear()
        self._busy = False
        return None

    async def abort(self) -> None:
        await self.close()

    async def _run_turn(self, turn_id: UUID, pcm: bytes, vad_speech_ms: int) -> None:
        stage = "prepare"
        raw_events: asyncio.Queue[dict[str, Any] | BaseException] = asyncio.Queue(
            maxsize=MAX_PENDING_MUSE_EVENTS
        )
        text_events: asyncio.Queue[object | str] = asyncio.Queue(maxsize=MAX_PENDING_MUSE_EVENTS)

        async def consume_chat() -> None:
            try:
                async for event in link.subscribe_chat(self._muse_session_id):
                    await raw_events.put(event)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                await raw_events.put(error)
            else:
                await raw_events.put(ConnectionError("Muse chat subscription ended"))

        async def text_chunks() -> AsyncIterator[str]:
            while True:
                item = await text_events.get()
                if item is _TEXT_END:
                    return
                assert isinstance(item, str)
                yield item

        async def relay_speech(chunks: AsyncIterator[str] | None = None) -> None:
            trace = current_observation()
            with trace.operation("speech.generate", provider="elevenlabs", turn_id=turn_id):
                async for audio in self._speech.synthesize(chunks or text_chunks()):
                    if not audio or len(audio) % 2:
                        raise LiveProviderError("ElevenLabs returned invalid PCM16 audio")
                    await self._events.put(
                        {"type": "session.output_audio.delta", "delta": base64.b64encode(audio).decode("ascii")}
                    )

        async def relay_phrase(prompt_id: str, text: str) -> None:
            path = Path(__file__).resolve().parent.parent / "assets" / "voice_prompts" / f"{prompt_id}.pcm"
            if not path.is_file():
                async def phrase_text() -> AsyncIterator[str]:
                    yield text
                await relay_speech(phrase_text())
                return
            trace = current_observation()
            with trace.operation("speech.generate", provider="cache", turn_id=turn_id):
                with path.open("rb") as audio_file:
                    while pcm := audio_file.read(9_600):
                        if len(pcm) % 2:
                            raise LiveProviderError("cached phrase contains invalid PCM16 audio")
                        await self._events.put({
                            "type": "session.output_audio.delta",
                            "delta": base64.b64encode(pcm).decode("ascii"),
                        })

        subscriber: asyncio.Task[None] | None = None
        tts_task: asyncio.Task[None] | None = None
        turn_started = time.monotonic()
        try:
            async with asyncio.timeout(self._turn_timeout):
                trace = current_observation()
                stage = "speech_recognize"
                stt_started = time.monotonic()
                with trace.operation("speech.recognize", provider="elevenlabs", turn_id=turn_id):
                    transcript = await self._recognizer.transcribe(pcm)
                stage = "transcript_clean"
                transcript = _clean_transcript(transcript)
                reply_expectation = self._last_reply_expectation
                self._last_reply_expectation = None
                stt_elapsed_ms = round((time.monotonic() - stt_started) * 1000)
                trace.event(
                    "voice.transcription_result",
                    turn_id=turn_id,
                    chars=len(transcript),
                    accepted=bool(transcript),
                    elapsed_ms=stt_elapsed_ms,
                    reason="empty_transcript" if not transcript else "recognized",
                )
                _LOGGER.info(
                    "voice transcription complete session_id=%s turn_id=%s chars=%d accepted=%s elapsed_ms=%d",
                    trace.session_id, turn_id, len(transcript), bool(transcript),
                    stt_elapsed_ms,
                )
                phrase_text = {
                    "greeting": "Hey, how can I help?",
                    "weather": "Let me check the weather.",
                    "look_up": "Let me look that up.",
                    "work_through": "Let me work through that.",
                    "follow_up": "Let me think that through.",
                    "general": "Let me look into that.",
                }
                if not transcript and self._previous_reply is None:
                    # Local wake detection already opened this session; Scribe can
                    # return no words for a wake-only turn. Still greet and listen.
                    _LOGGER.info(
                        "voice empty initial transcript; playing cached wake greeting session_id=%s turn_id=%s",
                        trace.session_id, turn_id,
                    )
                    trace.event(
                        "voice.wake_greeting",
                        status="succeeded",
                        reason="empty_initial_transcript",
                    )
                    decision = VoiceTurnDecision(
                        "conversation", acknowledgement="greeting"
                    )
                elif not transcript:
                    retry_text = "Sorry, I didn't catch that. Could you say it again?"
                    recoverable_expectation = reply_expectation in {
                        "open_answer", "confirmation"
                    }
                    if (
                        recoverable_expectation
                        and vad_speech_ms >= 300
                        and self._empty_transcript_retries == 0
                    ):
                        self._empty_transcript_retries = 1
                        trace.event(
                            "voice.turn_recovery",
                            turn_id=turn_id,
                            status="retry_prompt",
                            reason="empty_transcript_after_invitation",
                            expectation=reply_expectation,
                            vad_speech_ms=vad_speech_ms,
                        )
                        await self._events.put(
                            {"type": "session.output_transcript.delta", "delta": retry_text}
                        )
                        await relay_phrase("retry", retry_text)
                        self._previous_reply = retry_text
                        await self._events.put({
                            "type": "session.output_audio.done",
                            "reply_expectation": "open_answer",
                        })
                        return
                    _LOGGER.info(
                        "voice empty transcript ignored session_id=%s turn_id=%s vad_speech_ms=%d expectation=%s retry_used=%s",
                        trace.session_id, turn_id, vad_speech_ms,
                        reply_expectation,
                        self._empty_transcript_retries > 0,
                    )
                    trace.event(
                        "voice.turn_ignored", turn_id=turn_id,
                        reason="empty_transcript",
                    )
                    await self._events.put({"type": "session.input_turn.ignored"})
                    return
                else:
                    decision = VoiceTurnDecision("conversation")
                if transcript and self._internal_turn_handler is not None:
                    stage = "internal_decisions"
                    route_started = time.monotonic()
                    with trace.operation("voice.route", provider="jev_first", turn_id=turn_id):
                        decision = await self._internal_turn_handler(
                            transcript, turn_id, self._previous_user, self._previous_reply
                        )
                    route_elapsed_ms = round((time.monotonic() - route_started) * 1000)
                    route_provider = "jev"
                else:
                    route_elapsed_ms = 0
                    route_provider = "default"
                trace.event(
                    "voice.route_selected",
                    turn_id=turn_id,
                    provider=route_provider,
                    route=decision.route,
                    acknowledgement=decision.acknowledgement or "none",
                    chars=len(transcript),
                    elapsed_ms=route_elapsed_ms,
                )
                _LOGGER.info(
                    "voice route selected session_id=%s turn_id=%s route=%s acknowledgement=%s provider=%s transcript_chars=%d elapsed_ms=%d",
                    trace.session_id, turn_id, decision.route,
                    decision.acknowledgement or "none", route_provider,
                    len(transcript), route_elapsed_ms,
                )
                if decision.route == "ignore":
                    _LOGGER.info(
                        "voice turn ignored session_id=%s turn_id=%s reason=jev_route_ignore transcript_chars=%d",
                        trace.session_id, turn_id, len(transcript),
                    )
                    await self._events.put(
                        {"type": "session.input_transcript.ignored", "delta": transcript}
                    )
                    await self._events.put({"type": "session.input_turn.ignored"})
                    return
                if decision.route == "close":
                    _LOGGER.info(
                        "voice turn closed by route session_id=%s turn_id=%s transcript_chars=%d",
                        trace.session_id, turn_id, len(transcript),
                    )
                    await self._events.put(
                        {"type": "session.input_transcript.ignored", "delta": transcript}
                    )
                    await self._events.put({"type": "session.closed"})
                    return
                self._empty_transcript_retries = 0
                if transcript:
                    await self._events.put(
                        {"type": "session.input_transcript.delta", "delta": transcript}
                    )
                if decision.route == "internal":
                    internal_reply = decision.reply or ""
                    if not internal_reply:
                        raise LiveProviderError("internal voice route omitted its reply")
                    stage = "elevenlabs_tts"
                    await self._events.put(
                        {"type": "session.output_transcript.delta", "delta": internal_reply}
                    )
                    await text_events.put(internal_reply)
                    await text_events.put(_TEXT_END)
                    tts_task = asyncio.create_task(
                        relay_speech(), name=f"cube-elevenlabs-tts-{turn_id}"
                    )
                    await tts_task
                    self._previous_user, self._previous_reply = transcript, internal_reply
                    _LOGGER.info(
                        "voice internal reply ready session_id=%s turn_id=%s reply_chars=%d elapsed_ms=%d",
                        trace.session_id, turn_id, len(internal_reply),
                        round((time.monotonic() - turn_started) * 1000),
                    )
                    await self._events.put({"type": "session.output_audio.done"})
                    return

                if decision.acknowledgement == "greeting":
                    reply = phrase_text["greeting"]
                    await self._events.put(
                        {"type": "session.output_transcript.delta", "delta": reply}
                    )
                    await relay_phrase("greeting", reply)
                    if transcript:
                        self._previous_user = transcript
                    self._previous_reply = reply
                    await self._events.put({
                        "type": "session.output_audio.done",
                        "reply_expectation": "open_answer",
                    })
                    return

                tts_is_ack = False
                if decision.acknowledgement:
                    acknowledgement = phrase_text[decision.acknowledgement]
                    await self._events.put(
                        {"type": "session.output_transcript.delta", "delta": acknowledgement}
                    )
                    tts_task = asyncio.create_task(
                        relay_phrase(decision.acknowledgement, acknowledgement),
                        name=f"cube-elevenlabs-ack-{turn_id}",
                    )
                    tts_is_ack = True

                stage = "muse_connect"
                muse_started = time.monotonic()
                link = await self._connect_muse()
                turn_type, replay_scope = self._turn_types()
                new_side_chat = self._muse_session_id is None
                if new_side_chat:
                    # Muse creates side chats on first send, so subscribe after
                    # acknowledgement for that first turn.
                    self._muse_session_id = str(uuid4())
                subscribed = getattr(link, "chat_subscribed", None)
                if not new_side_chat:
                    subscriber = asyncio.create_task(
                        consume_chat(), name=f"cube-muse-events-{turn_id}"
                    )
                    if isinstance(subscribed, asyncio.Event):
                        await subscribed.wait()
                stage = "muse_send"
                raw_acknowledgement = await link.send_chat(
                    _format_cube_voice_prompt(transcript),
                    self._muse_session_id,
                    output_modality="text",
                )
                acknowledgement = _normalize_muse_acknowledgement(raw_acknowledgement)
                result = acknowledgement.get("result", acknowledgement)
                chat_id = result.get("session_id") if isinstance(result, dict) else None
                if not isinstance(chat_id, str) or not chat_id:
                    raise LiveProviderError("Muse voice acknowledgement omitted its chat ID")
                turn = turn_type(
                    self._muse_session_id or chat_id,
                    owns_chat=True,
                    replay_scope=replay_scope,
                )
                for _event in turn.acknowledge(acknowledgement):
                    pass
                if subscriber is None:
                    # The first send creates the side chat; subscribe immediately
                    # after its acknowledgement to receive that chat's events.
                    subscriber = asyncio.create_task(
                        consume_chat(), name=f"cube-muse-events-{turn_id}"
                    )
                    if isinstance(subscribed, asyncio.Event):
                        await subscribed.wait()
                spoken: dict[str, str] = {}
                muse_wait_span = trace.child("muse.wait", provider="muse", turn_id=turn_id)
                while True:
                    if tts_task is not None and tts_task.done():
                        try:
                            await tts_task
                        except Exception as error:
                            if not tts_is_ack:
                                stage = "elevenlabs_tts"
                                raise
                            trace.event("speech.ack_failed", status="failed",
                                        error_type=type(error).__name__)
                        tts_task = None
                        if tts_is_ack and not text_events.empty():
                            tts_task = asyncio.create_task(
                                relay_speech(), name=f"cube-elevenlabs-tts-{turn_id}"
                            )
                        tts_is_ack = False
                        stage = "muse_wait"
                    if turn.complete(time.monotonic()):
                        break
                    stage = "muse_wait"
                    try:
                        raw = await asyncio.wait_for(raw_events.get(), timeout=0.2)
                    except TimeoutError:
                        continue
                    if isinstance(raw, BaseException):
                        stage = f"muse_chat_{type(raw).__name__}"
                        raise LiveProviderError(
                            f"Muse chat subscription failed ({type(raw).__name__})"
                        ) from None
                    turn.feed(raw)
                    emitted = await self._emit_replies(turn, spoken, text_events)
                    if emitted and tts_task is None:
                        # Do not open ElevenLabs while waiting for Muse to produce text.
                        stage = "elevenlabs_tts"
                        tts_task = asyncio.create_task(
                            relay_speech(), name=f"cube-elevenlabs-tts-{turn_id}"
                        )
                muse_wait_span.end(status="completed")
                reply_text = "\n".join(
                    reply.text.strip()
                    for reply in turn.replies.values()
                    if reply.text.strip()
                )
                if not reply_text:
                    raise LiveProviderError("Muse returned no assistant reply")
                muse_elapsed_ms = round((time.monotonic() - muse_started) * 1000)
                trace.event(
                    "voice.reply_received",
                    turn_id=turn_id,
                    chars=len(reply_text),
                    elapsed_ms=muse_elapsed_ms,
                )
                _LOGGER.info(
                    "voice Muse reply received session_id=%s turn_id=%s reply_chars=%d elapsed_ms=%d",
                    trace.session_id, turn_id, len(reply_text), muse_elapsed_ms,
                )
                expectation_task = None
                classify_expectation = getattr(
                    self._internal_turn_handler, "classify_reply_expectation", None
                )
                if callable(classify_expectation):
                    expectation_task = asyncio.create_task(
                        classify_expectation(user_turn=transcript, reply=reply_text),
                        name=f"cube-reply-expectation-{turn_id}",
                    )
                self._previous_user = transcript
                self._previous_reply = reply_text
                self._muse_session_id = turn.session_id
                turn.retire()
                if self._conversation_registry is not None and self._conversation_key is not None:
                    stage = "conversation_persist"
                    await self._conversation_registry.save(
                        self._conversation_key,
                        self._muse_session_id,
                        self._replay_scope,
                    )
            await text_events.put(_TEXT_END)
            if tts_task is not None:
                stage = "elevenlabs_tts_finalize"
                try:
                    await tts_task
                except Exception as error:
                    if not tts_is_ack:
                        raise
                    trace.event("speech.ack_failed", status="failed",
                                error_type=type(error).__name__)
                if tts_is_ack and not text_events.empty():
                    tts_task = asyncio.create_task(
                        relay_speech(), name=f"cube-elevenlabs-tts-{turn_id}"
                    )
                    await tts_task
            if expectation_task is not None:
                try:
                    expectation = await asyncio.wait_for(expectation_task, 1.5)
                except Exception as error:
                    expectation_task.cancel()
                    await asyncio.gather(expectation_task, return_exceptions=True)
                    expectation = None
                    _LOGGER.info(
                        "voice reply expectation skipped session_id=%s turn_id=%s error_type=%s",
                        trace.session_id, turn_id, type(error).__name__,
                    )
            else:
                expectation = None
            self._last_reply_expectation = expectation
            await self._events.put(
                {
                    "type": "session.output_audio.done",
                    "reply_expectation": expectation,
                }
            )
            trace.event(
                "voice.reply_ready",
                turn_id=turn_id,
                chars=len(reply_text),
                expectation=expectation or "unknown",
                elapsed_ms=round((time.monotonic() - turn_started) * 1000),
            )
            _LOGGER.info(
                "voice reply ready session_id=%s turn_id=%s reply_chars=%d expectation=%s elapsed_ms=%d",
                trace.session_id, turn_id, len(reply_text), expectation or "unknown",
                round((time.monotonic() - turn_started) * 1000),
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            trace = current_observation()
            trace.event("voice.turn.failed", status="failed", stage=stage,
                        error_type=type(error).__name__)
            if "muse_wait_span" in locals():
                muse_wait_span.update(status="failed", error_type=type(error).__name__)
                muse_wait_span.end()
            _LOGGER.warning(
                "voice turn failed session_id=%s turn_id=%s stage=%s error_type=%s elapsed_ms=%d",
                current_observation().session_id,
                turn_id,
                stage,
                type(error).__name__,
                round((time.monotonic() - turn_started) * 1000),
            )
            await self._events.put(
                LiveProviderError(
                    f"voice turn failed ({type(error).__name__})"
                )
            )
        finally:
            if not text_events.full():
                text_events.put_nowait(_TEXT_END)
            other_tasks = []
            if subscriber is not None:
                other_tasks.append(subscriber)
            if tts_task is not None:
                other_tasks.append(tts_task)
            for task in other_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*other_tasks, return_exceptions=True)
            self._busy = False

    async def _connect_muse(self) -> Any:
        if self._link is not None:
            return self._link
        try:
            link = await self._host.wait_for_link(self._startup_timeout)
            if self._conversation_registry is not None and self._conversation_key is not None:
                self._muse_session_id, self._replay_scope = await self._conversation_registry.load(
                    self._conversation_key
                )
        except Exception as error:
            raise LiveProviderError(
                f"Muse conversation unavailable ({type(error).__name__})"
            ) from None
        self._link = link
        return link

    async def _emit_replies(
        self, turn: Any, spoken: dict[str, str], text_events: asyncio.Queue[object | str]
    ) -> bool:
        emitted = False
        for message_id, reply in turn.replies.items():
            previous = spoken.get(message_id, "")
            candidate = reply.streamed
            if reply.done and reply.text.startswith(candidate):
                candidate = reply.text
            if not candidate.startswith(previous):
                continue
            delta = candidate[len(previous) :]
            if not delta:
                continue
            spoken[message_id] = candidate
            emitted = True
            await self._events.put(
                {"type": "session.output_transcript.delta", "delta": delta}
            )
            await text_events.put(delta)
        return emitted

    def _turn_types(self) -> tuple[Callable[..., Any], Any]:
        if self._turn_factory is not None:
            return self._turn_factory, self._replay_scope
        try:
            from musegadget.muse_turn import MuseTurn, ReplayScope
        except ImportError:
            raise LiveProviderError(
                "Muse voice stack requires the optional muse dependency group"
            ) from None
        if self._replay_scope is None:
            self._replay_scope = ReplayScope()
        return MuseTurn, self._replay_scope


def muse_eleven_provider_factory(
    *,
    muse_host: MuseHost,
    speech: SpeechSynthesizer,
    recognizer: SpeechRecognizer,
    conversation_registry: MuseConversationRegistry | None = None,
    internal_turn_factory: InternalTurnFactory | None = None,
) -> Callable[[Any, Any], MuseElevenProvider]:
    """Create one isolated turn state object per authenticated Cube session."""

    def create_provider(principal: Any, session: Any) -> MuseElevenProvider:
        owner_id = getattr(principal, "owner_id", None)
        conversation_id = getattr(session, "conversation_id", None)
        key = (
            f"{owner_id}:{conversation_id}"
            if owner_id is not None and conversation_id is not None
            else None
        )
        return MuseElevenProvider(
            muse_host,
            speech,
            recognizer,
            conversation_registry=conversation_registry,
            conversation_key=key,
            internal_turn_handler=(
                internal_turn_factory(principal, session)
                if internal_turn_factory is not None
                else None
            ),
        )

    return create_provider


def _normalize_muse_acknowledgement(response: Any) -> dict[str, Any]:
    """Unwrap the pinned SDK's HTTP wrapper before feeding MuseTurn."""

    if not isinstance(response, dict):
        raise LiveProviderError("Muse returned an invalid chat acknowledgement")
    if "status" in response and "response" in response:
        status = response.get("status")
        if (
            response.get("ok") is not True
            or not isinstance(status, int)
            or isinstance(status, bool)
            or not 200 <= status < 300
        ):
            raise LiveProviderError("Muse rejected the chat request")
        body = response.get("response")
        if isinstance(body, dict):
            return body
        raise LiveProviderError("Muse returned an invalid chat acknowledgement")
    return response


_CONTROL_CHARACTERS = re.compile("[\\x00-\\x08\\x0b-\\x1f\\x7f-\\x9f]")


def _clean_transcript(value: Any) -> str:
    if not isinstance(value, str):
        raise LiveProviderError("ElevenLabs returned an invalid transcript")
    transcript = _CONTROL_CHARACTERS.sub("", value).strip()
    if len(transcript.encode("utf-8")) > 32_768:
        raise LiveProviderError("ElevenLabs transcript exceeded its limit")
    return transcript


def _format_cube_voice_prompt(transcript: str) -> str:
    """Wrap Cube speech for spoken answers while keeping UI transcript raw."""

    return (
        f"{CUBE_VOICE_RESPONSE_INSTRUCTION}\n\n"
        f"Spoken request:\n{transcript}"
    )
