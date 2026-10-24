"""Connect finalized Live delegations to durable jobs and spoken updates."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable, Sequence
from uuid import UUID

from contracts.models import (
    JobSnapshot,
    RecentTurn,
    VoiceContextAppend,
    VoiceSessionContextAppend,
    VoiceSessionEvent,
)
from device.transport import DeviceSocket, DeviceTransportError
from voice.delegation import DelegationTranscriptBuffer, build_task_request
from voice.jobs import GatewayJobClient, JobBridgeError
from voice.observability import start_session
from service.time_tools import current_time_place, get_current_time


TERMINAL_JOB_STATES = {"succeeded", "failed", "cancelled", "needs_reconciliation"}
_LOGGER = logging.getLogger(__name__)
_CLEAR_RESULT_CONSENT = {
    "yes",
    "yes please",
    "yes read it",
    "yes read them",
    "yes tell me",
    "yeah",
    "yeah please",
    "yeah read it",
    "yeah read them",
    "yeah tell me",
    "yep",
    "sure",
    "sure please",
    "sure read it",
    "sure read them",
    "okay",
    "okay please",
    "okay read it",
    "okay read them",
    "ok",
    "ok please",
    "ok read it",
    "ok read them",
    "go ahead",
    "read it",
    "read them",
    "tell me",
    "tell me please",
    "please do",
    "please read it",
    "please read them",
    "i want to hear it",
    "i want to hear them",
    "i would like to hear it",
    "i would like to hear them",
}


class VoiceJobCoordinator:
    """Commit complete user turns, submit jobs, and return safe context to Live."""

    def __init__(
        self,
        *,
        jobs: GatewayJobClient,
        device_id: str,
        conversation_id: UUID,
        session_id: UUID,
        locale: str,
        timezone_name: str,
        context_revision: int = 0,
        recent_turns: Sequence[RecentTurn] = (),
        transcript_quiet_seconds: float = 1.0,
        poll_interval_seconds: float = 0.5,
        pending_summaries: Callable[[], tuple[JobSnapshot, ...]] | None = None,
        mark_spoken: Callable[[UUID, int], None] | None = None,
    ) -> None:
        if not device_id.strip() or not locale.strip() or not timezone_name.strip():
            raise ValueError("voice job identity and locale context must not be blank")
        if transcript_quiet_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("voice job timing values must be positive")
        self._jobs = jobs
        self._device_id = device_id
        self._conversation_id = conversation_id
        self._session_id = session_id
        self._locale = locale
        self._timezone = timezone_name
        self._context_revision = context_revision
        self._recent_turns = tuple(recent_turns)
        self._quiet_seconds = transcript_quiet_seconds
        self._poll_seconds = poll_interval_seconds
        self._pending_summaries_provider = pending_summaries
        self._mark_spoken = mark_spoken
        self._pending_summaries: tuple[JobSnapshot, ...] = ()
        self._result_offer_sent = False
        self._result_offer_transcript = ""
        self._result_offer_seen = False
        self._result_offer_resolved = False
        self._consent_fragments: list[str] = []
        self._consent_audio_finished = False
        self._consent_task: asyncio.Task[None] | None = None
        self._released_summaries: tuple[JobSnapshot, ...] = ()
        self._buffer = DelegationTranscriptBuffer(
            transcript_quiet_seconds=transcript_quiet_seconds
        )
        self._audio_finished = False
        self._socket: DeviceSocket | None = None
        self._quiet_task: asyncio.Task[None] | None = None
        self._job_tasks: set[asyncio.Task[None]] = set()
        self._closed = False
        self._answer_wait_started: float | None = None
        self._delegation_deadline_task: asyncio.Task[None] | None = None
        self._trace = start_session(session_id, "voice.bridge")
        self._delegation_spans = {}
        self._clock_fragments: list[str] = []
        self._clock_last_end_ms: int | None = None
        self._clock_last_received: float | None = None
        self._clock_task: asyncio.Task[None] | None = None
        self._clock_handled_text: str | None = None
        self._clock_handled_answer: str | None = None
        self._clock_handled_at: float | None = None

    @property
    def has_pending_response(self) -> bool:
        """Keep the voice session open while a delegated answer is outstanding."""

        if self._closed:
            return False
        return (
            self._buffer.has_delegation
            or (self._quiet_task is not None and not self._quiet_task.done())
            or any(not task.done() for task in self._job_tasks)
            or (self._clock_task is not None and not self._clock_task.done())
            or (
                self._answer_wait_started is not None
                and time.monotonic() - self._answer_wait_started < 15.0
            )
        )

    async def on_event(self, event: VoiceSessionEvent, socket: DeviceSocket) -> None:
        """Accept gateway voice events; caller supplies local VAD completion separately."""

        if self._closed:
            return
        if event.session_id != self._session_id:
            raise JobBridgeError("voice event belongs to another session")
        self._socket = socket
        if event.type in {"session_closed", "session_error"}:
            await self.close()
            return
        if event.type == "session_ready":
            # Muse owns V2 turns; do not inject queued Cube worker results into
            # its conversation through the legacy Live context channel.
            if event.half_duplex is not True:
                await self._offer_pending_results()
            return
        received_at = time.monotonic()
        if event.type == "transcript_delta":
            self._trace.transcript(event.direction or "user", event.delta or "")
            self._trace.event("transcript.received", direction=event.direction,
                              chars=len(event.delta or ""), start_ms=event.start_ms, end_ms=event.end_ms)
            _LOGGER.info(
                "voice transcript event session_id=%s direction=%s chars=%d start_ms=%s end_ms=%s",
                self._session_id, event.direction, len(event.delta or ""), event.start_ms, event.end_ms,
            )
        if event.type == "transcript_delta" and event.direction == "assistant":
            self._answer_wait_started = None
            assert event.delta is not None
            self._record_result_offer_transcript(event.delta)
            if self._released_summaries:
                self._mark_released_summaries()
        elif event.type == "transcript_delta" and event.direction == "user":
            assert event.delta is not None
            accepted = self._buffer.record_transcript_delta(event.delta, received_at=received_at, start_ms=event.start_ms)
            self._trace.event("transcript.filtered", accepted=accepted,
                              chars=len(event.delta), start_ms=event.start_ms)
            _LOGGER.info(
                "voice user transcript boundary session_id=%s accepted=%s chars=%d start_ms=%s",
                self._session_id, accepted, len(event.delta), event.start_ms,
            )
            self._record_clock_fragment(event, received_at=received_at)
            if (
                self._pending_summaries
                and self._result_offer_seen
                and not self._result_offer_resolved
            ):
                self._consent_fragments.append(event.delta)
                if self._consent_audio_finished:
                    self._schedule_result_consent()
            self._schedule_commit()
        elif event.type == "delegation_created":
            _LOGGER.info("voice delegation event target=%s offset_ms=%s", event.target, event.offset_ms)
            assert event.delegation_id is not None and event.target is not None
            assert event.offset_ms is not None
            # A delayed provider delegation can refer to the read-only request
            # already answered by our clock. Supply facts quietly; don't submit
            # a duplicate worker job or speak the answer a second time.
            if (
                event.target == "client"
                and self._clock_handled_text == self._buffer.transcript
                and self._clock_handled_answer is not None
                and self._clock_handled_at is not None
                and received_at - self._clock_handled_at < 10.0
            ):
                await self._append_context(self._clock_handled_answer, kind="thinking",
                                           delegation_id=event.delegation_id)
                self._trace.event("clock.delegation_satisfied", delegation_id=event.delegation_id)
                self._buffer.reset_turn()
                self._audio_finished = False
                return
            self._buffer.record_delegation(
                delegation_id=event.delegation_id,
                target=event.target,
                offset_ms=event.offset_ms,
            )
            if event.target == "client":
                self._delegation_spans[event.delegation_id] = self._trace.child(
                    "delegation.wait", delegation_id=event.delegation_id, offset_ms=event.offset_ms,
                )
                if self._delegation_deadline_task is not None:
                    self._delegation_deadline_task.cancel()
                self._delegation_deadline_task = asyncio.create_task(
                    self._expire_incomplete_delegation(event.delegation_id)
                )
            self._schedule_commit()

    def mark_audio_finished(self) -> None:
        """Signal local VAD end-of-turn; transcript quiet timer then commits exact text."""

        if self._closed:
            return
        self._buffer.mark_audio_finished()
        self._trace.event("input.endpoint")
        self._audio_finished = True
        if self._pending_summaries:
            if self._result_offer_seen and not self._result_offer_resolved:
                self._consent_audio_finished = True
                self._schedule_result_consent()
            elif not self._result_offer_seen and not self._buffer.has_delegation:
                self._buffer.reset_turn()
                self._audio_finished = False
        self._schedule_commit()

    async def close(self) -> None:
        """Stop quiet/poll tasks and release the job client."""

        if self._closed:
            return
        self._closed = True
        for span in self._delegation_spans.values():
            span.end(status="session_closed")
        self._delegation_spans.clear()
        self._trace.end(status="closed")
        tasks = set(self._job_tasks)
        if self._clock_task is not None:
            tasks.add(self._clock_task)
        if self._delegation_deadline_task is not None:
            tasks.add(self._delegation_deadline_task)
        if self._quiet_task is not None:
            tasks.add(self._quiet_task)
            self._quiet_task = None
        if self._consent_task is not None:
            tasks.add(self._consent_task)
            self._consent_task = None
        current = asyncio.current_task()
        for task in tasks:
            if task is not current and not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(
                *(task for task in tasks if task is not current),
                return_exceptions=True,
            )
        await self._jobs.close()

    def _schedule_commit(self) -> None:
        # VAD endpoints without a provider request must never become pending
        # answer work. Keep transcript state for a later genuine delegation.
        if self._closed or not self._buffer.has_delegation:
            return
        if self._quiet_task is not None and not self._quiet_task.done():
            self._quiet_task.cancel()
        self._quiet_task = asyncio.create_task(
            self._commit_after_quiet(), name=f"cube-voice-commit-{self._session_id}"
        )

    def _record_clock_fragment(self, event: VoiceSessionEvent, *, received_at: float) -> None:
        """Bound a transcript turn and schedule only recognized clock requests."""
        if self._clock_task is not None and not self._clock_task.done():
            self._clock_task.cancel()
        self._clock_task = None
        timeline_gap = (
            event.start_ms is not None and self._clock_last_end_ms is not None
            and event.start_ms - self._clock_last_end_ms >= 1200
        )
        arrival_gap = (
            event.start_ms is None and self._clock_last_received is not None
            and received_at - self._clock_last_received >= 2.0
        )
        if timeline_gap or arrival_gap or self._clock_handled_text is not None:
            self._clock_fragments.clear()
            self._clock_handled_text = None
            self._clock_handled_answer = None
            self._clock_handled_at = None
        self._clock_fragments.append(event.delta or "")
        self._clock_last_end_ms = event.end_ms
        self._clock_last_received = received_at
        # Bound memory and refuse to route a clipped or mixed long transcript.
        text = "".join(self._clock_fragments).strip()
        if len(text) > 500:
            self._clock_fragments.clear()
            return
        if current_time_place(text) is not None:
            self._clock_task = asyncio.create_task(
                self._answer_clock_after_quiet(text), name=f"cube-clock-{self._session_id}",
            )

    async def _answer_clock_after_quiet(self, text: str) -> None:
        # Transcript fragments are approximate; wait for a complete quiet turn
        # rather than replying at the first partial city name.
        await asyncio.sleep(max(1.0, self._quiet_seconds))
        if self._closed or self._buffer.has_delegation:
            return  # Genuine delegation owns the worker's existing fast path.
        place = current_time_place(text)
        if place is None:
            return
        with self._trace.operation("clock.lookup", as_type="tool", tool_name="get_current_time") as span:
            span.io(input={"question": text, "place": place})
            result = get_current_time(place)
            span.io(output=result)
            span.update(status=result["status"])
        self._answer_wait_started = time.monotonic()
        content = "Application clock lookup result for the user's current question: " + result["answer"]
        if await self._append_session_context(content, kind="commentary"):
            self._clock_handled_text = text
            self._clock_handled_answer = result["answer"]
            self._clock_handled_at = time.monotonic()
            self._trace.event("clock.result_sent", status=result["status"])

    async def _commit_after_quiet(self) -> None:
        # Local VAD can remain open on TV/background speech. A provider
        # delegation plus a longer transcript quiet interval bounds that wait.
        await asyncio.sleep(self._quiet_seconds if self._audio_finished else max(2.0, self._quiet_seconds))
        if not self._audio_finished and self._buffer.has_delegation:
            self._buffer.mark_audio_finished()
        committed = self._buffer.ready(now=time.monotonic())
        if committed is None or self._closed:
            if self._buffer.has_delegation and not self._closed:
                _LOGGER.info("voice delegation commit waiting for transcript")
            return
        _LOGGER.info("voice delegation committed chars=%d offset_ms=%s", len(committed.transcript), committed.offset_ms)
        span = self._delegation_spans.pop(committed.delegation_id, None)
        if span is not None:
            span.end(status="committed", chars=len(committed.transcript))
        self._audio_finished = False
        request = build_task_request(
            committed,
            device_id=self._device_id,
            conversation_id=self._conversation_id,
            session_id=self._session_id,
            context_revision=self._context_revision,
            locale=self._locale,
            timezone_name=self._timezone,
            recent_turns=self._recent_turns,
        )
        try:
            with self._trace.operation("job.submit", delegation_id=committed.delegation_id,
                                       request_trace_id=request.trace_id) as submission:
                submission.io(input=request.request_text)
                accepted = await self._jobs.submit(request)
                submission.io(output={"job_id": str(accepted.job_id), "status": "accepted"})
                submission.update(job_id=accepted.job_id, status="accepted")
        except JobBridgeError:
            await self._append_context(
                "I could not start that request. Please try again.",
                kind="commentary",
                delegation_id=committed.delegation_id,
            )
            return
        self._recent_turns = request.recent_turns
        task = asyncio.create_task(
            self._follow_job(committed.delegation_id),
            name=f"cube-voice-job-{accepted.job_id}",
        )
        self._job_tasks.add(task)
        task.add_done_callback(self._job_tasks.discard)

    async def _expire_incomplete_delegation(self, delegation_id: str) -> None:
        await asyncio.sleep(8.0)
        if self._closed or not self._buffer.has_delegation:
            return
        self._buffer.reset_turn()
        self._audio_finished = False
        span = self._delegation_spans.pop(delegation_id, None)
        if span is not None:
            span.end(status="expired", reason="missing_bounded_transcript")
        _LOGGER.warning("voice delegation had no bounded request transcript; asking for repetition")
        await self._append_context(
            "I didn't catch the request clearly. Please repeat your question.",
            kind="commentary", delegation_id=delegation_id,
        )

    async def _follow_job(self, delegation_id: str) -> None:
        failures = 0
        while not self._closed:
            try:
                events = await self._jobs.events(delegation_id)
                failures = 0
            except JobBridgeError:
                failures += 1
                if failures >= 3:
                    await self._append_context(
                        "I lost access to that request's progress. It may still be running.",
                        kind="commentary",
                        delegation_id=delegation_id,
                    )
                    return
                await asyncio.sleep(self._poll_seconds)
                continue

            terminal = False
            for event in events:
                self._trace.event("job.event", delegation_id=delegation_id,
                                  job_id=event.job_id, kind=event.kind, status=event.status)
                if event.kind == "progress":
                    await self._append_context(
                        event.summary, kind="thinking", delegation_id=delegation_id
                    )
                elif event.kind == "approval_requested":
                    await self._append_context(
                        event.summary, kind="commentary", delegation_id=delegation_id
                    )
                elif event.kind == "result" or event.status in TERMINAL_JOB_STATES:
                    await self._append_context(
                        event.summary, kind="commentary", delegation_id=delegation_id
                    )
                terminal = event.status in TERMINAL_JOB_STATES
                if terminal:
                    break
            if terminal:
                return
            await asyncio.sleep(self._poll_seconds)

    async def _append_context(
        self,
        content: str,
        *,
        kind: str,
        delegation_id: str,
    ) -> bool:
        socket = self._socket
        if socket is None or self._closed:
            return False
        bounded = _truncate_utf8(content.strip(), 500)
        if not bounded:
            return False
        request = VoiceContextAppend(
            session_id=self._session_id,
            delegation_id=delegation_id,
            kind=kind,
            content=bounded,
        )
        # Context acceptance precedes the provider's spoken answer. Reserve a
        # bounded interval until its first assistant transcript arrives.
        if kind == "commentary":
            self._answer_wait_started = time.monotonic()
        try:
            await socket.send_context_append(request)
            delivered = self._trace.child("context.sent", delegation_id=delegation_id, kind=kind, chars=len(bounded))
            delivered.io(output=bounded)
            delivered.end()
        except DeviceTransportError:
            await self.close()
            return False
        return True

    async def _offer_pending_results(self) -> None:
        if self._pending_summaries_provider is None or self._pending_summaries:
            return
        try:
            pending = self._pending_summaries_provider()
        except Exception as error:
            _LOGGER.warning("pending result read failed (%s)", type(error).__name__)
            return
        if not pending:
            return
        self._pending_summaries = pending
        if len(pending) == 1:
            offer = "A result from an earlier request is ready. Would you like me to read it?"
        else:
            offer = "Results from earlier requests are ready. Would you like me to read them?"
        instructions = (
            f"Immediately say only: {offer} Do not reveal any title, summary, source, "
            "file, or other result detail until the user clearly agrees after hearing "
            "this offer. If they do not clearly agree, do not read details."
        )
        self._result_offer_sent = await self._append_session_context(
            instructions,
            kind="instructions",
        )

    def _record_result_offer_transcript(self, delta: str) -> None:
        if (
            not self._pending_summaries
            or not self._result_offer_sent
            or self._result_offer_seen
            or self._result_offer_resolved
        ):
            return
        self._result_offer_transcript = (
            self._result_offer_transcript + " " + delta
        )[-1200:]
        normalized = _normalize_transcript(self._result_offer_transcript)
        if any(
            phrase in normalized
            for phrase in (
                "would you like me to read it",
                "would you like me to read them",
            )
        ):
            self._result_offer_seen = True

    def _schedule_result_consent(self) -> None:
        if (
            self._closed
            or not self._pending_summaries
            or not self._result_offer_seen
            or self._result_offer_resolved
            or not self._consent_audio_finished
        ):
            return
        if self._consent_task is not None and not self._consent_task.done():
            self._consent_task.cancel()
        self._consent_task = asyncio.create_task(
            self._resolve_result_consent(),
            name=f"cube-result-consent-{self._session_id}",
        )

    async def _resolve_result_consent(self) -> None:
        await asyncio.sleep(self._quiet_seconds)
        if self._closed or not self._pending_summaries:
            return
        self._result_offer_resolved = True
        transcript = "".join(self._consent_fragments).strip()
        self._consent_fragments.clear()
        self._consent_audio_finished = False
        if not _is_clear_result_consent(transcript):
            if not self._buffer.has_delegation:
                self._buffer.reset_turn()
                self._audio_finished = False
            return

        self._buffer.reset_turn()
        self._audio_finished = False

        prefix = "The user clearly agreed to hear these completed results. Read them briefly: "
        included: list[JobSnapshot] = []
        content = prefix
        for snapshot in self._pending_summaries:
            addition = (" " if included else "") + snapshot.summary.strip()
            candidate = content + addition
            if len(candidate.encode("utf-8")) > 500:
                if not included:
                    available = 500 - len(content.encode("utf-8"))
                    excerpt = _truncate_utf8(snapshot.summary.strip(), available)
                    if excerpt:
                        content += excerpt
                        included.append(snapshot)
                break
            content = candidate
            included.append(snapshot)
        if not included:
            return
        if await self._append_session_context(
            content,
            kind="commentary",
        ):
            self._released_summaries = tuple(included)
            self._pending_summaries = tuple(
                snapshot for snapshot in self._pending_summaries if snapshot not in included
            )

    async def _append_session_context(self, content: str, *, kind: str) -> bool:
        socket = self._socket
        if socket is None or self._closed:
            return False
        bounded = _truncate_utf8(content.strip(), 500)
        if not bounded:
            return False
        request = VoiceSessionContextAppend(
            session_id=self._session_id,
            kind=kind,
            content=bounded,
        )
        try:
            await socket.send_session_context_append(request)
            delivered = self._trace.child("session_context.sent", kind=kind, chars=len(bounded))
            delivered.io(output=bounded)
            delivered.end()
        except DeviceTransportError:
            await self.close()
            return False
        return True

    def _mark_released_summaries(self) -> None:
        if self._mark_spoken is not None:
            for snapshot in self._released_summaries:
                try:
                    self._mark_spoken(snapshot.job_id, snapshot.context_revision)
                except Exception as error:
                    _LOGGER.warning("spoken result state failed (%s)", type(error).__name__)
        self._released_summaries = ()


def _truncate_utf8(text: str, maximum_bytes: int) -> str:
    encoded = text.encode("utf-8")[:maximum_bytes]
    while encoded:
        try:
            return encoded.decode("utf-8")
        except UnicodeDecodeError:
            encoded = encoded[:-1]
    return ""


def _is_clear_result_consent(text: str) -> bool:
    return _normalize_transcript(text) in _CLEAR_RESULT_CONSENT


def _normalize_transcript(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.casefold())
    return " ".join(normalized.split())
