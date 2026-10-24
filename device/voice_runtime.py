"""Compose idle wake capture, local session control, and gateway voice relay."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from uuid import UUID, uuid4
from voice.observability import Observation, start_session
from voice.audio_levels import PcmLevelMeter

from contracts.models import SessionResponse, VoiceSessionEvent
from device.audio_buffer import AudioFrame, BufferedAudioFrame
from device.audio_runtime import CaptureStream, PiAudioRuntime
from device.led_ring import VoiceIndicatorState
from device.reconnect import ReconnectSupervisor, RetryCallback
from device.session_control import (
    LocalSessionController,
    PreRollBuffer,
    SpeechDetector,
    WakeDetector,
)
from device.transport import DeviceSocket, DeviceTransportError
from device.timer_chime import timer_chime_frames
from device.voice_relay import CaptureFrameDecision, VoiceEventHandler
from device.voice_session import (
    SessionLifecycle,
    SocketFactory,
    VoiceEventHandlerFactory,
    VoiceSessionCallbacks,
    VoiceSessionCallbacksFactory,
    run_gateway_voice_session,
)


CaptureFactory = Callable[[], CaptureStream]
InterruptDetector = Callable[[bytes], bool]
TurnEndDetector = Callable[[bytes], bool]
NoAudioSend = Callable[[UUID, BufferedAudioFrame], Awaitable[None]]
IdleNotificationCheck = Callable[[], Awaitable[None]]
WakeAcknowledgement = Callable[[], Awaitable[None]]
ProcessingStarted = Callable[[], Awaitable[None]]
ResponseCompleted = Callable[[], None]


class _ActivityCallbacks:
    """Observe recognized input without changing job orchestration."""

    def __init__(self, wrapped: VoiceSessionCallbacks, controller: LocalSessionController,
                 trace: Observation | None = None,
                 on_processing_started: ProcessingStarted | None = None,
                 on_response_completed: ResponseCompleted | None = None) -> None:
        self._wrapped = wrapped
        self._controller = controller
        self._trace = trace or Observation()
        self._on_processing_started = on_processing_started
        self._on_response_completed = on_response_completed
        self.ready_at: float | None = None
        self.half_duplex = False
        self.has_recognized_input = False
        self.processing_response = False
        self.response_audio_active = False

    def set_trace(self, trace: Observation) -> None:
        """Bind the root after session creation; callbacks are constructed first."""
        self._trace = trace

    async def on_event(self, event: VoiceSessionEvent, socket: DeviceSocket) -> None:
        if event.type == "session_ready":
            self.ready_at = asyncio.get_running_loop().time()
            self.half_duplex = event.half_duplex is True
        if event.type == "transcript_delta" and event.delta:
            self._trace.transcript(event.direction or "user", event.delta)
            self._trace.event(
                "transcript.received",
                direction=event.direction or "user",
                chars=len(event.delta),
                accepted=True,
                start_ms=event.start_ms,
                end_ms=event.end_ms,
            )
        if event.type == "transcript_delta" and event.direction == "user" and event.delta:
            self.has_recognized_input = True
            self._controller.note_input_activity()
        if event.type == "assistant_processing":
            self.processing_response = True
            if self._on_processing_started is not None:
                await self._on_processing_started()
        elif event.type == "assistant_response_started":
            self.response_audio_active = True
            self._controller.note_input_activity()
            _LOGGER.info(
                "voice response audio started session_id=%s", event.session_id
            )
        elif event.type == "assistant_response_complete":
            self.processing_response = False
            self.response_audio_active = False
            self._controller.note_response_activity(event.reply_expectation)
            expectation = event.reply_expectation or "unknown"
            self._trace.event(
                "voice.reply_expectation",
                expectation=expectation,
                idle_timeout_seconds=self._controller.idle_timeout_seconds,
            )
            if self._on_response_completed is not None:
                self._on_response_completed()
            _LOGGER.info(
                "voice response audio completed after speaker drain session_id=%s expectation=%s follow_up_window_seconds=%s",
                event.session_id, expectation, self._controller.idle_timeout_seconds,
            )
        elif event.type == "assistant_turn_ignored":
            self.processing_response = False
            self.response_audio_active = False
            self._controller.restore_response_deadline()
            if self._on_response_completed is not None:
                self._on_response_completed()
        await self._wrapped.on_event(event, socket)

    def mark_audio_finished(self) -> None:
        self._wrapped.mark_audio_finished()

    async def close(self) -> None:
        await self._wrapped.close()

    @property
    def has_pending_response(self) -> bool:
        return bool(getattr(self._wrapped, "has_pending_response", False))


_LOGGER = logging.getLogger(__name__)


async def _discard_duplicate_audio(
    _session_id: UUID, _frame: BufferedAudioFrame
) -> None:
    """The gateway relay owns audio sending after session startup."""


class LocalVoiceRuntime:
    """Run injected wake/VAD logic around a reusable Pi audio runtime.

    Continuous capture can transfer the open microphone into the session.
    Bounded pre-roll and startup audio remain in memory; the session runtime
    owns microphone transmission after provider readiness.
    Wake, speech, and interruption decisions remain injected policies.
    """

    def __init__(
        self,
        *,
        session_client: SessionLifecycle,
        runtime: PiAudioRuntime,
        idle_capture_factory: CaptureFactory,
        gateway_url: str,
        device_id: str,
        device_token: str,
        conversation_id: UUID,
        locale: str,
        timezone_name: str,
        wake_detect: WakeDetector,
        speech_detect: SpeechDetector,
        interrupt_detect: InterruptDetector,
        turn_end_detect: TurnEndDetector | None = None,
        session_callbacks_factory: VoiceSessionCallbacksFactory | None = None,
        idle_timeout_seconds: float = 3.0,
        max_session_seconds: float | None = None,
        idle_check_interval_seconds: float = 0.1,
        idle_notification_check: IdleNotificationCheck | None = None,
        idle_notification_interval_seconds: float = 15.0,
        on_event: VoiceEventHandler | None = None,
        on_event_factory: VoiceEventHandlerFactory | None = None,
        socket_factory: SocketFactory = DeviceSocket.open,
        on_retry: RetryCallback | None = None,
        continuous_capture: bool = False,
        wake_acknowledgement: WakeAcknowledgement | None = None,
        session_end_acknowledgement: WakeAcknowledgement | None = None,
        speech_updates_idle: bool = True,
        indicator_state: Callable[[VoiceIndicatorState], None] | None = None,
        indicator_audio_level: Callable[[float], None] | None = None,
    ) -> None:
        if idle_check_interval_seconds <= 0 or idle_notification_interval_seconds <= 0:
            raise ValueError("idle check interval must be positive")
        audio_format = runtime.audio_format
        preroll_frames = min(
            runtime.capture_buffer.capacity_frames,
            max(1, 500 // audio_format.frame_duration_ms),
        )
        self._session_client = session_client
        self._runtime = runtime
        self._capture_factory = idle_capture_factory
        self._gateway_url = gateway_url
        self._device_id = device_id
        self._device_token = device_token
        self._conversation_id = conversation_id
        self._locale = locale
        self._timezone_name = timezone_name
        self._speech_detector = speech_detect
        self._wake_detector = wake_detect
        self._interrupt_detect = interrupt_detect
        self._turn_end_detect = turn_end_detect
        self._session_callbacks_factory = session_callbacks_factory
        self._idle_check_interval = idle_check_interval_seconds
        self._idle_notification_check = idle_notification_check
        self._idle_notification_interval = idle_notification_interval_seconds
        self._on_event = on_event
        self._on_event_factory = on_event_factory
        self._socket_factory = socket_factory
        self._on_retry = on_retry
        self._idle_capture: CaptureStream | None = None
        self._session_capture: CaptureStream | None = None
        self._continuous_capture = continuous_capture
        self._wake_acknowledgement = wake_acknowledgement
        self._session_end_acknowledgement = session_end_acknowledgement
        self._indicator_state_callback = indicator_state
        self._indicator_audio_level_callback = indicator_audio_level
        self._indicator_state: VoiceIndicatorState | None = None
        self._awaiting_response = False
        self._speech_muted: bool | None = None
        self._ack_task: asyncio.Task[None] | None = None
        self._active_task: asyncio.Task[SessionResponse] | None = None
        self._active_session_id: UUID | None = None
        self._active_callbacks: VoiceSessionCallbacks | None = None
        self._preroll_cutoff: datetime | None = None
        self._active_frames = 0
        self._turn_vad_speech_frames = 0
        self._trace = Observation()
        self._traced_playback = False
        self._idle_action_lock = asyncio.Lock()
        self._controller = LocalSessionController(
            pre_roll=PreRollBuffer(
                frame_bytes=audio_format.frame_bytes,
                frame_duration_ms=audio_format.frame_duration_ms,
                capacity_frames=preroll_frames,
            ),
            wake_detect=wake_detect,
            start_session=self._start_session,
            stop_session=self._stop_session,
            send_audio=_discard_duplicate_audio,
            flush_playback=runtime.flush_playback,
            speech_detect=speech_detect,
            idle_timeout_seconds=idle_timeout_seconds,
            max_session_seconds=max_session_seconds,
            speech_updates_idle=speech_updates_idle,
        )
        self._set_indicator_state(VoiceIndicatorState.OFF)

    @property
    def phase(self) -> str:
        return self._controller.phase

    def _set_indicator_state(self, state: VoiceIndicatorState) -> None:
        if self._indicator_state == state:
            return
        self._indicator_state = state
        if state != VoiceIndicatorState.SPEAKING:
            self._set_indicator_audio_level(0.0)
        if self._indicator_state_callback is None:
            return
        try:
            self._indicator_state_callback(state)
        except Exception as error:
            _LOGGER.warning("voice status update failed (%s)", type(error).__name__)

    def _set_indicator_audio_level(self, level: float) -> None:
        if self._indicator_audio_level_callback is None:
            return
        try:
            self._indicator_audio_level_callback(level)
        except Exception as error:
            _LOGGER.warning(
                "voice speech-level update failed (%s)", type(error).__name__
            )

    async def run(self, stop_event: asyncio.Event) -> None:
        """Run until stopped, retrying only sanitized audio/network failures."""

        if stop_event.is_set():
            return
        self._set_indicator_state(VoiceIndicatorState.OFF)
        supervisor: ReconnectSupervisor

        async def attempt() -> None:
            await self._run_until_stopped(stop_event)
            if stop_event.is_set():
                supervisor.stop()

        supervisor = ReconnectSupervisor(attempt, on_retry=self._on_retry)
        running = asyncio.create_task(supervisor.run(), name="cube-voice-reconnect")
        stopping = asyncio.create_task(stop_event.wait(), name="cube-voice-stop")
        idle_notifications = None
        if self._idle_notification_check is not None:
            idle_notifications = asyncio.create_task(
                self._monitor_idle_notifications(stop_event),
                name="cube-voice-idle-notifications",
            )
        try:
            done, _ = await asyncio.wait(
                (running, stopping), return_when=asyncio.FIRST_COMPLETED
            )
            if stopping in done:
                supervisor.stop()
            if running in done:
                await running
        finally:
            supervisor.stop()
            if not running.done():
                await running
            if not stopping.done():
                stopping.cancel()
            await asyncio.gather(stopping, return_exceptions=True)
            if idle_notifications is not None:
                idle_notifications.cancel()
                await asyncio.gather(idle_notifications, return_exceptions=True)
            await self._controller.close()
            await self._close_idle_capture()
            await self._close_session_capture()
            await self._finish_wake_acknowledgement()
            self._set_indicator_state(VoiceIndicatorState.OFF)

    async def _monitor_idle_notifications(self, stop_event: asyncio.Event) -> None:
        """Poll for local notifications without blocking audio-frame reads."""

        while not stop_event.is_set():
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=self._idle_notification_interval
                )
                return
            except TimeoutError:
                pass
            if (
                self._active_task is not None
                or self._controller.phase != "idle"
                or self._idle_notification_check is None
            ):
                continue
            async with self._idle_action_lock:
                if (
                    self._active_task is not None
                    or self._controller.phase != "idle"
                ):
                    continue
                try:
                    await self._idle_notification_check()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    _LOGGER.warning(
                        "idle notification check failed (%s)", type(error).__name__
                    )

    async def _run_until_stopped(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            if self._active_task is not None:
                await self._monitor_active_session(stop_event)
                continue

            capture = self._capture_factory()
            self._idle_capture = capture
            try:
                await capture.start()
                while not stop_event.is_set() and self._active_task is None:
                    payload = await capture.read_frame()
                    await self._controller.feed(
                        payload, captured_at=datetime.now(timezone.utc)
                    )
            finally:
                if self._idle_capture is capture:
                    self._idle_capture = None
                    await capture.close()

            if self._active_task is not None:
                await self._monitor_active_session(stop_event)

    async def _start_session(
        self, capture_preroll: tuple[BufferedAudioFrame, ...]
    ) -> UUID:
        async with self._idle_action_lock:
            return await self._start_session_locked(capture_preroll)

    async def _start_session_locked(
        self, capture_preroll: tuple[BufferedAudioFrame, ...]
    ) -> UUID:
        wake_started = asyncio.get_running_loop().time()
        attempt_id = uuid4()
        attempt_trace = start_session(attempt_id, "wake.attempt", attempt_id=attempt_id)
        try:
            if self._continuous_capture:
                capture = self._idle_capture
                if capture is None:
                    raise RuntimeError("wake capture missing at session handoff")
                capture.begin_session()
                self._runtime.adopt_capture(capture)
                self._session_capture = capture
                self._idle_capture = None
            else:
                await self._close_idle_capture()
        except BaseException as error:
            attempt_trace.update(
                status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
                failure_stage="capture_handoff", error_type=type(error).__name__,
                error_category="capture_handoff_failed",
                elapsed_ms=round((asyncio.get_running_loop().time() - wake_started) * 1000),
            )
            attempt_trace.end()
            raise
        self._active_frames = 0
        self._turn_vad_speech_frames = 0
        self._awaiting_response = False
        self._speech_muted = None
        self._set_indicator_state(VoiceIndicatorState.LISTENING)
        self._input_levels = PcmLevelMeter()
        self._initial_request_created_at = wake_started
        wake_metadata = getattr(self._wake_detector, "last_detection_metadata", {})
        self._traced_playback = False
        _LOGGER.info("wake detected; creating voice session attempt_id=%s", attempt_id)
        if self._wake_acknowledgement is not None:
            try:
                await self._wake_acknowledgement()
            except Exception as error:
                attempt_trace.event("wake.ack_failed", status="failed",
                                    error_type=type(error).__name__)
                _LOGGER.warning("wake acknowledgement failed attempt_id=%s error_type=%s",
                                attempt_id, type(error).__name__)
        self._reset_speech_detector()
        self._preroll_cutoff = max(
            (frame.captured_at for frame in capture_preroll), default=None
        )
        loop = asyncio.get_running_loop()
        created: asyncio.Future[SessionResponse] = loop.create_future()

        async def on_session_created(response: SessionResponse) -> None:
            self._active_session_id = response.session_id
            self._trace = start_session(response.session_id, "voice.device", attempt_id=attempt_id)
            attempt_trace.update(status="succeeded", voice_session_id=response.session_id,
                                 elapsed_ms=round((asyncio.get_running_loop().time() - wake_started) * 1000))
            if isinstance(self._active_callbacks, _ActivityCallbacks):
                self._active_callbacks.set_trace(self._trace)
            self._trace.event("wake.detected", **wake_metadata)
            self._trace.event("wake.session_created", wake_to_session_ms=round(
                (asyncio.get_running_loop().time() - wake_started) * 1000))
            if not created.done():
                created.set_result(response)

        def make_session_callbacks(
            response: SessionResponse,
        ) -> VoiceSessionCallbacks:
            assert self._session_callbacks_factory is not None
            callbacks = _ActivityCallbacks(
                self._session_callbacks_factory(response),
                self._controller,
                self._trace,
                on_processing_started=None,
                on_response_completed=self._mark_response_complete,
            )
            self._active_callbacks = callbacks
            return callbacks

        task = asyncio.create_task(
            run_gateway_voice_session(
                session_client=self._session_client,
                runtime=self._runtime,
                gateway_url=self._gateway_url,
                device_id=self._device_id,
                device_token=self._device_token,
                conversation_id=self._conversation_id,
                locale=self._locale,
                timezone=self._timezone_name,
                capture_preroll=capture_preroll,
                on_event=self._on_event,
                on_event_factory=self._on_event_factory,
                session_callbacks_factory=(
                    make_session_callbacks
                    if self._session_callbacks_factory is not None
                    else None
                ),
                on_session_created=on_session_created,
                on_capture_frame=self._handle_active_frame,
                socket_factory=self._socket_factory,
            ),
            name="cube-gateway-voice-session",
        )
        self._active_task = task
        try:
            done, _ = await asyncio.wait(
                (created, task), return_when=asyncio.FIRST_COMPLETED
            )
            if created not in done:
                await task
                raise RuntimeError("voice session ended before creation was acknowledged")
            response = created.result()
            if response.session_id != self._active_session_id:
                raise RuntimeError("created session identity changed")
            attempt_trace.end()
            return response.session_id
        except BaseException as error:
            outcome = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
            attempt_trace.update(status=outcome, failure_stage="session_create",
                                 error_type=type(error).__name__,
                                 error_category="timeout" if isinstance(error, TimeoutError) else "session_create_failed",
                                 elapsed_ms=round((asyncio.get_running_loop().time() - wake_started) * 1000))
            attempt_trace.end()
            _LOGGER.warning("voice session creation failed attempt_id=%s failure_stage=session_create error_type=%s",
                            attempt_id, type(error).__name__)
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if self._active_task is task:
                self._clear_active_session(task, status=outcome)
            await self._close_session_capture()
            raise

    async def _stop_session(self, session_id: UUID) -> None:
        if session_id != self._active_session_id:
            return
        task = self._active_task
        try:
            if task is not None and not task.done():
                task.cancel()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
        finally:
            if hasattr(self, "_input_levels"):
                levels = self._input_levels.snapshot()
                self._trace.event("audio.input_levels", status="final", **levels)
                _LOGGER.info("voice microphone final levels=%s", levels)
            await self._close_session_capture()
            await self._finish_wake_acknowledgement()
            if task is None or self._active_task is task:
                self._clear_active_session(task)

    async def _handle_active_frame(self, frame: AudioFrame) -> CaptureFrameDecision:
        captured_at = frame.metadata.captured_at
        if self._preroll_cutoff is not None and captured_at <= self._preroll_cutoff:
            return CaptureFrameDecision()
        callbacks = self._active_callbacks
        half_duplex = bool(getattr(callbacks, "half_duplex", False))
        playback_at = self._runtime.last_playback_activity
        playback_active = (
            playback_at is not None
            and (datetime.now(timezone.utc) - playback_at).total_seconds() <= 0.5
        )
        suppress_speech = half_duplex and (self._awaiting_response or playback_active)
        mute_reason = (
            "awaiting_response" if self._awaiting_response
            else "assistant_playback" if playback_active
            else "none"
        )
        if suppress_speech != self._speech_muted:
            self._speech_muted = suppress_speech
            self._trace.event(
                "audio.input_muted", muted=suppress_speech, reason=mute_reason,
            )
            _LOGGER.info(
                "voice upstream microphone %s session_id=%s reason=%s",
                "muted" if suppress_speech else "unmuted",
                self._active_session_id, mute_reason,
            )
        was_speech_active = self._controller.speech_active
        await self._controller.feed(
            frame.payload,
            captured_at=captured_at,
            suppress_speech=suppress_speech,
        )
        if self._controller.speech_active != was_speech_active:
            self._trace.event(
                "input.speech_candidate",
                status="started" if self._controller.speech_active else "ended",
                muted=suppress_speech,
            )
        if self._controller.speech_active:
            self._turn_vad_speech_frames += 1
        # Endpoint detection consumes the speech classification cached by feed.
        turn_ended = (
            not suppress_speech
            and self._turn_end_detect is not None
            and self._turn_end_detect(frame.payload)
        )
        if turn_ended:
            session_id = self._active_session_id
            sequence = getattr(frame.metadata, "sequence", None)
            vad_speech_ms = (
                self._turn_vad_speech_frames
                * self._runtime.audio_format.frame_duration_ms
            )
            self._turn_vad_speech_frames = 0
            self._trace.event(
                "input.endpoint",
                frames=self._active_frames + 1,
                reason="local_vad_endpoint",
                vad_speech_ms=vad_speech_ms,
            )
            _LOGGER.info(
                "voice input turn endpoint session_id=%s sequence=%s frames=%d vad_speech_ms=%d",
                session_id, sequence, self._active_frames + 1, vad_speech_ms,
            )
            self._awaiting_response = True
            self._set_indicator_state(VoiceIndicatorState.THINKING)
        else:
            vad_speech_ms = 0
        self._active_frames += 1
        self._input_levels.add(frame.payload)
        # An open conversation still acknowledges the wake phrase. Suppress
        # detection during audible assistant output to limit speaker echo.
        if (
            self._wake_acknowledgement is not None
            and not suppress_speech
            and (playback_at is None or (datetime.now(timezone.utc) - playback_at).total_seconds() > 1)
            and await self._wake_detector(frame.payload)
        ):
            _LOGGER.info("wake detected during active voice session")
            self._trace.event("wake.detected", **getattr(
                self._wake_detector, "last_detection_metadata", {}
            ))
            self._controller.note_input_activity()
            if self._ack_task is None or self._ack_task.done():
                self._ack_task = asyncio.create_task(self._play_active_wake_chime())
        if turn_ended:
            callbacks = self._active_callbacks
            if callbacks is not None:
                callbacks.mark_audio_finished()
        if not suppress_speech and self._interrupt_detect(frame.payload):
            await self._controller.interrupt()
        return CaptureFrameDecision(
            turn_ended=turn_ended,
            mute_audio=suppress_speech,
            vad_speech_ms=vad_speech_ms,
        )

    async def _monitor_active_session(self, stop_event: asyncio.Event) -> None:
        task = self._active_task
        if task is None:
            return
        intentional_cancel = False
        next_diagnostic = asyncio.get_running_loop().time() + 2
        pending_since: float | None = None
        pending_expired = False
        initial_window_reported = False
        while not task.done():
            if stop_event.is_set():
                intentional_cancel = True
                await self._controller.close()
                break
            playback_activity = self._runtime.last_playback_activity
            if playback_activity is not None:
                if not self._traced_playback:
                    self._traced_playback = True
                    self._trace.event("playback.first_audible_frame")
                self._controller.note_output_activity(playback_activity)
            clock_now = asyncio.get_running_loop().time()
            callbacks = self._active_callbacks
            # First-question readiness is separate from post-reply inactivity.
            # Provider connection/transcription delay must not consume the
            # entire five-second conversation idle window.
            initial_window = (
                isinstance(callbacks, _ActivityCallbacks)
                and not callbacks.has_recognized_input
                and not self._traced_playback
                and clock_now - (callbacks.ready_at if callbacks.ready_at is not None
                                 else self._initial_request_created_at) < 10.0
            )
            if initial_window and not initial_window_reported:
                initial_window_reported = True
                self._trace.event("input.initial_window", reason="10_seconds_after_ready")
                _LOGGER.info("voice first question window: up to 10 seconds after readiness")
            pending = bool(getattr(self._active_callbacks, "has_pending_response", False))
            if pending:
                if pending_since is None:
                    pending_since = clock_now
                    _LOGGER.info("voice waiting for delegated response; idle close suspended")
                    self._trace.event("idle.pending", pending=True)
                if clock_now - pending_since < 180:
                    self._controller.note_input_activity()
                else:
                    if not pending_expired:
                        _LOGGER.warning("voice pending-response limit reached; deferred result remains available")
                        pending_expired = True
                    pending = False
            elif pending_since is not None:
                pending_since = None
                pending_expired = False
                self._controller.note_input_activity()
                _LOGGER.info("voice delegated response released; idle window resumed")
                self._trace.event("idle.pending", pending=False)
            playback_is_active = (
                playback_activity is not None
                and (datetime.now(timezone.utc) - playback_activity).total_seconds()
                <= 0.5
            )
            processing_response = bool(getattr(callbacks, "processing_response", False))
            response_audio_active = bool(
                getattr(callbacks, "response_audio_active", False)
            )
            half_duplex = bool(getattr(callbacks, "half_duplex", False))
            speech_muted = half_duplex and (self._awaiting_response or playback_is_active)
            if playback_is_active:
                self._set_indicator_audio_level(self._runtime.playback_level)
                if not bool(getattr(callbacks, "processing_response", False)) and not bool(
                    getattr(callbacks, "response_audio_active", False)
                ):
                    self._awaiting_response = False
                self._set_indicator_state(VoiceIndicatorState.SPEAKING)
            else:
                self._set_indicator_audio_level(0.0)
                if pending or self._awaiting_response:
                    self._set_indicator_state(VoiceIndicatorState.THINKING)
                else:
                    self._set_indicator_state(VoiceIndicatorState.LISTENING)
            if clock_now >= next_diagnostic:
                next_diagnostic = clock_now + 2
                capture_stats, playback_stats = await self._runtime.stats()
                levels = self._input_levels.snapshot()
                self._trace.event("audio.input_levels", **levels)
                _LOGGER.info("voice microphone cumulative levels=%s", levels)
                self._trace.event("audio.queues", frames=self._active_frames,
                    capture_queued_ms=capture_stats.queued_milliseconds,
                    capture_dropped=capture_stats.dropped_overflow_frames,
                    playback_queued_ms=playback_stats.queued_milliseconds,
                    playback_dropped=playback_stats.dropped_overflow_frames,
                    muted=speech_muted,
                    reason=("assistant_processing_or_playback" if speech_muted else "none"))
                _LOGGER.info(
                    "voice activity phase=%s frames=%d idle_seconds=%s playback_at=%s pending=%s half_duplex=%s awaiting_response=%s processing_response=%s response_audio_active=%s upstream_mic_muted=%s",
                    self.phase,
                    self._active_frames,
                    self._controller.seconds_since_activity,
                    playback_activity,
                    pending,
                    half_duplex,
                    self._awaiting_response,
                    processing_response,
                    response_audio_active,
                    speech_muted,
                )
                _LOGGER.info(
                    "voice queues capture_queued_ms=%d capture_dropped=%d playback_queued_ms=%d playback_dropped=%d",
                    capture_stats.queued_milliseconds, capture_stats.dropped_overflow_frames,
                    playback_stats.queued_milliseconds, playback_stats.dropped_overflow_frames,
                )
            idle_seconds = self._controller.seconds_since_activity
            idle_timeout_seconds = self._controller.idle_timeout_seconds
            if (
                not initial_window
                and not pending
                and not self._awaiting_response
                and not processing_response
                and not response_audio_active
                and await self._controller.close_if_idle()
            ):
                session_id = self._active_session_id
                self._trace.event(
                    "voice.session_closed",
                    reason="idle_timeout",
                    elapsed_ms=round((idle_seconds or 0) * 1000),
                    idle_timeout_seconds=idle_timeout_seconds,
                )
                _LOGGER.info(
                    "voice idle timeout reached; returning to wake listener session_id=%s idle_seconds=%s timeout_seconds=%s",
                    session_id, idle_seconds, idle_timeout_seconds,
                )
                if self._session_end_acknowledgement is not None:
                    try:
                        await self._session_end_acknowledgement()
                    except Exception as error:
                        _LOGGER.warning("session end acknowledgement failed (%s)", type(error).__name__)
                intentional_cancel = True
                break
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=(
                        min(self._idle_check_interval, 0.04)
                        if self._indicator_audio_level_callback is not None
                        else self._idle_check_interval
                    ),
                )
            except TimeoutError:
                pass

        if task.cancelled():
            if not intentional_cancel and not stop_event.is_set():
                if self._controller.phase != "idle":
                    self._clear_active_session(task, status="failed")
                    raise RuntimeError("voice session task was cancelled unexpectedly")
            outcome = "cancelled" if stop_event.is_set() else "closed"
        else:
            error = task.exception()
            if error is not None:
                if isinstance(error, DeviceTransportError):
                    _LOGGER.warning("voice relay failed: %s", error)
                if self._controller.phase == "active":
                    await self._controller.close()
                self._trace.update(status="failed", error_type=type(error).__name__,
                                   error_category="voice_session_failed")
                self._clear_active_session(task, status="failed")
                raise error
            if self._controller.phase == "active":
                await self._controller.close()
            outcome = "closed"
        self._clear_active_session(task, status=outcome)

    async def _play_active_wake_chime(self) -> None:
        assert self._wake_acknowledgement is not None
        try:
            await self._wake_acknowledgement()
        except Exception as error:
            _LOGGER.warning("active wake acknowledgement failed (%s)", type(error).__name__)

    async def _play_processing_chime(self) -> None:
        """Play a short local cue while Muse prepares its spoken reply."""

        session_id = self._active_session_id
        if session_id is None:
            return
        try:
            frames = timer_chime_frames(
                self._runtime.audio_format,
                amplitude=0.035,
                # Keep the 200 ms cue within the bounded ten-frame speaker queue.
                notes=((659, 3), (784, 3), (988, 4)),
            )
            for frame in frames:
                if not await self._runtime.queue_playback_frame(session_id, frame):
                    return
        except Exception as error:
            _LOGGER.warning("processing chime failed (%s)", type(error).__name__)

    def _mark_response_complete(self) -> None:
        self._awaiting_response = False

    async def _finish_wake_acknowledgement(self) -> None:
        task, self._ack_task = self._ack_task, None
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _close_idle_capture(self) -> None:
        capture, self._idle_capture = self._idle_capture, None
        if capture is not None:
            await capture.close()

    async def _close_session_capture(self) -> None:
        capture, self._session_capture = self._session_capture, None
        if capture is not None:
            await capture.close()

    def _clear_active_session(
        self, task: asyncio.Task[SessionResponse] | None, *, status: str = "closed"
    ) -> None:
        if task is not None and self._active_task is not task:
            return
        self._active_task = None
        self._trace.end(status=status, frames=self._active_frames)
        self._trace = Observation()
        self._active_session_id = None
        self._active_callbacks = None
        self._preroll_cutoff = None
        self._awaiting_response = False
        self._turn_vad_speech_frames = 0
        self._set_indicator_state(VoiceIndicatorState.OFF)
        self._reset_speech_detector()

    def _reset_speech_detector(self) -> None:
        reset = getattr(self._speech_detector, "reset", None)
        if callable(reset):
            reset()
