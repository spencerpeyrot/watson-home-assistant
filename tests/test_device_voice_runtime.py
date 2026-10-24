from __future__ import annotations

import asyncio
import unittest
from collections import deque
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

from contracts.models import (
    AudioFrameMetadata,
    JobAccepted,
    JobEvent,
    SessionResponse,
    VoiceSessionEvent,
)
from device.audio_runtime import PiAudioRuntime
from device.transport import DeviceTransportError
from device.voice_runtime import LocalVoiceRuntime, _ActivityCallbacks
from voice.job_orchestrator import VoiceJobCoordinator
from voice.audio_levels import PcmLevelMeter


SESSION_ID = UUID("00000000-0000-4000-8000-000000000401")
CONVERSATION_ID = UUID("00000000-0000-4000-8000-000000000402")
FRAME_BYTES = 960


class FakeCapture:
    def __init__(self, frames: list[bytes], label: str, order: list[str]) -> None:
        self.frames = deque(frames)
        self.label = label
        self.order = order
        self.started = False
        self.close_count = 0

    async def start(self) -> None:
        self.started = True
        self.order.append(f"{self.label}:start")

    async def read_frame(self) -> bytes:
        if self.frames:
            return self.frames.popleft()
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def close(self) -> None:
        self.close_count += 1
        self.order.append(f"{self.label}:close")


class FakePlayback:
    async def start(self) -> None:
        return None

    async def write_frame(self, _payload: bytes) -> None:
        return None

    async def abort(self) -> None:
        return None


class FakeSocket:
    def __init__(self, order: list[str], messages=()) -> None:
        self.order = order
        self.messages = deque(messages)
        self.sent: list[bytes] = []
        self.contexts = []
        self.turn_ends = []
        self.active_audio_sent = asyncio.Event()
        self.context_sent = asyncio.Event()
        self.closed = False

    async def wait_until_ready(self):
        self.order.append("socket:ready")
        return VoiceSessionEvent(type="session_ready", session_id=SESSION_ID)

    async def send_audio_frame(self, frame, _buffer) -> bool:
        self.sent.append(frame.payload)
        if frame.payload == bytes([3]) * FRAME_BYTES:
            self.active_audio_sent.set()
        return True

    async def send_audio_turn_end(self, sequence: int, *, vad_speech_ms: int = 0) -> None:
        self.turn_ends.append((sequence, vad_speech_ms))

    async def receive_voice_message(self):
        if self.messages:
            return self.messages.popleft()
        await asyncio.Future()

    async def send_context_append(self, request) -> None:
        self.contexts.append(request)
        self.context_sent.set()

    async def close(self) -> None:
        self.closed = True
        self.order.append("socket:close")


class FakeSessionCallbacks:
    def __init__(self) -> None:
        self.finished_turns = 0
        self.closed = False

    async def on_event(self, _event, _socket) -> None:
        return None

    def mark_audio_finished(self) -> None:
        self.finished_turns += 1

    async def close(self) -> None:
        self.closed = True


class FakeJobs:
    def __init__(self) -> None:
        self.request = None
        self.event = None
        self.closed = False

    async def submit(self, request):
        self.request = request
        job_id = uuid4()
        self.event = JobEvent(
            device_id=request.device_id,
            job_id=job_id,
            event_id=uuid4(),
            sequence=1,
            kind="result",
            status="succeeded",
            conversation_id=request.conversation_id,
            session_id=request.session_id,
            turn_id=request.turn_id,
            delegation_id=request.delegation_id,
            trace_id=request.trace_id,
            context_revision=request.context_revision,
            summary="Done.",
            timestamp=datetime.now(timezone.utc),
        )
        return JobAccepted(
            job_id=job_id,
            status="queued",
            trace_id=request.trace_id,
            deduplicated=False,
            first_event_sequence=1,
        )

    async def events(self, _delegation_id):
        return (self.event,)

    async def close(self) -> None:
        self.closed = True


class DeviceVoiceRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_muse_processing_suspends_idle_timeout_and_plays_progress_chime(self) -> None:
        audio_runtime = PiAudioRuntime(
            capture_pcm="unused-capture",
            playback_pcm="unused-playback",
            capture_stream=FakeCapture([], "active", []),
            playback_stream=FakePlayback(),
        )
        queued_frames: list[bytes] = []

        async def queue_frame(_session_id, payload):
            queued_frames.append(payload)
            return True

        audio_runtime.queue_playback_frame = queue_frame
        runtime = LocalVoiceRuntime(
            session_client=SimpleNamespace(),
            runtime=audio_runtime,
            idle_capture_factory=lambda: FakeCapture([], "idle", []),
            gateway_url="https://gateway.example",
            device_id="cube-01",
            device_token="t" * 40,
            conversation_id=CONVERSATION_ID,
            locale="en-US",
            timezone_name="America/New_York",
            wake_detect=lambda _payload: _resolved(False),
            speech_detect=lambda _payload: False,
            interrupt_detect=lambda _payload: False,
            idle_timeout_seconds=0.01,
            idle_check_interval_seconds=0.001,
        )

        class IdleCheckController:
            phase = "active"
            seconds_since_activity = 10.0
            idle_timeout_seconds = 3.0

            def __init__(self) -> None:
                self.checks = 0
                self.restored = 0

            def note_input_activity(self) -> None:
                return None

            def note_response_activity(self, _expectation=None) -> None:
                return None

            def restore_response_deadline(self) -> None:
                self.restored += 1

            def note_output_activity(self, _at) -> None:
                return None

            async def close_if_idle(self) -> bool:
                self.checks += 1
                return False

            async def close(self) -> None:
                self.phase = "idle"

        controller = IdleCheckController()
        runtime._controller = controller
        runtime._active_session_id = SESSION_ID
        runtime._awaiting_response = True
        session_finished = asyncio.Event()

        async def session_task_body() -> None:
            await session_finished.wait()

        session_task = asyncio.create_task(session_task_body())
        runtime._active_task = session_task
        callbacks = _ActivityCallbacks(
            FakeSessionCallbacks(),
            controller,
            on_processing_started=runtime._play_processing_chime,
            on_response_completed=runtime._mark_response_complete,
        )
        callbacks.ready_at = asyncio.get_running_loop().time() - 20
        callbacks.has_recognized_input = True
        runtime._active_callbacks = callbacks
        stop_event = asyncio.Event()
        monitor = asyncio.create_task(runtime._monitor_active_session(stop_event))
        try:
            await callbacks.on_event(
                VoiceSessionEvent(type="assistant_processing", session_id=SESSION_ID),
                SimpleNamespace(),
            )
            await asyncio.sleep(0.03)
            self.assertTrue(callbacks.processing_response)
            self.assertFalse(callbacks.response_audio_active)
            self.assertTrue(runtime._awaiting_response)
            # Processing state alone must keep the follow-up idle check suspended.
            runtime._awaiting_response = False
            await asyncio.sleep(0.03)
            self.assertEqual(controller.checks, 0)
            self.assertGreater(len(queued_frames), 0)
            self.assertLessEqual(len(queued_frames), audio_runtime.playback_buffer.capacity_frames)
            self.assertTrue(all(len(frame) == audio_runtime.audio_format.frame_bytes for frame in queued_frames))
            self.assertTrue(any(any(frame) for frame in queued_frames))

            # Audio-active signal alone must also suspend idle close, even if
            # an unrelated lifecycle update has cleared the broader processing
            # flag while the speaker still drains its response.
            callbacks.processing_response = False
            await callbacks.on_event(
                VoiceSessionEvent(
                    type="assistant_response_started", session_id=SESSION_ID
                ),
                SimpleNamespace(),
            )
            self.assertTrue(callbacks.response_audio_active)
            await asyncio.sleep(0.03)
            self.assertEqual(controller.checks, 0)

            await callbacks.on_event(
                VoiceSessionEvent(
                    type="assistant_response_complete", session_id=SESSION_ID
                ),
                SimpleNamespace(),
            )
            for _ in range(50):
                if controller.checks:
                    break
                await asyncio.sleep(0.001)
            self.assertFalse(callbacks.processing_response)
            self.assertFalse(callbacks.response_audio_active)
            self.assertFalse(runtime._awaiting_response)
            self.assertGreater(controller.checks, 0)
            await callbacks.on_event(
                VoiceSessionEvent(
                    type="assistant_turn_ignored", session_id=SESSION_ID
                ),
                SimpleNamespace(),
            )
            self.assertEqual(controller.restored, 1)
        finally:
            session_finished.set()
            await asyncio.wait_for(monitor, timeout=1)

    async def test_muse_half_duplex_suppresses_vad_endpoints_during_reply_audio(self) -> None:
        detector_calls = []
        wake_calls = []
        interrupt_calls = []
        audio_runtime = PiAudioRuntime(
            capture_pcm="unused-capture",
            playback_pcm="unused-playback",
            capture_stream=FakeCapture([], "active", []),
            playback_stream=FakePlayback(),
        )
        runtime = LocalVoiceRuntime(
            session_client=SimpleNamespace(),
            runtime=audio_runtime,
            idle_capture_factory=lambda: FakeCapture([], "idle", []),
            gateway_url="https://gateway.example",
            device_id="cube-01",
            device_token="t" * 40,
            conversation_id=CONVERSATION_ID,
            locale="en-US",
            timezone_name="America/New_York",
            wake_detect=lambda _payload: _resolved(wake_calls.append(True) or False),
            speech_detect=lambda _payload: True,
            interrupt_detect=lambda _payload: interrupt_calls.append(True) or True,
            turn_end_detect=lambda _payload: detector_calls.append("endpoint") or True,
            wake_acknowledgement=lambda: _resolved(True),
        )
        suppressions = []

        class Controller:
            speech_active = False

            async def feed(self, _payload, *, captured_at, suppress_speech=False):
                suppressions.append(suppress_speech)
                self.speech_active = not suppress_speech
                if not suppress_speech:
                    detector_calls.append("speech")

            def note_input_activity(self):
                return None

            async def interrupt(self):
                interrupt_calls.append("controller")
                return False

        callbacks = FakeSessionCallbacks()
        callbacks.half_duplex = True
        runtime._controller = Controller()
        runtime._active_callbacks = callbacks
        runtime._awaiting_response = True
        runtime._input_levels = PcmLevelMeter()
        frame = SimpleNamespace(
            metadata=AudioFrameMetadata(
                session_id=SESSION_ID,
                sequence=0,
                captured_at=datetime.now(timezone.utc),
                encoding="pcm_s16le",
                sample_rate_hz=24_000,
                channels=1,
                frame_duration_ms=20,
            ),
            payload=bytes(FRAME_BYTES),
        )

        ended = await runtime._handle_active_frame(frame)

        self.assertTrue(suppressions[0])
        self.assertFalse(ended.turn_ended)
        self.assertTrue(ended.mute_audio)
        self.assertEqual(callbacks.finished_turns, 0)
        self.assertEqual(wake_calls, [])
        self.assertEqual(interrupt_calls, [])

        runtime._awaiting_response = False
        ended = await runtime._handle_active_frame(frame)
        self.assertEqual(detector_calls, ["speech", "endpoint"])
        self.assertEqual(ended.vad_speech_ms, 20)

    async def test_idle_notification_check_runs_off_the_capture_loop(self) -> None:
        order: list[str] = []
        idle_capture = FakeCapture([], "idle", order)
        audio_runtime = PiAudioRuntime(
            capture_pcm="unused-capture",
            playback_pcm="unused-playback",
            capture_stream=FakeCapture([], "active", order),
            playback_stream=FakePlayback(),
        )
        checked = asyncio.Event()
        phases: list[str] = []

        async def check_notifications() -> None:
            phases.append("idle")
            checked.set()

        runtime = LocalVoiceRuntime(
            session_client=SimpleNamespace(),
            runtime=audio_runtime,
            idle_capture_factory=lambda: idle_capture,
            gateway_url="https://gateway.example",
            device_id="cube-01",
            device_token="t" * 40,
            conversation_id=CONVERSATION_ID,
            locale="en-US",
            timezone_name="America/New_York",
            wake_detect=lambda _payload: _resolved(False),
            speech_detect=lambda _payload: False,
            interrupt_detect=lambda _payload: False,
            idle_notification_check=check_notifications,
            idle_notification_interval_seconds=0.01,
        )

        stop_event = asyncio.Event()
        running = asyncio.create_task(runtime.run(stop_event))
        try:
            await asyncio.wait_for(checked.wait(), timeout=1)
            self.assertEqual(runtime.phase, "idle")
        finally:
            stop_event.set()
        await asyncio.wait_for(running, timeout=1)

        self.assertEqual(phases, ["idle"])
        self.assertEqual(idle_capture.close_count, 1)

    async def test_wake_hands_off_preroll_then_shutdown_cleans_session(self) -> None:
        order: list[str] = []
        idle_capture = FakeCapture(
            [bytes([1]) * FRAME_BYTES, bytes([2]) * FRAME_BYTES],
            "idle",
            order,
        )
        active_capture = FakeCapture([bytes([3]) * FRAME_BYTES], "active", order)
        playback = FakePlayback()
        audio_runtime = PiAudioRuntime(
            capture_pcm="unused-capture",
            playback_pcm="unused-playback",
            capture_stream=active_capture,
            playback_stream=playback,
        )
        socket = FakeSocket(order)
        response = SessionResponse(
            session_id=SESSION_ID,
            conversation_id=CONVERSATION_ID,
            state="connecting",
            opened_at=datetime.now(timezone.utc),
            idle_close_seconds=30,
            max_session_seconds=60,
        )

        class FakeSessionClient:
            def __init__(self) -> None:
                self.created = 0
                self.closed: list[UUID] = []

            async def create_session(self, **_values):
                self.created += 1
                return response

            async def close_session(self, session_id: UUID) -> None:
                self.closed.append(session_id)

        session_client = FakeSessionClient()
        callbacks = FakeSessionCallbacks()

        async def socket_factory(*_args):
            return socket

        def callbacks_factory(created_response):
            self.assertEqual(created_response, response)
            return callbacks

        runtime = LocalVoiceRuntime(
            session_client=session_client,
            runtime=audio_runtime,
            idle_capture_factory=lambda: idle_capture,
            gateway_url="https://gateway.example",
            device_id="cube-01",
            device_token="t" * 40,
            conversation_id=CONVERSATION_ID,
            locale="en-US",
            timezone_name="America/New_York",
            wake_detect=lambda payload: _resolved(payload == bytes([2]) * FRAME_BYTES),
            speech_detect=lambda payload: payload == bytes([3]) * FRAME_BYTES,
            interrupt_detect=lambda _payload: False,
            turn_end_detect=lambda payload: payload == bytes([3]) * FRAME_BYTES,
            session_callbacks_factory=callbacks_factory,
            socket_factory=socket_factory,
        )

        stop_event = asyncio.Event()
        running = asyncio.create_task(runtime.run(stop_event))
        try:
            await asyncio.wait_for(socket.active_audio_sent.wait(), timeout=1)
        finally:
            stop_event.set()
        await asyncio.wait_for(running, timeout=1)

        self.assertEqual(session_client.created, 1)
        self.assertEqual(session_client.closed, [SESSION_ID])
        self.assertEqual(socket.sent, [
            bytes([1]) * FRAME_BYTES,
            bytes([2]) * FRAME_BYTES,
            bytes([3]) * FRAME_BYTES,
        ])
        self.assertEqual(idle_capture.close_count, 1)
        self.assertTrue(socket.closed)
        self.assertEqual(callbacks.finished_turns, 1)
        self.assertTrue(callbacks.closed)
        self.assertIsNone(audio_runtime.session_id)
        self.assertLess(order.index("idle:close"), order.index("active:start"))

    async def test_local_turn_end_commits_voice_job_and_returns_result(self) -> None:
        order: list[str] = []
        idle_capture = FakeCapture([bytes([2]) * FRAME_BYTES], "idle", order)
        active_capture = FakeCapture([bytes([3]) * FRAME_BYTES], "active", order)
        audio_runtime = PiAudioRuntime(
            capture_pcm="unused-capture",
            playback_pcm="unused-playback",
            capture_stream=active_capture,
            playback_stream=FakePlayback(),
        )
        socket = FakeSocket(
            order,
            messages=(
                VoiceSessionEvent(
                    type="transcript_delta",
                    session_id=SESSION_ID,
                    direction="user",
                    delta="Do the task",
                    start_ms=0,
                    end_ms=900,
                ),
                VoiceSessionEvent(
                    type="delegation_created",
                    session_id=SESSION_ID,
                    delegation_id="delegation-voice-1",
                    target="client",
                    offset_ms=900,
                ),
            ),
        )
        response = SessionResponse(
            session_id=SESSION_ID,
            conversation_id=CONVERSATION_ID,
            state="connecting",
            opened_at=datetime.now(timezone.utc),
            idle_close_seconds=30,
            max_session_seconds=60,
        )
        jobs = FakeJobs()

        class FakeSessionClient:
            def __init__(self) -> None:
                self.closed: list[UUID] = []

            async def create_session(self, **_values):
                return response

            async def close_session(self, session_id: UUID) -> None:
                self.closed.append(session_id)

        session_client = FakeSessionClient()

        async def socket_factory(*_args):
            return socket

        def callbacks_factory(created_response):
            self.assertEqual(created_response, response)
            return VoiceJobCoordinator(
                jobs=jobs,  # type: ignore[arg-type]
                device_id="cube-01",
                conversation_id=created_response.conversation_id,
                session_id=created_response.session_id,
                locale="en-US",
                timezone_name="America/New_York",
                transcript_quiet_seconds=0.01,
                poll_interval_seconds=0.01,
            )

        runtime = LocalVoiceRuntime(
            session_client=session_client,
            runtime=audio_runtime,
            idle_capture_factory=lambda: idle_capture,
            gateway_url="https://gateway.example",
            device_id="cube-01",
            device_token="t" * 40,
            conversation_id=CONVERSATION_ID,
            locale="en-US",
            timezone_name="America/New_York",
            wake_detect=lambda _payload: _resolved(True),
            speech_detect=lambda payload: payload == bytes([3]) * FRAME_BYTES,
            interrupt_detect=lambda _payload: False,
            turn_end_detect=lambda payload: payload == bytes([3]) * FRAME_BYTES,
            session_callbacks_factory=callbacks_factory,
            socket_factory=socket_factory,
        )

        stop_event = asyncio.Event()
        running = asyncio.create_task(runtime.run(stop_event))
        try:
            await asyncio.wait_for(socket.context_sent.wait(), timeout=2)
        finally:
            stop_event.set()
        await asyncio.wait_for(running, timeout=1)

        self.assertEqual(jobs.request.request_text, "Do the task")
        self.assertEqual(jobs.request.delegation_id, "delegation-voice-1")
        self.assertEqual(len(socket.contexts), 1)
        self.assertEqual(socket.contexts[0].content, "Done.")
        self.assertTrue(jobs.closed)
        self.assertEqual(session_client.closed, [SESSION_ID])

    async def test_transient_gateway_failure_reopens_idle_capture_and_retries(self) -> None:
        order: list[str] = []
        wake_frame = bytes([2]) * FRAME_BYTES
        first_idle = FakeCapture([wake_frame], "idle-first", order)
        second_idle = FakeCapture([wake_frame], "idle-second", order)
        idle_captures = [first_idle, second_idle]
        active_capture = FakeCapture([bytes([3]) * FRAME_BYTES], "active", order)
        audio_runtime = PiAudioRuntime(
            capture_pcm="unused-capture",
            playback_pcm="unused-playback",
            capture_stream=active_capture,
            playback_stream=FakePlayback(),
        )
        socket = FakeSocket(order)
        response = SessionResponse(
            session_id=SESSION_ID,
            conversation_id=CONVERSATION_ID,
            state="connecting",
            opened_at=datetime.now(timezone.utc),
            idle_close_seconds=30,
            max_session_seconds=60,
        )

        class FlakySessionClient:
            def __init__(self) -> None:
                self.attempts = 0
                self.closed: list[UUID] = []

            async def create_session(self, **_values):
                self.attempts += 1
                if self.attempts == 1:
                    raise DeviceTransportError("private transport detail")
                return response

            async def close_session(self, session_id: UUID) -> None:
                self.closed.append(session_id)

        session_client = FlakySessionClient()
        retry_notices = []

        async def socket_factory(*_args):
            return socket

        runtime = LocalVoiceRuntime(
            session_client=session_client,
            runtime=audio_runtime,
            idle_capture_factory=lambda: idle_captures.pop(0),
            gateway_url="https://gateway.example",
            device_id="cube-01",
            device_token="t" * 40,
            conversation_id=CONVERSATION_ID,
            locale="en-US",
            timezone_name="America/New_York",
            wake_detect=lambda _payload: _resolved(True),
            speech_detect=lambda payload: payload == bytes([3]) * FRAME_BYTES,
            interrupt_detect=lambda _payload: False,
            socket_factory=socket_factory,
            on_retry=retry_notices.append,
        )

        stop_event = asyncio.Event()
        running = asyncio.create_task(runtime.run(stop_event))
        try:
            await asyncio.wait_for(socket.active_audio_sent.wait(), timeout=2)
        finally:
            stop_event.set()
        await asyncio.wait_for(running, timeout=1)

        self.assertEqual(session_client.attempts, 2)
        self.assertEqual(session_client.closed, [SESSION_ID])
        self.assertEqual(len(retry_notices), 1)
        self.assertEqual(retry_notices[0].error_type, "DeviceTransportError")
        self.assertNotIn("private transport detail", repr(retry_notices[0]))
        self.assertEqual(idle_captures, [])
        self.assertEqual((first_idle.close_count, second_idle.close_count), (1, 1))
        self.assertTrue(socket.closed)
        self.assertIsNone(audio_runtime.session_id)


async def _resolved(value: bool) -> bool:
    return value


if __name__ == "__main__":
    unittest.main()
