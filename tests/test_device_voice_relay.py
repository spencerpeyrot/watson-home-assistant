from __future__ import annotations

import asyncio
import unittest
from collections import deque
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID

from contracts.models import AudioOutputFrameMetadata, SessionResponse, VoiceSessionEvent
from device.audio_buffer import AudioFrame, BufferedAudioFrame
from device.voice_relay import CaptureFrameDecision, run_device_voice_relay
from device.voice_session import run_gateway_voice_session


SESSION_ID = UUID("00000000-0000-4000-8000-000000000301")


class FakeSocket:
    def __init__(self, messages):
        self.messages = deque(messages)
        self.ready = False
        self.closed = False
        self.sent = []
        self.turn_ends = []

    async def wait_until_ready(self):
        self.ready = True

    async def send_audio_frame(self, frame, _buffer):
        self.sent.append(frame)
        return True

    async def send_audio_turn_end(self, sequence, *, vad_speech_ms=0):
        self.turn_ends.append((sequence, vad_speech_ms))

    async def receive_voice_message(self):
        return self.messages.popleft()

    async def close(self):
        self.closed = True


class BlockingSocket(FakeSocket):
    async def receive_voice_message(self):
        await asyncio.Future()


class FakeRuntime:
    def __init__(self, capture_frame):
        self.audio_format = SimpleNamespace(
            sample_rate_hz=24_000,
            channels=1,
            frame_duration_ms=20,
            frame_bytes=960,
        )
        self.capture_buffer = object()
        self.started = False
        self.stopped = False
        self.preroll = ()
        self.capture_frame = capture_frame
        self.playback: list[bytes] = []
        self.drain_calls: list[bool] = []

    async def start(self, session_id, *, capture_preroll=()):
        self.started = session_id == SESSION_ID
        self.preroll = tuple(capture_preroll)

    async def next_capture_frame(self):
        frame, self.capture_frame = self.capture_frame, None
        if frame is not None:
            return frame
        await asyncio.Future()

    async def queue_playback_frame(self, session_id, payload):
        if session_id != SESSION_ID:
            return False
        self.playback.append(payload)
        return True

    async def drain_playback(self, *, reopen=False):
        self.drain_calls.append(reopen)

    async def wait_failure(self):
        await asyncio.Future()

    async def stop(self, session_id=None):
        self.stopped = session_id == SESSION_ID
        return self.stopped


class DeviceVoiceRelayTests(unittest.IsolatedAsyncioTestCase):
    async def test_relay_sends_preroll_reframes_playback_and_cleans_up(self):
        from contracts.models import AudioFrameMetadata

        capture_payload = bytes(960)
        frame = AudioFrame(
            metadata=AudioFrameMetadata(
                session_id=SESSION_ID,
                sequence=0,
                captured_at=datetime.now(timezone.utc),
                encoding="pcm_s16le",
                sample_rate_hz=24_000,
                channels=1,
                frame_duration_ms=20,
            ),
            payload=capture_payload,
            generation=1,
        )
        output = bytes(range(256)) * 3 + bytes(range(232))
        messages = [
            VoiceSessionEvent(
                type="transcript_delta",
                session_id=SESSION_ID,
                direction="assistant",
                delta="Ready.",
                start_ms=0,
                end_ms=120,
            ),
            (
                AudioOutputFrameMetadata(
                    session_id=SESSION_ID,
                    sequence=0,
                    encoding="pcm_s16le",
                    payload_bytes=len(output),
                ),
                output,
            ),
            VoiceSessionEvent(type="session_closed", session_id=SESSION_ID),
        ]
        socket = FakeSocket(messages)
        runtime = FakeRuntime(frame)
        preroll = (BufferedAudioFrame(capture_payload, datetime.now(timezone.utc)),)
        events = []

        await run_device_voice_relay(
            session_id=SESSION_ID,
            socket=socket,  # type: ignore[arg-type]
            runtime=runtime,  # type: ignore[arg-type]
            capture_preroll=preroll,
            on_event=lambda event, _socket: _append(events, event),
        )

        self.assertTrue(socket.ready)
        self.assertEqual(socket.sent, [frame])
        self.assertEqual(runtime.preroll, preroll)
        self.assertEqual(runtime.playback[0], output[:960])
        self.assertEqual(runtime.playback[1], output[960:] + bytes(920))
        self.assertEqual(runtime.drain_calls, [False])
        self.assertEqual([event.type for event in events], ["transcript_delta", "session_closed"])
        self.assertTrue(runtime.stopped)
        self.assertTrue(socket.closed)

    async def test_vad_turn_end_follows_final_audio_frame(self):
        from contracts.models import AudioFrameMetadata

        payload = bytes(960)
        frame = AudioFrame(
            metadata=AudioFrameMetadata(
                session_id=SESSION_ID,
                sequence=0,
                captured_at=datetime.now(timezone.utc),
                encoding="pcm_s16le",
                sample_rate_hz=24_000,
                channels=1,
                frame_duration_ms=20,
            ),
            payload=payload,
            generation=1,
        )
        socket = FakeSocket([VoiceSessionEvent(type="session_closed", session_id=SESSION_ID)])
        runtime = FakeRuntime(frame)

        async def endpoint(_frame):
            return CaptureFrameDecision(turn_ended=True, vad_speech_ms=80)

        await run_device_voice_relay(
            session_id=SESSION_ID,
            socket=socket,  # type: ignore[arg-type]
            runtime=runtime,  # type: ignore[arg-type]
            on_capture_frame=endpoint,
        )

        self.assertEqual(socket.sent, [frame])
        self.assertEqual(socket.turn_ends, [(0, 80)])

    async def test_half_duplex_reply_sends_silence_upstream_without_mutating_capture(self):
        from contracts.models import AudioFrameMetadata

        payload = bytes([7]) * 960
        frame = AudioFrame(
            metadata=AudioFrameMetadata(
                session_id=SESSION_ID,
                sequence=0,
                captured_at=datetime.now(timezone.utc),
                encoding="pcm_s16le",
                sample_rate_hz=24_000,
                channels=1,
                frame_duration_ms=20,
            ),
            payload=payload,
            generation=1,
        )
        socket = FakeSocket([VoiceSessionEvent(type="session_closed", session_id=SESSION_ID)])
        runtime = FakeRuntime(frame)

        async def during_reply(_frame):
            return CaptureFrameDecision(mute_audio=True)

        await run_device_voice_relay(
            session_id=SESSION_ID,
            socket=socket,  # type: ignore[arg-type]
            runtime=runtime,  # type: ignore[arg-type]
            on_capture_frame=during_reply,
        )

        sent = socket.sent[0]
        self.assertEqual(sent.payload, bytes(960))
        self.assertEqual(sent.metadata.sequence, frame.metadata.sequence)
        self.assertEqual(frame.payload, payload)
        self.assertEqual(socket.turn_ends, [])

    async def test_response_complete_drains_speaker_before_runtime_unmutes(self):
        socket = FakeSocket(
            [
                VoiceSessionEvent(
                    type="assistant_response_complete", session_id=SESSION_ID
                ),
                VoiceSessionEvent(type="session_closed", session_id=SESSION_ID),
            ]
        )
        runtime = FakeRuntime(None)
        observed: list[tuple[bool, ...]] = []

        async def on_event(event, _socket):
            if event.type == "assistant_response_complete":
                observed.append(tuple(runtime.drain_calls))

        await run_device_voice_relay(
            session_id=SESSION_ID,
            socket=socket,  # type: ignore[arg-type]
            runtime=runtime,  # type: ignore[arg-type]
            on_event=on_event,
        )

        self.assertEqual(observed, [(True,)])
        self.assertEqual(runtime.drain_calls, [True, False])

    async def test_gateway_voice_session_composes_http_lifecycle_and_socket(self):
        conversation_id = UUID("00000000-0000-4000-8000-000000000302")
        response = SessionResponse(
            session_id=SESSION_ID,
            conversation_id=conversation_id,
            state="connecting",
            opened_at=datetime.now(timezone.utc),
            idle_close_seconds=30,
            max_session_seconds=60,
        )

        class FakeSessionClient:
            def __init__(self):
                self.created = None
                self.closed = []

            async def create_session(self, **values):
                self.created = values
                return response

            async def close_session(self, session_id):
                self.closed.append(session_id)

        session_client = FakeSessionClient()
        socket = FakeSocket([VoiceSessionEvent(type="session_closed", session_id=SESSION_ID)])
        runtime = FakeRuntime(None)
        opened = []
        handled = []

        async def socket_factory(gateway_url, device_id, device_token, session_id):
            opened.append((gateway_url, device_id, device_token, session_id))
            return socket

        def event_handler_factory(created_session):
            self.assertEqual(created_session, response)

            async def handle(event, event_socket):
                handled.append((event, event_socket))

            return handle

        result = await run_gateway_voice_session(
            session_client=session_client,  # type: ignore[arg-type]
            runtime=runtime,  # type: ignore[arg-type]
            gateway_url="https://gateway.example",
            device_id="cube-01",
            device_token="t" * 40,
            conversation_id=conversation_id,
            locale="en-US",
            timezone="America/New_York",
            on_event_factory=event_handler_factory,
            socket_factory=socket_factory,  # type: ignore[arg-type]
        )

        self.assertEqual(result, response)
        self.assertEqual(session_client.created["conversation_id"], conversation_id)
        self.assertEqual(session_client.closed, [SESSION_ID])
        self.assertEqual(opened[0][3], SESSION_ID)
        self.assertEqual(handled[0][0].type, "session_closed")
        self.assertIs(handled[0][1], socket)
        self.assertTrue(runtime.stopped)
        self.assertTrue(socket.closed)

    async def test_gateway_session_cap_stops_relay_and_closes_durable_session(self):
        conversation_id = UUID("00000000-0000-4000-8000-000000000303")
        response = SessionResponse(
            session_id=SESSION_ID,
            conversation_id=conversation_id,
            state="connecting",
            opened_at=datetime.now(timezone.utc),
            idle_close_seconds=30,
            max_session_seconds=1,
        )

        class FakeSessionClient:
            def __init__(self):
                self.closed = []

            async def create_session(self, **_values):
                return response

            async def close_session(self, session_id):
                self.closed.append(session_id)

        session_client = FakeSessionClient()
        socket = BlockingSocket([])
        runtime = FakeRuntime(None)

        async def socket_factory(*_args):
            return socket

        result = await run_gateway_voice_session(
            session_client=session_client,  # type: ignore[arg-type]
            runtime=runtime,  # type: ignore[arg-type]
            gateway_url="https://gateway.example",
            device_id="cube-01",
            device_token="t" * 40,
            conversation_id=conversation_id,
            locale="en-US",
            timezone="America/New_York",
            socket_factory=socket_factory,  # type: ignore[arg-type]
        )

        self.assertEqual(result.max_session_seconds, 1)
        self.assertEqual(session_client.closed, [SESSION_ID])
        self.assertTrue(socket.closed)
        self.assertTrue(runtime.stopped)


async def _append(items, value):
    items.append(value)


if __name__ == "__main__":
    unittest.main()
