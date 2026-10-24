from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import UUID

from fastapi import Request, WebSocket
from fastapi.testclient import TestClient

from contracts.models import SessionResponse
from service.durable_api import AuthenticatedPrincipal
from service.muse_provider import CUBE_VOICE_RESPONSE_INSTRUCTION
from service.server import create_app


OWNER_ID = UUID("00000000-0000-4000-8000-000000000701")
SESSION_ID = UUID("00000000-0000-4000-8000-000000000702")
CONVERSATION_ID = UUID("00000000-0000-4000-8000-000000000703")
DEVICE_ID = "cube-muse-01"
DEVICE_TOKEN = "d" * 48
MUSE_TOKEN = "mgst_" + ("A" * 42) + "A"


class FakeStore:
    def __init__(self, _database_url):
        self.session = SessionResponse(
            session_id=SESSION_ID,
            conversation_id=CONVERSATION_ID,
            state="connecting",
            opened_at=datetime.now(timezone.utc),
            idle_close_seconds=30,
            max_session_seconds=60,
        )
        self.closed = False
        self.ready = False

    async def open(self):
        return None

    async def close(self):
        return None

    async def get_connecting_session_for_device(self, **_values):
        return self.session if self.session.state == "connecting" else None

    async def claim_session_start(self, **_values):
        if self.session.state != "connecting":
            return False
        self.session = self.session.model_copy(update={"state": "starting"})
        return True

    async def mark_session_ready(self, **_values):
        self.ready = self.session.state == "starting"
        if self.ready:
            self.session = self.session.model_copy(update={"state": "ready"})
        return self.ready

    async def close_session(self, *, session_id, **_values):
        self.closed = session_id == SESSION_ID
        self.session = self.session.model_copy(update={"state": "closed"})
        return self.closed


class FakeMuseLink:
    def __init__(self):
        self.chat_subscribed = asyncio.Event()
        self.events: asyncio.Queue[dict] = asyncio.Queue()
        self.sent_text = None

    async def subscribe_chat(self, _session_id=None):
        self.chat_subscribed.set()
        try:
            while True:
                yield await self.events.get()
        finally:
            self.chat_subscribed.clear()

    async def send_chat(self, message, session_id=None, *, output_modality="text"):
        self.sent_text = message
        assert session_id is not None
        assert output_modality == "text"
        chat_id, user_id = session_id, "user-message-1"
        base = {
            "session_id": chat_id,
            "message_id": "assistant-message-1",
            "reply_to_message_id": user_id,
            "role": "assistant",
        }
        await self.events.put({"type": "event", "event": "delta.message_start", "payload": base})
        await self.events.put({
            "type": "event", "event": "delta.text_append", "payload": {**base, "text": "Hello from the integrated V2 path."}
        })
        await self.events.put({
            "type": "event", "event": "delta.message_done", "payload": {**base, "display_text": "Hello from the integrated V2 path."}
        })
        await self.events.put({
            "type": "event", "event": "task.status", "payload": {
                "session_id": chat_id,
                "status": "completed",
                "task_id": "task-1",
                "reply_to_message_id": user_id,
            }
        })
        return {
            "ok": True,
            "status": 200,
            "response": {
                "result": {
                    "message_id": user_id,
                    "session_id": chat_id,
                    "is_thread": True,
                }
            },
        }


class FakeMuseHost:
    def __init__(self, _token):
        self.link = FakeMuseLink()
        self.stopped = asyncio.Event()

    async def run(self, stop):
        try:
            await stop.wait()
        finally:
            # App lifespan signals stop, then cancels background tasks.
            self.stopped.set()

    async def wait_for_link(self, _timeout_seconds=20):
        return self.link


class FakeElevenLabsSocket:
    def __init__(self):
        self.empty_text = asyncio.Event()
        self.sent = []
        self.read_audio = False
        self.closed = False

    async def send(self, raw):
        message = json.loads(raw)
        self.sent.append(message)
        if message.get("text") == "":
            self.empty_text.set()

    async def recv(self):
        await self.empty_text.wait()
        if not self.read_audio:
            self.read_audio = True
            return json.dumps({"audio": base64.b64encode(b"\x01\x00" * 480).decode(), "is_final": False})
        return json.dumps({"is_final": True})

    async def close(self):
        self.closed = True


class FakeElevenLabsSTTSocket:
    def __init__(self):
        self.committed = asyncio.Event()
        self.sent = []
        self.closed = False

    async def send(self, raw):
        message = json.loads(raw)
        self.sent.append(message)
        if message.get("commit") is True:
            self.committed.set()

    async def recv(self):
        if not self.sent:
            return json.dumps({"message_type": "session_started", "session_id": "stt-1"})
        await self.committed.wait()
        return json.dumps({
            "message_type": "committed_transcript",
            "text": "What is the weather today?",
        })

    async def close(self):
        self.closed = True


async def authenticate_http(_request: Request):
    return AuthenticatedPrincipal(role="device", owner_id=OWNER_ID, device_id=DEVICE_ID)


async def authenticate_websocket(websocket: WebSocket):
    if websocket.headers.get("authorization") == f"Bearer {DEVICE_TOKEN}":
        return AuthenticatedPrincipal(role="device", owner_id=OWNER_ID, device_id=DEVICE_ID)
    return AuthenticatedPrincipal(role="owner", owner_id=OWNER_ID)


class MuseStackIntegrationTests(unittest.TestCase):
    def test_muse_gateway_turn_streams_through_elevenlabs_to_cube_playback(self):
        with tempfile.TemporaryDirectory() as temporary:
            host_instances = []
            tts_sockets = []
            stt_sockets = []

            def host_factory(token):
                host = FakeMuseHost(token)
                host_instances.append(host)
                return host

            async def tts_connector(_url, **_options):
                socket = FakeElevenLabsSocket()
                tts_sockets.append(socket)
                return socket

            async def stt_connector(_url, **_options):
                socket = FakeElevenLabsSTTSocket()
                stt_sockets.append(socket)
                return socket

            settings = {
                "DATABASE_URL": "postgresql://test.invalid/cube",
                "OPENAI_API_KEY": "backend-test-key",
                "CUBE_OWNER_ID": str(OWNER_ID),
                "CUBE_DEVICE_ID": DEVICE_ID,
                "CUBE_OWNER_TOKEN": "o" * 48,
                "CUBE_DEVICE_TOKEN": DEVICE_TOKEN,
                "CUBE_OPERATOR_TOKEN": "x" * 48,
                "CUBE_VOICE_STACK": "muse",
                "MUSE_SDK_TOKEN": MUSE_TOKEN,
                "TYPESAFE_API_KEY": "typesafe-test-key",
                "MUSEGADGET_STATE_DIR": temporary,
                "ELEVENLABS_API_KEY": "eleven-test-key",
                "ELEVENLABS_VOICE_ID": "voice-123",
            }
            class FakeDecisions:
                async def decide(self, *, input_text, questions):
                    self.input_text = input_text
                    self.questions = questions
                    return {"route": ("conversation", 0.99)}

            with patch("service.server.JevDecisionsClient", return_value=FakeDecisions()):
                app = create_app(
                    settings,
                    store_factory=lambda url: FakeStore(url),  # type: ignore[arg-type]
                    muse_host_factory=host_factory,  # type: ignore[arg-type]
                    elevenlabs_connector=tts_connector,
                    elevenlabs_stt_connector=stt_connector,
                )
            try:
                with TestClient(app) as client:
                    with client.websocket_connect(
                        f"/v1/devices/{DEVICE_ID}/connect?session_id={SESSION_ID}",
                        headers={"Authorization": f"Bearer {DEVICE_TOKEN}"},
                    ) as websocket:
                        ready = websocket.receive_json()
                        self.assertEqual(ready["type"], "session_ready")
                        self.assertTrue(ready["half_duplex"])
                        websocket.send_json({
                            "type": "audio_frame_metadata",
                            "payload": {
                                "schema_version": 1,
                                "session_id": str(SESSION_ID),
                                "sequence": 0,
                                "captured_at": "2026-10-14T12:00:00Z",
                                "encoding": "pcm_s16le",
                                "sample_rate_hz": 24_000,
                                "channels": 1,
                                "frame_duration_ms": 20,
                            },
                        })
                        websocket.send_bytes(b"\x01\x00" * 480)
                        websocket.send_json({
                            "type": "voice_audio_turn_end",
                            "schema_version": 1,
                            "session_id": str(SESSION_ID),
                            "turn_id": "00000000-0000-4000-8000-000000000704",
                            "last_audio_sequence": 0,
                        })
                        processing = websocket.receive_json()
                        input_transcript = websocket.receive_json()
                        assistant_transcript = websocket.receive_json()
                        response_started = websocket.receive_json()
                        output_metadata = websocket.receive_json()
                        output_audio = websocket.receive_bytes()
                        complete = websocket.receive_json()

                        self.assertEqual(processing["type"], "assistant_processing")
                        self.assertEqual(input_transcript["type"], "transcript_delta")
                        self.assertEqual(input_transcript["direction"], "user")
                        self.assertEqual(input_transcript["delta"], "What is the weather today?")
                        self.assertEqual(assistant_transcript["type"], "transcript_delta")
                        self.assertEqual(assistant_transcript["direction"], "assistant")
                        self.assertEqual(assistant_transcript["delta"], "Hello from the integrated V2 path.")
                        self.assertEqual(
                            response_started["type"], "assistant_response_started"
                        )
                        self.assertEqual(output_metadata["type"], "audio_output_metadata")
                        self.assertEqual(output_metadata["payload"]["sample_rate_hz"], 24_000)
                        self.assertEqual(output_audio, b"\x01\x00" * 480)
                        self.assertEqual(complete["type"], "assistant_response_complete")
                        self.assertEqual(
                            host_instances[0].link.sent_text,
                            f"{CUBE_VOICE_RESPONSE_INSTRUCTION}\n\n"
                            "Spoken request:\nWhat is the weather today?",
                        )
                        self.assertTrue(stt_sockets[0].closed)
                        self.assertTrue(tts_sockets[0].closed)
            finally:
                client.close()

            self.assertTrue(host_instances[0].stopped.is_set())
            self.assertEqual(
                host_instances[0].link.sent_text,
                f"{CUBE_VOICE_RESPONSE_INSTRUCTION}\n\n"
                "Spoken request:\nWhat is the weather today?",
            )
            self.assertTrue(tts_sockets[0].closed)


if __name__ == "__main__":
    unittest.main()
