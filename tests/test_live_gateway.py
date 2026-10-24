from __future__ import annotations

import base64
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID

from fastapi import Request, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from contracts.models import SessionResponse
from service.durable_api import AuthenticatedPrincipal, create_durable_app
from service.live_provider import LiveSessionClose


OWNER_ID = UUID("00000000-0000-4000-8000-000000000201")
SESSION_ID = UUID("00000000-0000-4000-8000-000000000202")
CONVERSATION_ID = UUID("00000000-0000-4000-8000-000000000203")
DEVICE_ID = "cube-01"


class FakeStore:
    def __init__(self) -> None:
        self.session = SessionResponse(
            session_id=SESSION_ID,
            conversation_id=CONVERSATION_ID,
            state="connecting",
            opened_at=datetime.now(timezone.utc),
            idle_close_seconds=30,
            max_session_seconds=120,
        )
        self.ready = False
        self.start_claimed = False
        self.closed: list[tuple[UUID, str]] = []
        self.usage: list[dict[str, object]] = []

    async def get_connecting_session_for_device(self, *, owner_id, device_id, session_id):
        if (owner_id, device_id, session_id) != (OWNER_ID, DEVICE_ID, SESSION_ID):
            return None
        if self.session.state != "connecting":
            return None
        return self.session

    async def claim_session_start(self, *, owner_id, device_id, session_id):
        self.start_claimed = (owner_id, device_id, session_id) == (
            OWNER_ID,
            DEVICE_ID,
            SESSION_ID,
        ) and self.session.state == "connecting"
        if self.start_claimed:
            self.session = self.session.model_copy(update={"state": "starting"})
        return self.start_claimed

    async def mark_session_ready(self, *, owner_id, device_id, session_id):
        self.ready = (
            (owner_id, device_id, session_id) == (OWNER_ID, DEVICE_ID, SESSION_ID)
            and self.session.state == "starting"
        )
        if self.ready:
            self.session = self.session.model_copy(update={"state": "ready"})
        return self.ready

    async def close_session(self, *, owner_id, session_id, device_id=None):
        self.closed.append((session_id, device_id))
        self.session = self.session.model_copy(update={"state": "closed"})
        return True

    async def record_usage_once(self, **values):
        self.usage.append(values)
        return values


class FakeProvider:
    def __init__(self) -> None:
        import asyncio

        self.events = asyncio.Queue()
        self.audio: list[bytes] = []
        self.context: list[tuple[str, str, str | None]] = []
        self.received_events: list[str] = []
        self.started = False
        self.closed = False
        self.aborted = False

    async def start(self) -> str:
        self.started = True
        return "live-session-test"

    async def append_audio(self, pcm_s16le: bytes) -> None:
        self.audio.append(pcm_s16le)
        self.events.put_nowait(
            {
                "type": "session.input_transcript.delta",
                "delta": "Turn on the",
                "start_ms": 0,
                "end_ms": 500,
            }
        )
        self.events.put_nowait(
            {
                "type": "session.delegation.created",
                "offset_ms": 500,
                "delegation": {
                    "id": "item_test_123",
                    "target": "client",
                },
            }
        )
        self.events.put_nowait(
            {
                "type": "session.output_audio.delta",
                "delta": base64.b64encode(bytes(20_000)).decode("ascii"),
            }
        )

    async def append_context(self, content, *, kind, delegation_id):
        self.context.append((content, kind, delegation_id))
        self.events.put_nowait({"type": "session.closed"})
        return "context-event-test"

    async def receive_event(self):
        event = await self.events.get()
        self.received_events.append(event["type"])
        return event

    async def close(self) -> LiveSessionClose:
        self.closed = True
        return LiveSessionClose(
            provider_session_id="live-session-test",
            connected_seconds=Decimal("12.5"),
            final_event={
                "type": "session.closed",
                "usage": {"seconds": 12.5},
            },
        )

    async def abort(self) -> None:
        self.aborted = True


class TurnProvider(FakeProvider):
    half_duplex = True

    def __init__(self) -> None:
        super().__init__()
        self.turn_ids = []
        self.vad_speech_ms = []

    async def append_audio(self, pcm_s16le: bytes) -> None:
        self.audio.append(pcm_s16le)

    async def finish_turn(self, turn_id, *, vad_speech_ms=0) -> None:
        self.turn_ids.append(turn_id)
        self.vad_speech_ms.append(vad_speech_ms)
        self.events.put_nowait({"type": "session.processing"})
        self.events.put_nowait({"type": "session.output_transcript.delta", "delta": "Muse reply."})
        self.events.put_nowait({
            "type": "session.output_audio.delta",
            "delta": base64.b64encode(b"\x01\x00" * 480).decode("ascii"),
        })
        self.events.put_nowait({
            "type": "session.output_audio.done",
            "reply_expectation": "open_answer",
        })

    async def close(self):
        self.closed = True
        return None


async def authenticate_http(_request: Request) -> AuthenticatedPrincipal:
    return AuthenticatedPrincipal(role="device", owner_id=OWNER_ID, device_id=DEVICE_ID)


async def authenticate_websocket(websocket: WebSocket) -> AuthenticatedPrincipal:
    if websocket.headers.get("authorization") == "Bearer device-test":
        return AuthenticatedPrincipal(role="device", owner_id=OWNER_ID, device_id=DEVICE_ID)
    return AuthenticatedPrincipal(role="owner", owner_id=OWNER_ID)


class LiveGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeStore()
        self.providers: list[FakeProvider] = []

        def make_provider(_principal, _session):
            provider = FakeProvider()
            self.providers.append(provider)
            return provider

        self.client = TestClient(
            create_durable_app(
                store=self.store,  # type: ignore[arg-type]
                authenticate=authenticate_http,
                authenticate_websocket=authenticate_websocket,
                live_session_factory=make_provider,
            )
        )

    def tearDown(self) -> None:
        self.client.close()

    def test_authenticated_relay_maps_provider_events_and_final_usage(self) -> None:
        url = f"/v1/devices/{DEVICE_ID}/connect?session_id={SESSION_ID}"
        with self.client.websocket_connect(
            url, headers={"Authorization": "Bearer device-test"}
        ) as websocket:
            ready = websocket.receive_json()
            self.assertEqual(ready["type"], "session_ready")

            websocket.send_json(
                {
                    "type": "audio_frame_metadata",
                    "payload": {
                        "schema_version": 1,
                        "session_id": str(SESSION_ID),
                        "sequence": 0,
                        "captured_at": "2026-10-07T12:00:00Z",
                        "encoding": "pcm_s16le",
                        "sample_rate_hz": 24000,
                        "channels": 1,
                        "frame_duration_ms": 20,
                    },
                }
            )
            websocket.send_bytes(bytes(960))

            transcript = websocket.receive_json()
            delegation = websocket.receive_json()
            websocket.send_json(
                {
                    "type": "voice_context_append",
                    "payload": {
                        "schema_version": 1,
                        "session_id": str(SESSION_ID),
                        "delegation_id": "item_test_123",
                        "kind": "commentary",
                        "content": "The light is on.",
                    },
                }
            )
            output_frames = []
            try:
                response_started = websocket.receive_json()
                self.assertEqual(
                    response_started["type"], "assistant_response_started"
                )
                for _ in range(3):
                    audio_metadata = websocket.receive_json()
                    audio_payload = websocket.receive_bytes()
                    output_frames.append((audio_metadata, audio_payload))
            except WebSocketDisconnect as error:
                self.fail(
                    f"relay closed early: code={error.code}, context={self.providers[0].context}, "
                    f"events={self.providers[0].received_events}"
                )
            closed = websocket.receive_json()

        self.assertEqual(transcript["type"], "transcript_delta")
        self.assertEqual(transcript["direction"], "user")
        self.assertEqual(delegation["type"], "delegation_created")
        self.assertEqual(delegation["delegation_id"], "item_test_123")
        self.assertEqual(
            [len(payload) for _metadata, payload in output_frames], [9600, 9600, 800]
        )
        self.assertEqual(
            [metadata["payload"]["sequence"] for metadata, _payload in output_frames],
            [0, 1, 2],
        )
        self.assertTrue(all(metadata["type"] == "audio_output_metadata" for metadata, _ in output_frames))
        self.assertTrue(
            all(metadata["payload"]["payload_bytes"] == len(payload) for metadata, payload in output_frames)
        )
        self.assertEqual(closed["type"], "session_closed")
        self.assertTrue(self.store.ready)
        self.assertTrue(self.store.start_claimed)
        self.assertEqual(self.store.closed, [(SESSION_ID, DEVICE_ID)])
        self.assertEqual(len(self.store.usage), 1)
        self.assertEqual(self.store.usage[0]["provider"], "openai-live")
        self.assertEqual(self.store.usage[0]["measurement"].connected_seconds, Decimal("12.5"))
        self.assertEqual(self.providers[0].audio, [bytes(960)])
        self.assertEqual(
            self.providers[0].context,
            [("The light is on.", "commentary", "item_test_123")],
        )
        self.assertTrue(self.providers[0].closed)
        self.assertFalse(self.providers[0].aborted)

    def test_non_device_credential_cannot_start_provider_session(self) -> None:
        url = f"/v1/devices/{DEVICE_ID}/connect?session_id={SESSION_ID}"
        with self.assertRaises(WebSocketDisconnect) as raised:
            with self.client.websocket_connect(url, headers={"Authorization": "Bearer owner-test"}):
                pass
        self.assertEqual(raised.exception.code, 1008)
        self.assertEqual(self.providers, [])

    def test_expired_session_cap_rejects_before_provider_start(self) -> None:
        self.store.session = self.store.session.model_copy(
            update={"opened_at": datetime.now(timezone.utc) - timedelta(seconds=121)}
        )
        url = f"/v1/devices/{DEVICE_ID}/connect?session_id={SESSION_ID}"

        with self.assertRaises(WebSocketDisconnect) as raised:
            with self.client.websocket_connect(
                url, headers={"Authorization": "Bearer device-test"}
            ):
                pass

        self.assertEqual(raised.exception.code, 1008)
        self.assertEqual(self.store.closed, [(SESSION_ID, DEVICE_ID)])
        self.assertEqual(self.providers, [])

    def test_session_cap_closes_provider_and_persists_confirmed_usage(self) -> None:
        self.store.session = self.store.session.model_copy(
            update={"max_session_seconds": 1}
        )
        url = f"/v1/devices/{DEVICE_ID}/connect?session_id={SESSION_ID}"

        with self.client.websocket_connect(
            url, headers={"Authorization": "Bearer device-test"}
        ) as websocket:
            self.assertEqual(websocket.receive_json()["type"], "session_ready")
            closed = websocket.receive_json()

        self.assertEqual(closed["type"], "session_closed")
        self.assertTrue(self.providers[0].closed)
        self.assertEqual(len(self.store.usage), 1)
        self.assertEqual(
            self.store.usage[0]["measurement"].connected_seconds, Decimal("12.5")
        )
        self.assertEqual(self.store.closed, [(SESSION_ID, DEVICE_ID)])

    def test_device_cannot_claim_another_or_unknown_session(self) -> None:
        unknown_session_id = UUID("00000000-0000-4000-8000-000000000299")
        url = f"/v1/devices/{DEVICE_ID}/connect?session_id={unknown_session_id}"
        with self.assertRaises(WebSocketDisconnect) as raised:
            with self.client.websocket_connect(
                url, headers={"Authorization": "Bearer device-test"}
            ):
                pass
        self.assertEqual(raised.exception.code, 1008)
        self.assertEqual(self.providers, [])

    def test_input_sequence_gap_closes_without_forwarding_audio(self) -> None:
        url = f"/v1/devices/{DEVICE_ID}/connect?session_id={SESSION_ID}"
        with self.client.websocket_connect(
            url, headers={"Authorization": "Bearer device-test"}
        ) as websocket:
            self.assertEqual(websocket.receive_json()["type"], "session_ready")
            websocket.send_json(
                {
                    "type": "audio_frame_metadata",
                    "payload": {
                        "schema_version": 1,
                        "session_id": str(SESSION_ID),
                        "sequence": 1,
                        "captured_at": "2026-10-07T12:00:00Z",
                        "encoding": "pcm_s16le",
                        "sample_rate_hz": 24000,
                        "channels": 1,
                        "frame_duration_ms": 20,
                    },
                }
            )
            error = websocket.receive_json()
            closed = websocket.receive_json()

        self.assertEqual(error["type"], "session_error")
        self.assertEqual(error["code"], "INVALID_REQUEST")
        self.assertEqual(closed["type"], "session_closed")
        self.assertEqual(self.providers[0].audio, [])

    def test_context_append_requires_a_known_client_delegation(self) -> None:
        url = f"/v1/devices/{DEVICE_ID}/connect?session_id={SESSION_ID}"
        with self.client.websocket_connect(
            url, headers={"Authorization": "Bearer device-test"}
        ) as websocket:
            self.assertEqual(websocket.receive_json()["type"], "session_ready")
            websocket.send_json(
                {
                    "type": "voice_context_append",
                    "payload": {
                        "schema_version": 1,
                        "session_id": str(SESSION_ID),
                        "delegation_id": "forged_delegation",
                        "kind": "commentary",
                        "content": "Do this unrelated action.",
                    },
                }
            )
            error = websocket.receive_json()
            websocket.receive_json()

        self.assertEqual(error["type"], "session_error")
        self.assertEqual(error["code"], "INVALID_REQUEST")
        self.assertEqual(self.providers[0].context, [])

    def test_output_audio_is_divided_into_bounded_pcm_frames(self) -> None:
        from service.live_gateway import OUTPUT_FRAME_CHUNK_BYTES

        self.assertEqual(OUTPUT_FRAME_CHUNK_BYTES, 9600)
        self.assertTrue(OUTPUT_FRAME_CHUNK_BYTES % 2 == 0)

    def test_audio_turn_end_calls_provider_and_relays_streamed_reply(self) -> None:
        providers: list[TurnProvider] = []

        def factory(_principal, _session):
            provider = TurnProvider()
            providers.append(provider)
            return provider

        client = TestClient(
            create_durable_app(
                store=self.store,  # type: ignore[arg-type]
                authenticate=authenticate_http,
                authenticate_websocket=authenticate_websocket,
                live_session_factory=factory,
            )
        )
        turn_id = UUID("00000000-0000-4000-8000-000000000204")
        try:
            with client.websocket_connect(
                f"/v1/devices/{DEVICE_ID}/connect?session_id={SESSION_ID}",
                headers={"Authorization": "Bearer device-test"},
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
                        "captured_at": "2026-10-07T12:00:00Z",
                        "encoding": "pcm_s16le",
                        "sample_rate_hz": 24000,
                        "channels": 1,
                        "frame_duration_ms": 20,
                    },
                })
                websocket.send_bytes(bytes(960))
                websocket.send_json({
                    "type": "voice_audio_turn_end",
                    "schema_version": 1,
                    "session_id": str(SESSION_ID),
                    "turn_id": str(turn_id),
                    "last_audio_sequence": 0,
                            "vad_speech_ms": 460,
                })
                processing = websocket.receive_json()
                transcript = websocket.receive_json()
                response_started = websocket.receive_json()
                audio_metadata = websocket.receive_json()
                audio_message = websocket.receive()
                complete = websocket.receive_json()
                self.assertEqual(processing["type"], "assistant_processing")
                self.assertEqual(response_started["type"], "assistant_response_started")
                self.assertIn("bytes", audio_message, audio_message)
                audio = audio_message["bytes"]
                self.assertEqual(transcript["type"], "transcript_delta")
                self.assertEqual(transcript["direction"], "assistant")
                self.assertEqual(transcript["delta"], "Muse reply.")
                self.assertEqual(audio_metadata["type"], "audio_output_metadata")
                self.assertEqual(audio_metadata["payload"]["sequence"], 0)
                self.assertEqual(audio, b"\x01\x00" * 480)
                self.assertEqual(complete["type"], "assistant_response_complete")
                self.assertEqual(complete["reply_expectation"], "open_answer")
        finally:
            client.close()

        self.assertEqual(providers[0].audio, [bytes(960)])
        self.assertEqual(providers[0].turn_ids, [turn_id])
        self.assertEqual(providers[0].vad_speech_ms, [460])
        self.assertTrue(providers[0].closed)
        self.assertEqual(self.store.usage, [])


if __name__ == "__main__":
    unittest.main()
