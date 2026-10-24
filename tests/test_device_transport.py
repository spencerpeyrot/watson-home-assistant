from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

from websockets.asyncio.server import serve

from device.audio_buffer import AudioFrameBuffer
from device.transport import DeviceSocket, _websocket_close_detail, device_websocket_url


SESSION = UUID("00000000-0000-4000-8000-000000000020")


class FakeConnection:
    def __init__(self) -> None:
        self.sent: list[str | bytes] = []
        self.closed = False
        self.received: list[str | bytes] = ["ack"]

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        return self.received.pop(0)

    async def close(self) -> None:
        self.closed = True


class DeviceTransportTests(unittest.IsolatedAsyncioTestCase):
    def test_gateway_url_converts_scheme_and_quotes_device_id(self) -> None:
        self.assertEqual(
            device_websocket_url("https://gateway.example/base/", "cube 01"),
            "wss://gateway.example/base/v1/devices/cube%2001/connect",
        )
        self.assertEqual(
            device_websocket_url("http://127.0.0.1:8080", "cube-01"),
            "ws://127.0.0.1:8080/v1/devices/cube-01/connect",
        )
        self.assertEqual(
            device_websocket_url("https://gateway.example", "cube-01", SESSION),
            f"wss://gateway.example/v1/devices/cube-01/connect?session_id={SESSION}",
        )

    def test_gateway_url_rejects_cleartext_remote_and_embedded_credentials(self) -> None:
        with self.assertRaises(ValueError):
            device_websocket_url("http://gateway.example", "cube-01")
        with self.assertRaises(ValueError):
            device_websocket_url("https://owner:secret@gateway.example", "cube-01")
        with self.assertRaises(ValueError):
            device_websocket_url("https://gateway.example?token=secret", "cube-01")
        with self.assertRaises(ValueError):
            device_websocket_url("https://gateway.example", "../admin")

    async def test_send_audio_frame_sends_contract_then_binary_and_checks_session(self) -> None:
        buffer = AudioFrameBuffer()
        await buffer.start_session(SESSION)
        payload = bytes(buffer.frame_bytes)
        await buffer.put(SESSION, payload)
        frame = await buffer.get()
        self.assertIsNotNone(frame)
        assert frame is not None
        connection = FakeConnection()
        socket = DeviceSocket(connection)  # type: ignore[arg-type]

        sent = await socket.send_audio_frame(frame, buffer)
        received = await socket.receive()
        await socket.close()

        self.assertTrue(sent)
        self.assertEqual(received, "ack")
        self.assertEqual(len(connection.sent), 2)
        envelope = json.loads(connection.sent[0])
        self.assertEqual(envelope["type"], "audio_frame_metadata")
        self.assertEqual(envelope["payload"]["session_id"], str(SESSION))
        self.assertEqual(connection.sent[1], payload)
        self.assertTrue(connection.closed)

    async def test_send_rejects_frame_after_session_change(self) -> None:
        buffer = AudioFrameBuffer()
        await buffer.start_session(SESSION)
        await buffer.put(SESSION, bytes(buffer.frame_bytes))
        frame = await buffer.get()
        self.assertIsNotNone(frame)
        assert frame is not None
        await buffer.close_session(SESSION)
        connection = FakeConnection()
        socket = DeviceSocket(connection)  # type: ignore[arg-type]

        sent = await socket.send_audio_frame(frame, buffer)

        self.assertFalse(sent)
        self.assertEqual(connection.sent, [])

    async def test_authenticated_socket_open_sends_metadata_then_binary(self) -> None:
        observed: dict[str, object] = {}

        async def handler(connection) -> None:
            observed["path"] = connection.request.path
            observed["authorization"] = connection.request.headers.get("Authorization")
            observed["metadata"] = json.loads(await connection.recv())
            observed["audio"] = await connection.recv()
            await connection.send("audio_frame_accepted")

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            socket = await DeviceSocket.open(
                f"http://127.0.0.1:{port}", "cube-01", "t" * 40, SESSION
            )
            buffer = AudioFrameBuffer()
            await buffer.start_session(SESSION)
            payload = bytes(buffer.frame_bytes)
            await buffer.put(SESSION, payload)
            frame = await buffer.get()
            self.assertIsNotNone(frame)
            assert frame is not None

            sent = await socket.send_audio_frame(frame, buffer)
            acknowledged = await socket.receive()
            await socket.close()

        self.assertTrue(sent)
        self.assertEqual(acknowledged, "audio_frame_accepted")
        self.assertEqual(
            observed["path"], f"/v1/devices/cube-01/connect?session_id={SESSION}"
        )
        self.assertEqual(observed["authorization"], f"Bearer {'t' * 40}")
        metadata = observed["metadata"]
        assert isinstance(metadata, dict)
        self.assertEqual(metadata["type"], "audio_frame_metadata")
        self.assertEqual(observed["audio"], payload)

    async def test_voice_socket_relies_on_continuous_pcm_instead_of_ping_timeout(self) -> None:
        connection = FakeConnection()
        with patch("device.transport.connect", new_callable=AsyncMock) as connector:
            connector.return_value = connection
            socket = await DeviceSocket.open(
                "http://127.0.0.1:8080", "cube-01", "t" * 40, SESSION
            )
            await socket.close()

        assert connector.await_args is not None
        self.assertIsNone(connector.await_args.kwargs["ping_interval"])
        self.assertNotIn("ping_timeout", connector.await_args.kwargs)

    def test_websocket_close_diagnostic_includes_code_and_bounded_reason(self) -> None:
        error = RuntimeError("transport closed")
        error.sent = SimpleNamespace(code=1011, reason="keepalive ping timeout\n")
        error.rcvd = SimpleNamespace(code=1001, reason="server restart")

        self.assertEqual(
            _websocket_close_detail(error),
            "; sent_code=1011 sent_reason='keepalive ping timeout'; "
            "received_code=1001 received_reason='server restart'",
        )

    async def test_voice_receiver_validates_ready_and_playback_frame_pair(self) -> None:
        connection = FakeConnection()
        connection.received = [
            json.dumps({"schema_version": 1, "type": "session_ready", "session_id": str(SESSION)}),
            json.dumps(
                {
                    "type": "audio_output_metadata",
                    "payload": {
                        "schema_version": 1,
                        "session_id": str(SESSION),
                        "sequence": 0,
                        "encoding": "pcm_s16le",
                        "sample_rate_hz": 24000,
                        "channels": 1,
                        "payload_bytes": 4,
                    },
                }
            ),
            b"\x00\x00\x01\x00",
        ]
        socket = DeviceSocket(connection, session_id=SESSION)  # type: ignore[arg-type]

        await socket.wait_until_ready()
        frame = await socket.receive_voice_message()

        self.assertEqual(frame[0].sequence, 0)  # type: ignore[index]
        self.assertEqual(frame[1], b"\x00\x00\x01\x00")  # type: ignore[index]

    async def test_voice_receiver_rejects_wrong_output_byte_count(self) -> None:
        from device.transport import DeviceTransportError

        connection = FakeConnection()
        connection.received = [
            json.dumps(
                {
                    "type": "audio_output_metadata",
                    "payload": {
                        "schema_version": 1,
                        "session_id": str(SESSION),
                        "sequence": 0,
                        "encoding": "pcm_s16le",
                        "sample_rate_hz": 24000,
                        "channels": 1,
                        "payload_bytes": 4,
                    },
                }
            ),
            b"\x00\x00",
        ]
        socket = DeviceSocket(connection, session_id=SESSION)  # type: ignore[arg-type]

        with self.assertRaises(DeviceTransportError):
            await socket.receive_voice_message()

    async def test_send_context_append_is_session_bound(self) -> None:
        from contracts.models import VoiceContextAppend

        connection = FakeConnection()
        socket = DeviceSocket(connection, session_id=SESSION)  # type: ignore[arg-type]
        request = VoiceContextAppend(
            session_id=SESSION,
            delegation_id="item_test_123",
            kind="commentary",
            content="The verified result is ready.",
        )

        await socket.send_context_append(request)

        envelope = json.loads(connection.sent[0])
        self.assertEqual(envelope["type"], "voice_context_append")
        self.assertEqual(envelope["payload"]["delegation_id"], "item_test_123")
        self.assertEqual(envelope["payload"]["kind"], "commentary")
        with self.assertRaises(ValueError):
            await socket.send_context_append(
                request.model_copy(update={"session_id": UUID("00000000-0000-4000-8000-000000000021")})
            )


if __name__ == "__main__":
    unittest.main()
