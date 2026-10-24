from __future__ import annotations

import asyncio
import base64
import json
import unittest

from service.elevenlabs_tts import ElevenLabsError, ElevenLabsTTS


class FakeWebSocket:
    def __init__(self, *, odd_audio: bool = False, final_field: str = "is_final") -> None:
        self.sent: list[dict] = []
        self.closed = False
        self.empty_text_sent = asyncio.Event()
        self.odd_audio = odd_audio
        self.final_field = final_field
        self.received_audio = False

    async def send(self, raw: str) -> None:
        message = json.loads(raw)
        self.sent.append(message)
        if message.get("text") == "":
            self.empty_text_sent.set()

    async def recv(self) -> str:
        await self.empty_text_sent.wait()
        if not self.received_audio:
            self.received_audio = True
            audio = b"\x00\x01\x02" if self.odd_audio else b"\x00\x01" * 480
            return json.dumps({"audio": base64.b64encode(audio).decode("ascii"), "is_final": False})
        return json.dumps({self.final_field: True})

    async def close(self) -> None:
        self.closed = True


class ElevenLabsTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_tts_requests_matching_pcm_and_yields_audio(self):
        socket = FakeWebSocket()
        captured = {}

        async def connector(url, **options):
            captured["url"] = url
            captured["options"] = options
            return socket

        async def texts():
            yield "Hello "
            yield "there."

        tts = ElevenLabsTTS(
            "secret-test-key",
            "voice id/1",
            connector=connector,
        )
        audio = [chunk async for chunk in tts.synthesize(texts())]

        self.assertEqual(audio, [b"\x00\x01" * 480])
        self.assertIn("pcm_24000", captured["url"])
        self.assertIn("eleven_flash_v2_5", captured["url"])
        self.assertIn("voice%20id%2F1", captured["url"])
        self.assertNotIn("additional_headers", captured["options"])
        self.assertEqual([item["text"] for item in socket.sent], [" ", "Hello ", "there.", ""])
        self.assertEqual(socket.sent[0]["xi_api_key"], "secret-test-key")
        self.assertEqual(socket.sent[0]["generation_config"]["chunk_length_schedule"], [50, 120, 160, 290])
        self.assertEqual([item.get("flush", False) for item in socket.sent], [False, False, True, False])
        self.assertTrue(socket.closed)

    async def test_rejects_non_pcm16_output_and_closes_connection(self):
        socket = FakeWebSocket(odd_audio=True)

        async def connector(_url, **_options):
            return socket

        async def texts():
            yield "Hi."

        tts = ElevenLabsTTS("secret", "voice", connector=connector)
        with self.assertRaisesRegex(ElevenLabsError, "PCM16"):
            _ = [chunk async for chunk in tts.synthesize(texts())]
        self.assertTrue(socket.closed)

    async def test_accepts_camel_case_provider_final_event(self):
        socket = FakeWebSocket(final_field="isFinal")

        async def connector(_url, **_options):
            return socket

        async def texts():
            yield "Hi."

        tts = ElevenLabsTTS("secret", "voice", connector=connector)
        audio = [chunk async for chunk in tts.synthesize(texts())]

        self.assertEqual(audio, [b"\x00\x01" * 480])
        self.assertTrue(socket.closed)


if __name__ == "__main__":
    unittest.main()
