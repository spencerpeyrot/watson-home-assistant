from __future__ import annotations

import base64
import asyncio
import json
import unittest

from service.elevenlabs_stt import ElevenLabsSTT, ElevenLabsSTTError


class FakeWebSocket:
    def __init__(self, *, response: dict | None = None) -> None:
        self.sent: list[dict] = []
        self.closed = False
        self.response = response or {
            "message_type": "committed_transcript",
            "text": "Turn on the desk lamp.",
        }

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    async def recv(self) -> str:
        if len(self.sent) == 0:
            return json.dumps({"message_type": "session_started", "session_id": "stt-1"})
        return json.dumps(self.response)

    async def close(self) -> None:
        self.closed = True


class ElevenLabsSTTTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_bounded_pcm_chunks_and_waits_for_committed_transcript(self):
        socket = FakeWebSocket()
        captured = {}

        async def connector(url, **options):
            captured["url"] = url
            captured["options"] = options
            return socket

        pcm = bytes(range(256)) * 40
        recognizer = ElevenLabsSTT("secret-test-key", connector=connector)
        transcript = await recognizer.transcribe(pcm)

        self.assertEqual(transcript, "Turn on the desk lamp.")
        self.assertIn("scribe_v2_realtime", captured["url"])
        self.assertIn("pcm_24000", captured["url"])
        self.assertIn("commit_strategy=manual", captured["url"])
        self.assertEqual(
            captured["options"]["additional_headers"],
            {"xi-api-key": "secret-test-key"},
        )
        self.assertEqual(len(socket.sent), 3)
        self.assertEqual(socket.sent[0]["message_type"], "input_audio_chunk")
        self.assertEqual(
            base64.b64decode(socket.sent[0]["audio_base_64"]), pcm[:4_800]
        )
        self.assertFalse(socket.sent[0].get("commit", False))
        self.assertEqual(
            base64.b64decode(socket.sent[1]["audio_base_64"]), pcm[4_800:9_600]
        )
        self.assertFalse(socket.sent[1].get("commit", False))
        self.assertEqual(base64.b64decode(socket.sent[2]["audio_base_64"]), pcm[9_600:])
        self.assertTrue(socket.sent[2]["commit"])
        self.assertTrue(socket.closed)

    async def test_rejects_invalid_pcm_before_connecting(self):
        called = False

        async def connector(*_args, **_kwargs):
            nonlocal called
            called = True
            return FakeWebSocket()

        recognizer = ElevenLabsSTT("secret", connector=connector)
        with self.assertRaisesRegex(ElevenLabsSTTError, "PCM16"):
            await recognizer.transcribe(b"\x00")
        self.assertFalse(called)

    async def test_provider_rejection_is_sanitized_and_socket_closes(self):
        socket = FakeWebSocket(
            response={"message_type": "rate_limited", "error": "private detail"}
        )

        async def connector(*_args, **_kwargs):
            return socket

        recognizer = ElevenLabsSTT("secret", connector=connector)
        with self.assertRaisesRegex(ElevenLabsSTTError, "rejected transcription") as caught:
            await recognizer.transcribe(b"\x00\x00")
        self.assertNotIn("private detail", str(caught.exception))
        self.assertTrue(socket.closed)
        self.assertEqual(caught.exception.observability["error_category"], "rate_limit")
        self.assertEqual(caught.exception.observability["failure_stage"], "provider_response")

    async def test_committed_transcript_timeout_has_stage_metadata(self):
        class HangingSocket(FakeWebSocket):
            async def recv(self):
                if not self.sent:
                    return json.dumps({"message_type": "session_started"})
                await asyncio.Event().wait()

        socket = HangingSocket()

        async def connector(*_args, **_kwargs):
            return socket

        recognizer = ElevenLabsSTT("secret", connector=connector)
        recognizer.TRANSCRIPT_TIMEOUT_SECONDS = 0.001
        with self.assertRaises(ElevenLabsSTTError) as caught:
            await recognizer.transcribe(b"\x00\x00")
        self.assertEqual(caught.exception.observability["failure_stage"], "await_transcript")
        self.assertTrue(socket.closed)


if __name__ == "__main__":
    unittest.main()
