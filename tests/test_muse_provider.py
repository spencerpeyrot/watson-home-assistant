from __future__ import annotations

import asyncio
import stat
import tempfile
import unittest
from pathlib import Path
from uuid import UUID, uuid4

from service.muse_provider import (
    CUBE_VOICE_RESPONSE_INSTRUCTION,
    MAX_TURN_PCM_BYTES,
    MuseConversationRegistry,
    MuseElevenProvider,
    muse_eleven_provider_factory,
)
from service.live_provider import LiveProviderError
from service.voice_commands import VoiceTurnDecision


def muse_event(name: str, payload: dict) -> dict:
    return {"type": "event", "event": name, "payload": payload}


class FakeMuseLink:
    def __init__(self) -> None:
        self.chat_subscribed = asyncio.Event()
        self.subscribe_delay_seconds = 0.0
        self.events: asyncio.Queue[dict] = asyncio.Queue()
        self.turns: list[tuple[str, str | None, str]] = []
        self.subscriptions: list[str | None] = []
        self.operations: list[tuple[str, str | None]] = []

    async def subscribe_chat(self, session_id: str | None = None):
        self.subscriptions.append(session_id)
        self.operations.append(("subscribe", session_id))
        if self.subscribe_delay_seconds:
            await asyncio.sleep(self.subscribe_delay_seconds)
        self.chat_subscribed.set()
        try:
            while True:
                yield await self.events.get()
        finally:
            self.chat_subscribed.clear()

    async def send_chat(
        self,
        message: str,
        session_id: str | None = None,
        *,
        output_modality: str = "text",
    ) -> dict:
        self.turns.append((message, session_id, output_modality))
        self.operations.append(("send", session_id))
        user_id = f"user-{len(self.turns)}"
        chat_id = session_id or "muse-chat-1"
        if len(self.turns) > 1:
            # Old chat events can replay on Muse reconnect. MuseTurn replay
            # scope must filter the completed reply from the preceding turn.
            await self.events.put(
                muse_event(
                    "message.assistant",
                    {
                        "session_id": chat_id,
                        "message_id": "assistant-1",
                        "reply_to_message_id": "user-1",
                        "role": "assistant",
                        "display_text": "stale answer",
                    },
                )
            )
        answer = "Hello from Muse." if len(self.turns) == 1 else "Second answer."
        assistant_id = f"assistant-{len(self.turns)}"
        base = {
            "session_id": chat_id,
            "message_id": assistant_id,
            "reply_to_message_id": user_id,
            "role": "assistant",
        }
        await self.events.put(muse_event("delta.message_start", base))
        await self.events.put(
            muse_event("delta.text_append", {**base, "text": answer})
        )
        await self.events.put(
            muse_event("delta.message_done", {**base, "display_text": answer})
        )
        await self.events.put(
            muse_event(
                "task.status",
                {"session_id": chat_id, "status": "completed", "task_id": f"task-{len(self.turns)}", "reply_to_message_id": user_id},
            )
        )
        return {
            "ok": True,
            "status": 200,
            "response": {"result": {
                "message_id": user_id,
                "session_id": chat_id,
                "is_thread": True,
            }},
        }


class FakeHost:
    def __init__(self, link: FakeMuseLink) -> None:
        self.link = link

    async def wait_for_link(self, _timeout_seconds: float = 20):
        return self.link


class FakeSpeech:
    def __init__(self) -> None:
        self.text = ""
        self.calls = 0

    async def synthesize(self, text_chunks):
        self.calls += 1
        async for text in text_chunks:
            self.text += text
            yield b"\x01\x00" * 480


class FailingSpeech:
    async def synthesize(self, text_chunks):
        async for _text in text_chunks:
            pass
        yield b"\x01\x00"
        raise RuntimeError("private provider detail")


class FakeRecognizer:
    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.text = "Hello Muse."

    async def transcribe(self, pcm_s16le: bytes) -> str:
        self.audio.append(pcm_s16le)
        return self.text


class MuseProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_followup_routing_receives_previous_accepted_exchange(self):
        link = FakeMuseLink()
        recognizer = FakeRecognizer()
        received = []

        async def route(text, _turn_id, previous_user, previous_reply):
            received.append((text, previous_user, previous_reply))
            return VoiceTurnDecision("conversation")

        provider = MuseElevenProvider(
            FakeHost(link), FakeSpeech(), recognizer, internal_turn_handler=route
        )
        await provider.start()
        for transcript in ("What's the weather?", "Tomorrow too?"):
            recognizer.text = transcript
            await provider.append_audio(b"\x01\x00" * 960)
            await provider.finish_turn(uuid4())
            await self._collect_turn(provider)

        self.assertEqual(
            received,
            [
                ("What's the weather?", None, None),
                ("Tomorrow too?", "What's the weather?", "Hello from Muse."),
            ],
        )
        await provider.close()

    async def test_ignored_turn_never_reaches_muse_or_closes_session(self):
        link = FakeMuseLink()

        async def ignore(_text, _turn_id, _previous_user, _previous_reply):
            return VoiceTurnDecision("ignore")

        provider = MuseElevenProvider(
            FakeHost(link), FakeSpeech(), FakeRecognizer(), internal_turn_handler=ignore
        )
        await provider.start()
        await provider.append_audio(b"\x01\x00" * 960)
        await provider.finish_turn(uuid4())

        events = await self._collect_turn(provider)

        self.assertEqual(link.turns, [])
        self.assertNotIn("session.closed", [e["type"] for e in events])
        self.assertIn("session.input_turn.ignored", [e["type"] for e in events])
        self.assertIn("session.input_transcript.ignored", [e["type"] for e in events])
        await provider.close()

    async def test_reply_expectation_classifies_completed_muse_reply(self):
        link = FakeMuseLink()
        calls = []

        class TurnHandler:
            async def __call__(self, *_args):
                return VoiceTurnDecision("conversation")

            async def classify_reply_expectation(self, *, user_turn, reply):
                calls.append((user_turn, reply))
                return "open_answer"

        provider = MuseElevenProvider(
            FakeHost(link), FakeSpeech(), FakeRecognizer(),
            internal_turn_handler=TurnHandler(),
        )
        await provider.start()
        await provider.append_audio(b"\x01\x00" * 960)
        await provider.finish_turn(uuid4())
        events = await self._collect_turn(provider)

        self.assertEqual(calls, [("Hello Muse.", "Hello from Muse.")])
        self.assertEqual(events[-1]["reply_expectation"], "open_answer")
        await provider.close()

    async def test_turn_waits_for_chat_subscription_past_startup_timeout(self):
        link = FakeMuseLink()
        link.subscribe_delay_seconds = 0.05
        provider = MuseElevenProvider(
            FakeHost(link),
            FakeSpeech(),
            FakeRecognizer(),
            startup_timeout_seconds=0.01,
        )
        await provider.start()
        await provider.append_audio(b"\x01\x00" * 960)
        await provider.finish_turn(uuid4())

        events = await self._collect_turn(provider)

        self.assertEqual(events[0]["type"], "session.processing")
        self.assertEqual(events[-1]["type"], "session.output_audio.done")
        await provider.close()

    async def test_tts_starts_after_muse_first_text_arrives(self):
        link = FakeMuseLink()
        link.subscribe_delay_seconds = 0.05
        speech = FakeSpeech()
        provider = MuseElevenProvider(FakeHost(link), speech, FakeRecognizer())
        await provider.start()
        await provider.append_audio(b"\x01\x00" * 960)
        await provider.finish_turn(uuid4())
        await asyncio.sleep(0.01)
        self.assertEqual(speech.calls, 0)
        await self._collect_turn(provider)
        self.assertEqual(speech.calls, 1)
        await provider.close()

    async def test_jev_acknowledgement_is_spoken_before_muse_reply(self):
        link = FakeMuseLink()

        async def route(*_args):
            return VoiceTurnDecision(
                "conversation", acknowledgement="weather"
            )

        speech = FakeSpeech()
        provider = MuseElevenProvider(
            FakeHost(link), speech, FakeRecognizer(), internal_turn_handler=route
        )
        await provider.start()
        await provider.append_audio(b"\x01\x00" * 960)
        await provider.finish_turn(uuid4())
        events = await self._collect_turn(provider)
        self.assertEqual(speech.text, "Hello from Muse.")
        transcripts = [
            event["delta"] for event in events
            if event["type"] == "session.output_transcript.delta"
        ]
        self.assertEqual(transcripts[0], "Let me check the weather.")
        await provider.close()

    async def test_wake_greeting_uses_cached_clip_without_calling_muse(self):
        link = FakeMuseLink()
        recognizer = FakeRecognizer()
        recognizer.text = "Hey Watson"

        async def route(*_args):
            return VoiceTurnDecision("conversation", acknowledgement="greeting")

        speech = FakeSpeech()
        provider = MuseElevenProvider(
            FakeHost(link), speech, recognizer, internal_turn_handler=route
        )
        await provider.start()
        await provider.append_audio(b"\x01\x00" * 960)
        await provider.finish_turn(uuid4())
        events = await self._collect_turn(provider)

        self.assertEqual(link.turns, [])
        self.assertEqual(speech.calls, 0)
        self.assertEqual(events[-1]["reply_expectation"], "open_answer")
        self.assertIn(
            "Hey, how can I help?",
            "".join(e.get("delta", "") for e in events),
        )
        await provider.close()

    async def test_empty_first_transcript_greets_and_keeps_session_open(self):
        link = FakeMuseLink()
        recognizer = FakeRecognizer()
        recognizer.text = ""
        routed = []

        async def route(text, _turn_id, previous_user, previous_reply):
            routed.append((text, previous_user, previous_reply))
            return VoiceTurnDecision("conversation")

        provider = MuseElevenProvider(
            FakeHost(link), FakeSpeech(), recognizer, internal_turn_handler=route
        )
        await provider.start()
        await provider.append_audio(b"\x01\x00" * 960)
        await provider.finish_turn(uuid4())
        events = await self._collect_turn(provider)

        self.assertEqual(link.turns, [])
        self.assertEqual(routed, [])
        self.assertEqual(
            [event["delta"] for event in events
             if event["type"] == "session.output_transcript.delta"],
            ["Hey, how can I help?"],
        )
        self.assertTrue(any(event["type"] == "session.output_audio.delta" for event in events))
        self.assertEqual(events[-1]["reply_expectation"], "open_answer")

        recognizer.text = "What's the weather?"
        await provider.append_audio(b"\x01\x00" * 960)
        await provider.finish_turn(uuid4())
        await self._collect_turn(provider)
        self.assertEqual(
            routed,
            [("What's the weather?", None, "Hey, how can I help?")],
        )
        await provider.close()

    async def test_empty_followup_gets_one_repeat_prompt_then_stops_retrying(self):
        link = FakeMuseLink()
        recognizer = FakeRecognizer()
        provider = MuseElevenProvider(FakeHost(link), FakeSpeech(), recognizer)
        await provider.start()

        await provider.append_audio(b"\x01\x00" * 960)
        await provider.finish_turn(uuid4())
        await self._collect_turn(provider)

        recognizer.text = ""
        await provider.append_audio(b"\x01\x00" * 18_000)  # 750 ms at 24 kHz mono PCM16
        provider._last_reply_expectation = "open_answer"
        await provider.finish_turn(uuid4(), vad_speech_ms=760)
        retry = await self._collect_turn(provider)
        retry_text = "".join(
            event.get("delta", "") for event in retry
            if event["type"] == "session.output_transcript.delta"
        )
        self.assertEqual(retry_text, "Sorry, I didn't catch that. Could you say it again?")
        self.assertEqual(retry[-1]["reply_expectation"], "open_answer")

        await provider.append_audio(b"\x01\x00" * 18_000)
        await provider.finish_turn(uuid4(), vad_speech_ms=760)
        ignored = await self._collect_turn(provider)
        self.assertIn("session.input_turn.ignored", [event["type"] for event in ignored])
        self.assertFalse(any(event["type"] == "session.output_audio.done" for event in ignored))
        await provider.close()

    async def test_empty_transcript_without_invitation_stays_silent(self):
        provider = MuseElevenProvider(FakeHost(FakeMuseLink()), FakeSpeech(), FakeRecognizer())
        await provider.start()
        await provider.append_audio(b"\x01\x00" * 960)
        await provider.finish_turn(uuid4())
        await self._collect_turn(provider)

        provider._last_reply_expectation = "none"
        provider._recognizer.text = ""
        await provider.append_audio(b"\x01\x00" * 18_000)
        await provider.finish_turn(uuid4(), vad_speech_ms=760)
        ignored = await self._collect_turn(provider)

        self.assertIn("session.input_turn.ignored", [event["type"] for event in ignored])
        self.assertFalse(any(event["type"] == "session.output_audio.done" for event in ignored))
        await provider.close()

    async def test_speech_failure_logs_safe_stage_and_error_type(self):
        link = FakeMuseLink()
        provider = MuseElevenProvider(FakeHost(link), FailingSpeech(), FakeRecognizer())
        await provider.start()
        await provider.append_audio(b"\x01\x00" * 960)
        await provider.finish_turn(uuid4())

        with self.assertLogs("service.muse_provider", level="WARNING") as captured:
            while True:
                try:
                    await asyncio.wait_for(provider.receive_event(), timeout=3)
                except LiveProviderError:
                    break

        logs = "\n".join(captured.output)
        self.assertRegex(
            logs,
            r"voice turn failed .*stage=elevenlabs_tts(?:_finalize)? error_type=RuntimeError",
        )
        self.assertNotIn("private provider detail", logs)
        await provider.close()

    async def test_voice_turns_round_trip_through_muse_and_speech_stream(self):
        link = FakeMuseLink()
        speech = FakeSpeech()
        recognizer = FakeRecognizer()
        with tempfile.TemporaryDirectory() as temporary:
            registry = MuseConversationRegistry(Path(temporary) / "conversations.sqlite3")
            provider = MuseElevenProvider(
                FakeHost(link), speech, recognizer, conversation_registry=registry,
                conversation_key="owner:conversation",
            )
            self.assertIsNone(provider._turn_timeout)

            self.assertEqual(await provider.start(), "muse-gadget-session")
            pcm = b"\x01\x00" * 960
            await provider.append_audio(pcm)
            await provider.finish_turn(uuid4())
            first = await self._collect_turn(provider)

            expected_prompt = (
                f"{CUBE_VOICE_RESPONSE_INSTRUCTION}\n\n"
                "Spoken request:\nHello Muse."
            )
            self.assertEqual(link.turns[0][0], expected_prompt)
            self.assertIn("greetings and small talk", CUBE_VOICE_RESPONSE_INSTRUCTION)
            self.assertIsNotNone(link.turns[0][1])
            self.assertEqual(link.turns[0][2], "text")
            UUID(link.turns[0][1])
            self.assertEqual(link.subscriptions[0], link.turns[0][1])
            self.assertEqual(
                link.operations[:2],
                [("send", link.turns[0][1]), ("subscribe", link.turns[0][1])],
            )
            self.assertEqual(recognizer.audio, [pcm])
            self.assertEqual(
                [event["type"] for event in first],
                [
                    "session.processing",
                    "session.input_transcript.delta",
                    "session.output_transcript.delta",
                    "session.output_audio.delta",
                    "session.output_audio.done",
                ],
            )
            self.assertEqual(first[1]["delta"], "Hello Muse.")
            self.assertEqual(speech.text, "Hello from Muse.")
            await provider.close()
            self.assertEqual(
                stat.S_IMODE((Path(temporary) / "conversations.sqlite3").stat().st_mode),
                0o600,
            )

            # Recover the dedicated Muse chat and replay filter after a Cube
            # voice session closes or the API process restarts.
            provider = MuseElevenProvider(
                FakeHost(link), speech, recognizer, conversation_registry=registry,
                conversation_key="owner:conversation",
            )
            await provider.start()
            await provider.append_audio(pcm)
            await provider.finish_turn(uuid4())
            second = await self._collect_turn(provider)
            self.assertEqual(
                link.turns[1], (expected_prompt, link.turns[0][1], "text")
            )
            self.assertEqual(link.subscriptions[1], link.turns[0][1])
            self.assertEqual(
                link.operations[2:],
                [("subscribe", link.turns[0][1]), ("send", link.turns[0][1])],
            )
            self.assertEqual(speech.text, "Hello from Muse.Second answer.")
            self.assertEqual(len(second), 5)
            self.assertTrue(all("stale" not in str(event) for event in second))
            await provider.close()

    async def test_rejects_oversized_turn_before_sending_to_muse(self):
        link = FakeMuseLink()
        provider = MuseElevenProvider(FakeHost(link), FakeSpeech(), FakeRecognizer())
        await provider.start()
        with self.assertRaisesRegex(Exception, "2 MiB"):
            await provider.append_audio(bytes(MAX_TURN_PCM_BYTES + 2))
        self.assertEqual(link.turns, [])
        await provider.close()

    async def test_factory_creates_isolated_provider_per_voice_session(self):
        link = FakeMuseLink()
        factory = muse_eleven_provider_factory(
            muse_host=FakeHost(link), speech=FakeSpeech(), recognizer=FakeRecognizer()
        )
        first = factory(object(), object())
        second = factory(object(), object())
        self.assertIsNot(first, second)

    async def _collect_turn(self, provider: MuseElevenProvider) -> list[dict]:
        events = []
        for _ in range(20):
            event = await asyncio.wait_for(provider.receive_event(), timeout=3)
            events.append(event)
            if event["type"] in {
                "session.output_audio.done", "session.closed", "session.input_turn.ignored"
            }:
                break
        else:
            self.fail("provider turn omitted its terminal event")
        for _ in range(100):
            if not provider._busy:
                break
            await asyncio.sleep(0.01)
        return events


if __name__ == "__main__":
    unittest.main()
