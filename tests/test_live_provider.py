from __future__ import annotations

import asyncio
import json
import unittest
from decimal import Decimal

from service.live_provider import (
    LIVE_SESSIONS_URL,
    LiveFinalizationUnconfirmed,
    LiveProviderError,
    OpenAILiveSession,
)


class FakeConnection:
    def __init__(self, events=()):
        self.incoming: asyncio.Queue[str] = asyncio.Queue()
        for event in events:
            self.incoming.put_nowait(json.dumps(event))
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        return await self.incoming.get()

    async def close(self) -> None:
        self.closed = True

    def emit(self, event: dict) -> None:
        self.incoming.put_nowait(json.dumps(event))


class OpenAILiveSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_audio_and_final_usage_lifecycle(self) -> None:
        connection = FakeConnection(
            [
                {
                    "type": "session.started",
                    "session": {"id": "live_test_1"},
                }
            ]
        )
        connect_args = {}

        async def connector(url, **kwargs):
            connect_args["url"] = url
            connect_args.update(kwargs)
            return connection

        session = OpenAILiveSession(
            "sk-test-do-not-log",
            instructions="Be concise.",
            connector=connector,
        )
        self.assertEqual(await session.start(), "live_test_1")
        self.assertEqual(connect_args["url"], LIVE_SESSIONS_URL)
        self.assertEqual(
            connect_args["additional_headers"],
            {"Authorization": "Bearer sk-test-do-not-log"},
        )
        self.assertEqual(connection.sent[0]["session"]["delegation"], {"type": "client"})

        await session.append_audio(b"\x00\x00" * 480)
        audio_event = connection.sent[-1]
        self.assertEqual(audio_event["type"], "session.input_audio.append")
        self.assertTrue(audio_event["audio"])

        connection.emit({"type": "session.usage.updated", "usage": {"seconds": 12}})
        self.assertEqual((await session.receive_event())["type"], "session.usage.updated")
        connection.emit({"type": "session.usage.updated", "usage": {"seconds": 15}})
        self.assertEqual((await session.receive_event())["usage"]["seconds"], 15)
        self.assertEqual(session.latest_usage_seconds, Decimal("15"))

        connection.emit({"type": "session.closed", "usage": {"seconds": 16}, "reason": "close_requested"})
        closed = await session.close()
        self.assertEqual(closed.provider_session_id, "live_test_1")
        self.assertEqual(closed.connected_seconds, Decimal("16"))
        self.assertEqual(connection.sent[-1]["type"], "session.close")
        self.assertTrue(connection.closed)

    async def test_audio_frames_are_bounded_and_only_pcm16(self) -> None:
        connection = FakeConnection([{"type": "session.started", "session": {"id": "live_test_2"}}])

        async def connector(_url, **_kwargs):
            return connection

        session = OpenAILiveSession("key", instructions="Be concise.", connector=connector)
        await session.start()

        for payload in (b"", b"\x00", b"\x00" * 9_602):
            with self.assertRaises(ValueError):
                await session.append_audio(payload)
        await session.abort()

    async def test_start_error_sanitizes_provider_payload_and_closes_socket(self) -> None:
        connection = FakeConnection(
            [{"type": "error", "error": {"message": "sk-secret in provider body"}}]
        )

        async def connector(_url, **_kwargs):
            return connection

        session = OpenAILiveSession("sk-secret", instructions="Be concise.", connector=connector)
        with self.assertRaises(LiveProviderError) as error:
            await session.start()

        self.assertNotIn("sk-secret", str(error.exception))
        self.assertTrue(connection.closed)

    async def test_close_timeout_marks_usage_unconfirmed(self) -> None:
        connection = FakeConnection([{"type": "session.started", "session": {"id": "live_test_3"}}])

        async def connector(_url, **_kwargs):
            return connection

        session = OpenAILiveSession(
            "key",
            instructions="Be concise.",
            close_timeout_seconds=0.01,
            connector=connector,
        )
        await session.start()
        with self.assertRaises(LiveFinalizationUnconfirmed):
            await session.close()
        self.assertTrue(connection.closed)

    async def test_context_events_require_active_session_and_correlation(self) -> None:
        connection = FakeConnection([{"type": "session.started", "session": {"id": "live_test_4"}}])

        async def connector(_url, **_kwargs):
            return connection

        session = OpenAILiveSession("key", instructions="Be concise.", connector=connector)
        await session.start()
        event_id = await session.append_context(
            "The backend verified task result.",
            kind="commentary",
            delegation_id="delegation-4",
        )
        self.assertEqual(connection.sent[-1]["event_id"], event_id)
        self.assertEqual(connection.sent[-1]["delegation_id"], "delegation-4")
        with self.assertRaises(ValueError):
            await session.append_context(
                "x" * 501, kind="thinking", delegation_id=None
            )
        with self.assertRaises(ValueError):
            await session.append_context(
                "context", kind="thinking", delegation_id=" "
            )
        await session.abort()


if __name__ == "__main__":
    unittest.main()
