from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import uuid4

from contracts.models import SessionResponse
from service.live_composition import openai_live_provider_factory
from service.live_provider import OpenAILiveSession


class LiveCompositionTests(unittest.TestCase):
    def test_factory_creates_isolated_provider_per_session_without_connecting(self):
        async def connector(*_args, **_kwargs):
            raise AssertionError("factory must not connect during composition")

        factory = openai_live_provider_factory(
            api_key="test-key-never-logged",
            model="gpt-live-1",
            voice="marin",
            instructions="Use client delegation for tasks.",
            connector=connector,
        )
        session = SessionResponse(
            session_id=uuid4(),
            conversation_id=uuid4(),
            state="connecting",
            opened_at=datetime.now(timezone.utc),
            idle_close_seconds=30,
        )

        first = factory(object(), session)
        second = factory(object(), session)

        self.assertIsInstance(first, OpenAILiveSession)
        self.assertIsInstance(second, OpenAILiveSession)
        self.assertIsNot(first, second)
        self.assertEqual(first._model, "gpt-live-1")
        self.assertEqual(first._voice, "marin")
        self.assertEqual(first._instructions, "Use client delegation for tasks.")
        self.assertIs(first._connector, connector)

    def test_factory_rejects_missing_key_or_instructions(self):
        with self.assertRaisesRegex(ValueError, "API key"):
            openai_live_provider_factory(api_key=" ", instructions="assistant")
        with self.assertRaisesRegex(ValueError, "instructions"):
            openai_live_provider_factory(api_key="test-key", instructions=" ")


if __name__ == "__main__":
    unittest.main()
