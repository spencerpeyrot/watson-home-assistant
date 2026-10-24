from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx

from service.voice_commands import (
    DecisionsUnavailable,
    FallbackDecisionsClient,
    JevDecisionsClient,
    VoiceCommandFlow,
)


class FakeDecisions:
    def __init__(self, choice: str = "conversation", confidence: float = 0.9) -> None:
        self.choice = choice
        self.confidence = confidence
        self.input_text = ""
        self.questions = []

    async def decide(self, *, input_text, questions):
        self.input_text = input_text
        self.questions = questions
        return {
            "route": (self.choice, self.confidence),
            "acknowledgement": ("none", 0.95),
        }


def flow(decisions: FakeDecisions) -> VoiceCommandFlow:
    principal = SimpleNamespace(
        role="device", owner_id=uuid4(), device_id="cube-01"
    )
    session = SimpleNamespace(session_id=uuid4(), conversation_id=uuid4())
    return VoiceCommandFlow(
        decisions=decisions,
        principal=principal,
        session=session,
        tool_registry=None,
        ledger=None,
    )


class VoiceCommandRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_jev_choice_is_rejected(self):
        transport = httpx.MockTransport(lambda _request: httpx.Response(
            200,
            json={"answers": {
                "expectation": {
                    "type": "choice", "choice": "unexpected", "confidence": 0.95
                }
            }},
        ))
        client = httpx.AsyncClient(transport=transport)
        question = {
            "type": "choice",
            "name": "expectation",
            "instructions": "Classify reply.",
            "choices": [{"value": "none", "description": "No answer expected."}],
        }

        with patch("service.voice_commands.httpx.AsyncClient", return_value=client):
            with self.assertRaisesRegex(DecisionsUnavailable, "invalid_choice_answer"):
                await JevDecisionsClient("test-key").decide(
                    input_text="{}", questions=[question]
                )

    async def test_contextual_followup_routes_to_conversation_in_one_request(self):
        decisions = FakeDecisions()
        result = await flow(decisions).handle(
            "New York City",
            uuid4(),
            previous_user="What's the weather tomorrow?",
            previous_reply="Where are you located?",
        )

        self.assertEqual(result.route, "conversation")
        state = json.loads(decisions.input_text)
        self.assertEqual(state["current_transcript"], "New York City")
        self.assertEqual(state["previous_user"], "What's the weather tomorrow?")
        self.assertEqual(state["previous_cube_reply"], "Where are you located?")
        self.assertEqual(len(decisions.questions), 2)
        self.assertEqual(
            {choice["value"] for choice in decisions.questions[0]["choices"]},
            {"ignore", "conversation", "internal", "close"},
        )
        self.assertIn("greeting", decisions.questions[0]["instructions"])
        self.assertEqual(decisions.questions[1]["name"], "acknowledgement")

    async def test_jev_selects_bounded_acknowledgement_for_slow_lookup(self):
        class LookupDecisions(FakeDecisions):
            async def decide(self, *, input_text, questions):
                self.input_text = input_text
                self.questions = questions
                return {
                    "route": ("conversation", 0.95),
                    "acknowledgement": ("weather", 0.95),
                }

        result = await flow(LookupDecisions()).handle(
            "What's the weather tomorrow?", uuid4()
        )
        self.assertEqual(result.acknowledgement, "weather")

    async def test_low_confidence_conversation_is_ignored(self):
        result = await flow(FakeDecisions(confidence=0.4)).handle("Yes", uuid4())
        self.assertEqual(result.route, "ignore")

    async def test_classifies_whether_cube_reply_invites_followup(self):
        class ExpectationDecisions(FakeDecisions):
            def __init__(self, choice):
                super().__init__()
                self.expectation_choice = choice

            async def decide(self, *, input_text, questions):
                self.input_text = input_text
                self.questions = questions
                name = questions[0]["name"]
                if name == "expectation":
                    return {name: (self.expectation_choice, 0.9)}
                return {"route": ("conversation", 0.9)}

        decisions = ExpectationDecisions("open_answer")
        voice_flow = flow(decisions)
        expectation = await voice_flow.classify_reply_expectation(
            user_turn="Hey Watson, how are you?",
            reply="Doing well. What’s on your mind?",
        )

        self.assertEqual(expectation, "open_answer")
        self.assertIn("cube_reply", json.loads(decisions.input_text))
        result = await voice_flow.handle(
            "Just wanted to say hi.", uuid4(), "Hey Watson, how are you?",
            "Doing well. What’s on your mind?",
        )
        self.assertEqual(result.route, "conversation")

    async def test_bare_ack_after_no_invitation_is_routed_with_reply_state(self):
        class ExpectationDecisions(FakeDecisions):
            async def decide(self, *, input_text, questions):
                self.questions = questions
                self.input_text = input_text
                if questions[0]["name"] == "expectation":
                    return {"expectation": ("none", 0.95)}
                return {"route": ("ignore", 0.95)}

        decisions = ExpectationDecisions()
        voice_flow = flow(decisions)
        self.assertEqual(
            await voice_flow.classify_reply_expectation(
                user_turn="What time is my haircut?",
                reply="10:00 in the morning.",
            ),
            "none",
        )

        result = await voice_flow.handle(
            "Yes!", uuid4(), "What time is my haircut?", "10:00 in the morning."
        )

        self.assertEqual(result.route, "ignore")
        self.assertEqual(decisions.questions[0]["name"], "route")
        self.assertEqual(
            json.loads(decisions.input_text)["previous_reply_expectation"], "none"
        )

    async def test_low_confidence_reply_expectation_uses_default_window(self):
        class ExpectationDecisions(FakeDecisions):
            async def decide(self, *, input_text, questions):
                return {"expectation": ("closing", 0.3)}

        expectation = await flow(ExpectationDecisions()).classify_reply_expectation(
            user_turn="Thanks, that’s all.", reply="Take care."
        )
        self.assertIsNone(expectation)

    async def test_reply_expectation_uses_luna_when_jev_is_unavailable(self):
        calls = []

        class FailedJev:
            async def decide(self, **_kwargs):
                calls.append("jev")
                raise DecisionsUnavailable("timeout")

        class Luna:
            async def decide(self, **_kwargs):
                calls.append("luna")
                return {"expectation": ("confirmation", 0.9)}

        expectation = await flow(FallbackDecisionsClient(FailedJev(), Luna())).classify_reply_expectation(
            user_turn="Want the other forecast?", reply="Would you like it?"
        )

        self.assertEqual(expectation, "confirmation")
        self.assertEqual(calls, ["jev", "luna"])

    async def test_low_confidence_close_keeps_session_open(self):
        result = await flow(FakeDecisions("close", 0.4)).handle("Goodbye", uuid4())
        self.assertEqual(result.route, "ignore")

    async def test_high_confidence_goodbye_closes_session(self):
        result = await flow(FakeDecisions("close", 0.9)).handle("Goodbye", uuid4())
        self.assertEqual(result.route, "close")

    async def test_provider_failure_fails_closed_instead_of_reaching_muse(self):
        class FailedDecisions(FakeDecisions):
            async def decide(self, **_kwargs):
                from service.voice_commands import DecisionsUnavailable

                raise DecisionsUnavailable("offline")

        result = await flow(FailedDecisions()).handle("background words", uuid4())
        self.assertEqual(result.route, "ignore")

    async def test_unavailable_primary_uses_luna_fallback(self):
        calls = []

        class FailedJev:
            async def decide(self, **_kwargs):
                calls.append("jev")
                raise DecisionsUnavailable("invalid_answer")

        class Luna:
            async def decide(self, **_kwargs):
                calls.append("luna")
                return {"route": ("conversation", 0.9)}

        decisions = FallbackDecisionsClient(FailedJev(), Luna())
        result = await flow(decisions).handle("What time is it?", uuid4())

        self.assertEqual(result.route, "conversation")
        self.assertEqual(calls, ["jev", "luna"])
