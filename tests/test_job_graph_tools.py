from __future__ import annotations

import json
import unittest
from uuid import UUID

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from contracts.models import TaskRequest, ToolResult
from service.home_assistant import HomeAssistantLocalBridge
from service.job_graph import TaskGraph
from service.tool_policy import ActionClaim, ToolPolicyRegistry


OWNER_ID = UUID("00000000-0000-4000-8000-000000000711")
JOB_ID = UUID("00000000-0000-4000-8000-000000000712")


def task_request(text: str) -> TaskRequest:
    return TaskRequest(
        device_id="cube-01",
        conversation_id=UUID("00000000-0000-4000-8000-000000000713"),
        session_id=UUID("00000000-0000-4000-8000-000000000714"),
        turn_id=UUID("00000000-0000-4000-8000-000000000715"),
        delegation_id="delegation-tool-test",
        trace_id=UUID("00000000-0000-4000-8000-000000000716"),
        request_text=text,
        context_revision=0,
        locale="en-US",
        timezone="America/New_York",
        idempotency_key="tool-graph-test",
    )


class FakeToolModel:
    def __init__(self, entity_id: str):
        self.entity_id = entity_id
        self.bound_names: list[str] = []
        self.bound_tools = []
        self.tool_results: list[dict] = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        self.bound_names = [tool.name for tool in tools]
        return self

    async def ainvoke(self, messages):
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        if not tool_messages:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "set_light_state",
                        "args": {"entity_id": self.entity_id, "state": "on"},
                        "id": "call-light-1",
                        "type": "tool_call",
                    }
                ],
            )
        result = json.loads(tool_messages[-1].content)
        self.tool_results.append(result)
        if result.get("status") == "succeeded" and result.get("verified"):
            return AIMessage(content="Desk light is on.")
        return AIMessage(content="I could not verify that light action.")


class FakeLedger:
    def __init__(self):
        self.claims = []
        self.completions = []

    async def claim_action(self, **kwargs):
        self.claims.append(kwargs)
        return ActionClaim(
            outcome="execute", action_id=kwargs["action_id"], state="executing"
        )

    async def complete_action(self, **kwargs):
        self.completions.append(kwargs)

    async def mark_action_uncertain(self, **_kwargs):
        raise AssertionError("fake local action should not become uncertain")


class TaskGraphToolTests(unittest.IsolatedAsyncioTestCase):
    def make_graph(self, entity_id: str):
        sender_calls = []

        async def deliver(request):
            sender_calls.append(request)
            return ToolResult(
                action_id=request.action_id,
                trace_id=request.trace_id,
                device_id=request.device_id,
                conversation_id=request.conversation_id,
                session_id=request.session_id,
                delegation_id=request.delegation_id,
                context_revision=request.context_revision,
                job_id=request.job_id,
                turn_id=request.turn_id,
                status="succeeded",
                verified=True,
                observed_state={"entity_id": entity_id, "state": "on"},
                retryable=False,
            )

        registry = ToolPolicyRegistry()
        HomeAssistantLocalBridge(
            allowed_light_entities=frozenset({"light.desk"}), deliver=deliver
        ).register_tools(registry)
        ledger = FakeLedger()
        model = FakeToolModel(entity_id)
        graph = TaskGraph(
            luna_model=model,
            sol_model=RunnableLambda(lambda _messages: AIMessage(content="unused")),
            tool_registry=registry,
            action_ledger=ledger,
        )
        return graph, model, ledger, sender_calls

    async def test_simple_luna_tool_call_uses_policy_ledger_and_verified_device_result(self):
        graph, model, ledger, sender_calls = self.make_graph("light.desk")

        result = await graph.run(
            task_request("Turn on the desk light."), job_id=JOB_ID, owner_id=OWNER_ID
        )

        self.assertEqual(result.response, "Desk light is on.")
        self.assertEqual(
            model.bound_names, ["get_light_state", "set_light_state"]
        )
        entity_schema = model.bound_tools[1].args_schema.model_json_schema()["properties"]["entity_id"]
        entity_enum = entity_schema.get("enum", [entity_schema.get("const")])
        self.assertEqual(entity_enum, ["light.desk"])
        self.assertEqual(len(sender_calls), 1)
        self.assertEqual(sender_calls[0].payload, {"entity_id": "light.desk", "state": "on"})
        self.assertEqual(len(ledger.claims), 1)
        self.assertEqual(len(ledger.completions), 1)
        self.assertEqual(ledger.claims[0]["owner_id"], OWNER_ID)
        self.assertEqual(ledger.claims[0]["job_id"], JOB_ID)
        self.assertTrue(model.tool_results[0]["verified"])

    async def test_unlisted_light_is_denied_before_ledger_or_device_delivery(self):
        graph, model, ledger, sender_calls = self.make_graph("light.unauthorized")

        result = await graph.run(
            task_request("Turn on the other light."), job_id=JOB_ID, owner_id=OWNER_ID
        )

        self.assertEqual(result.response, "I could not verify that light action.")
        self.assertEqual(model.tool_results[0]["status"], "denied")
        self.assertEqual(model.tool_results[0]["error_code"], "ACTION_DENIED")
        self.assertEqual(ledger.claims, [])
        self.assertEqual(sender_calls, [])


if __name__ == "__main__":
    unittest.main()
