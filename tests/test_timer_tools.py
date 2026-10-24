from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from uuid import UUID

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool

from contracts.models import TaskRequest, TimerRecord
from service.job_graph import TaskGraph
from service.timer_tools import TimerTools


OWNER_ID = UUID("00000000-0000-4000-8000-000000000801")
CONVERSATION_ID = UUID("00000000-0000-4000-8000-000000000802")
TIMER_ID = UUID("00000000-0000-4000-8000-000000000803")


class FakeTimerRepository:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.listed: list[dict[str, object]] = []
        self.cancelled: list[dict[str, object]] = []
        self.records: tuple[TimerRecord, ...] = ()
        self.cancel_result = True

    async def create_timer(self, **values) -> TimerRecord:
        self.created.append(values)
        request = values["request"]
        now = datetime.now(timezone.utc)
        return TimerRecord(
            timer_id=TIMER_ID,
            conversation_id=request.conversation_id,
            due_at=request.due_at,
            message=request.message,
            status="scheduled",
            created_at=now,
        )

    async def list_timers(self, **values) -> tuple[TimerRecord, ...]:
        self.listed.append(values)
        return self.records

    async def cancel_timer(self, **values) -> bool:
        self.cancelled.append(values)
        return self.cancel_result


class FakeTimerModel:
    def __init__(self) -> None:
        self.bound_names: list[str] = []
        self.tool_results: list[dict[str, object]] = []
        self.due_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

    def bind_tools(self, tools):
        self.bound_names = [tool.name for tool in tools]
        return self

    async def ainvoke(self, messages):
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        if not tool_messages:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_timer",
                        "args": {"due_at": self.due_at, "message": "Tea is ready"},
                        "id": "call-timer-1",
                        "type": "tool_call",
                    }
                ],
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "total_tokens": 13,
                },
            )
        self.tool_results.append(json.loads(tool_messages[-1].content))
        return AIMessage(
            content="Timer set for tea.",
            usage_metadata={"input_tokens": 12, "output_tokens": 2, "total_tokens": 14},
        )


def timer_record(status: str = "scheduled") -> TimerRecord:
    now = datetime.now(timezone.utc)
    return TimerRecord(
        timer_id=TIMER_ID,
        conversation_id=CONVERSATION_ID,
        due_at=now + timedelta(minutes=5),
        message="Tea is ready",
        status=status,
        created_at=now,
    )


def timer_tools(repository: FakeTimerRepository) -> TimerTools:
    return TimerTools(
        repository,
        owner_id=OWNER_ID,
        device_id="cube-01",
        conversation_id=CONVERSATION_ID,
        task_idempotency_key="device/session/delegation",
        context_revision=2,
    )


class TimerToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_graph_exposes_owner_scoped_timer_tools(self) -> None:
        repository = FakeTimerRepository()
        model = FakeTimerModel()
        graph = TaskGraph(
            luna_model=model,
            sol_model=RunnableLambda(lambda _messages: AIMessage(content="unused")),
            timer_repository=repository,
        )
        request = TaskRequest(
            device_id="cube-01",
            conversation_id=CONVERSATION_ID,
            session_id=UUID("00000000-0000-4000-8000-000000000804"),
            turn_id=UUID("00000000-0000-4000-8000-000000000805"),
            delegation_id="delegation-timer-test",
            trace_id=UUID("00000000-0000-4000-8000-000000000806"),
            request_text="Set a timer for tea in five minutes.",
            context_revision=2,
            locale="en-US",
            timezone="America/New_York",
            idempotency_key="device/session/delegation",
        )

        result = await graph.run(
            request,
            job_id=UUID("00000000-0000-4000-8000-000000000807"),
            owner_id=OWNER_ID,
        )

        self.assertEqual(result.response, "Timer set for tea.")
        self.assertEqual(
            model.bound_names,
            ["create_timer", "list_timers", "cancel_timer"],
        )
        self.assertEqual(repository.created[0]["owner_id"], OWNER_ID)
        self.assertEqual(model.tool_results[0]["status"], "scheduled")
        self.assertEqual(result.measured_usage.input_tokens, 22)
        self.assertEqual(result.measured_usage.output_tokens, 5)

    async def test_unowned_graph_does_not_expose_timer_tools(self) -> None:
        repository = FakeTimerRepository()
        captured = []

        async def answer(messages):
            captured.extend(messages)
            return AIMessage(content="Timers are unavailable here.")

        graph = TaskGraph(
            luna_model=RunnableLambda(answer),
            sol_model=RunnableLambda(lambda _messages: AIMessage(content="unused")),
            timer_repository=repository,
        )
        request = TaskRequest(
            device_id="cube-01",
            conversation_id=CONVERSATION_ID,
            session_id=UUID("00000000-0000-4000-8000-000000000804"),
            turn_id=UUID("00000000-0000-4000-8000-000000000805"),
            delegation_id="delegation-timer-test-unowned",
            trace_id=UUID("00000000-0000-4000-8000-000000000806"),
            request_text="Set a timer for tea in five minutes.",
            context_revision=2,
            locale="en-US",
            timezone="America/New_York",
            idempotency_key="device/session/delegation-unowned",
        )

        result = await graph.run(
            request, job_id=UUID("00000000-0000-4000-8000-000000000808")
        )

        self.assertEqual(result.response, "Timers are unavailable here.")
        self.assertIn("Never claim a timer was created", captured[0].content)
        self.assertEqual(repository.created, [])

    async def test_invalid_timezone_withholds_timer_tools(self) -> None:
        repository = FakeTimerRepository()
        captured = []

        async def answer(messages):
            captured.extend(messages)
            return AIMessage(content="Please confirm your timezone first.")

        graph = TaskGraph(
            luna_model=RunnableLambda(answer),
            sol_model=RunnableLambda(lambda _messages: AIMessage(content="unused")),
            timer_repository=repository,
        )
        request = TaskRequest(
            device_id="cube-01",
            conversation_id=CONVERSATION_ID,
            session_id=UUID("00000000-0000-4000-8000-000000000804"),
            turn_id=UUID("00000000-0000-4000-8000-000000000805"),
            delegation_id="delegation-timer-invalid-zone",
            trace_id=UUID("00000000-0000-4000-8000-000000000806"),
            request_text="Set a timer for tea in five minutes.",
            context_revision=2,
            locale="en-US",
            timezone="Not/AZone",
            idempotency_key="device/session/delegation-invalid-zone",
        )

        result = await graph.run(
            request,
            job_id=UUID("00000000-0000-4000-8000-000000000809"),
            owner_id=OWNER_ID,
        )

        self.assertEqual(result.response, "Please confirm your timezone first.")
        self.assertIn("Timer actions are unavailable", captured[0].content)
        self.assertEqual(repository.created, [])

    async def test_create_is_owner_scoped_future_only_and_retry_stable(self) -> None:
        repository = FakeTimerRepository()
        tools = {tool.name: tool for tool in timer_tools(repository).model_tools()}
        self.assertEqual(set(tools), {"create_timer", "list_timers", "cancel_timer"})
        due_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

        first = json.loads(
            await tools["create_timer"].ainvoke(
                {"due_at": due_at, "message": "  Tea is ready  "}
            )
        )
        second = json.loads(
            await tools["create_timer"].ainvoke(
                {"due_at": due_at, "message": "Tea is ready"}
            )
        )

        self.assertEqual(first["status"], "scheduled")
        self.assertEqual(second["timer_id"], first["timer_id"])
        self.assertEqual(len(repository.created), 2)
        self.assertEqual(repository.created[0]["owner_id"], OWNER_ID)
        self.assertEqual(repository.created[0]["device_id"], "cube-01")
        self.assertEqual(repository.created[0]["request"].conversation_id, CONVERSATION_ID)
        self.assertEqual(
            repository.created[0]["request"].idempotency_key,
            repository.created[1]["request"].idempotency_key,
        )

    async def test_tools_convert_to_openai_function_schemas(self) -> None:
        repository = FakeTimerRepository()
        functions = [
            convert_to_openai_tool(tool)
            for tool in timer_tools(repository).model_tools()
        ]

        self.assertEqual(
            [item["function"]["name"] for item in functions],
            ["create_timer", "list_timers", "cancel_timer"],
        )
        create_schema = functions[0]["function"]["parameters"]
        self.assertEqual(create_schema["properties"]["due_at"]["format"], "date-time")
        self.assertEqual(set(create_schema["required"]), {"due_at", "message"})

    async def test_past_timer_is_rejected_without_repository_write(self) -> None:
        repository = FakeTimerRepository()
        tools = {tool.name: tool for tool in timer_tools(repository).model_tools()}

        result = json.loads(
            await tools["create_timer"].ainvoke(
                {
                    "due_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
                    "message": "Tea is ready",
                }
            )
        )

        self.assertEqual(result["error_code"], "INVALID_REQUEST")
        self.assertEqual(repository.created, [])

    async def test_list_and_cancel_are_scoped_to_authenticated_owner(self) -> None:
        repository = FakeTimerRepository()
        repository.records = (timer_record(),)
        tools = {tool.name: tool for tool in timer_tools(repository).model_tools()}

        listed = json.loads(await tools["list_timers"].ainvoke({}))
        cancelled = json.loads(
            await tools["cancel_timer"].ainvoke({"timer_id": str(TIMER_ID)})
        )

        self.assertEqual(listed["timers"][0]["timer_id"], str(TIMER_ID))
        self.assertEqual(repository.listed, [{"owner_id": OWNER_ID, "limit": 10}])
        self.assertEqual(cancelled, {"status": "cancelled", "timer_id": str(TIMER_ID)})
        self.assertEqual(repository.cancelled, [{"owner_id": OWNER_ID, "timer_id": TIMER_ID}])


if __name__ == "__main__":
    unittest.main()
