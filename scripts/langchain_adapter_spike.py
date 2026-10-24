"""Exercise Luna/Sol streaming, local tool continuation, and stream cancellation."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from scripts.check_openai_access import load_key


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gpt-6-luna", "gpt-6.1-sol")
STREAM_TIMEOUT_SECONDS = 40


class SpikeError(RuntimeError):
    """Sanitized failure safe to print."""


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers using the local test tool."""
    return a + b


def _model(model_name: str, api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        use_responses_api=True,
        output_version="responses/v1",
        reasoning={"effort": "low"},
        max_tokens=128,
        timeout=STREAM_TIMEOUT_SECONDS,
        max_retries=0,
    )


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        )
    return ""


async def _check_stream(model: ChatOpenAI) -> dict[str, Any]:
    chunks = 0
    fragments: list[str] = []
    async for chunk in model.astream("Reply with exactly STREAM_OK."):
        chunks += 1
        fragments.append(_text(chunk.content))
    transcript = "".join(fragments).strip()
    if not transcript or "stream_ok" not in transcript.lower():
        raise SpikeError("streaming response did not contain the expected marker")
    return {"chunk_count": chunks, "transcript": transcript}


async def _check_tool_continuation(model: ChatOpenAI) -> dict[str, Any]:
    model_with_tool = model.bind_tools(
        [add_numbers],
        tool_choice="add_numbers",
        strict=True,
        parallel_tool_calls=False,
    )
    messages = [
        SystemMessage(
            content=(
                "Use the provided tool for arithmetic. Do not calculate in the model. "
                "After the tool returns, report only the numeric result."
            )
        ),
        HumanMessage(content="Add 19 and 23."),
    ]
    first = await model_with_tool.ainvoke(messages)
    if len(first.tool_calls) != 1:
        raise SpikeError("model did not return exactly one tool call")
    call = first.tool_calls[0]
    if call.get("name") != "add_numbers" or call.get("args") != {"a": 19, "b": 23}:
        raise SpikeError("model returned unexpected local tool arguments")
    result = add_numbers.invoke(call["args"])
    continuation = await model.ainvoke(
        [
            *messages,
            first,
            ToolMessage(content=str(result), tool_call_id=call["id"]),
        ]
    )
    transcript = _text(continuation.content).strip()
    if not re.search(r"\b42\b|\bforty[- ]two\b", transcript, re.IGNORECASE):
        raise SpikeError("model did not use the local tool result in its continuation")
    return {
        "tool_name": call["name"],
        "tool_arguments": call["args"],
        "tool_result": result,
        "continuation": transcript,
        "response_id": continuation.response_metadata.get("id"),
    }


async def _check_cancellation(model: ChatOpenAI) -> dict[str, Any]:
    first_chunk = asyncio.Event()
    chunk_count = 0

    async def consume_stream() -> None:
        nonlocal chunk_count
        stream = model.astream(
            "Write a detailed 1000-word explanation of how a paper airplane flies."
        )
        try:
            async for chunk in stream:
                if _text(chunk.content):
                    chunk_count += 1
                    first_chunk.set()
        finally:
            await stream.aclose()

    task = asyncio.create_task(consume_stream())
    try:
        await asyncio.wait_for(first_chunk.wait(), timeout=STREAM_TIMEOUT_SECONDS)
        if task.done():
            raise SpikeError("cancellation request finished before first chunk could be cancelled")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        if not task.cancelled():
            raise SpikeError("LangChain stream task did not enter cancelled state")
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    return {
        "first_chunk_received": first_chunk.is_set(),
        "chunks_before_cancel": chunk_count,
        "langchain_task_cancelled": task.cancelled(),
        "provider_final_usage_confirmed": False,
    }


async def _run(api_key: str) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "schema_version": 1,
        "test": "langchain_openai_responses_adapter",
        "integration": "langchain-openai ChatOpenAI",
        "transport": "OpenAI Responses API",
        "models": {},
        "external_tools": 0,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    for model_name in MODELS:
        model_trace: dict[str, Any] = {}
        model = _model(model_name, api_key)
        for name, check in (
            ("streaming", _check_stream),
            ("tool_continuation", _check_tool_continuation),
            ("cancellation", _check_cancellation),
        ):
            try:
                model_trace[name] = await check(model)
            except SpikeError:
                raise
            except Exception as error:
                raise SpikeError(f"{model_name} {name} failed ({type(error).__name__})") from None
        trace["models"][model_name] = model_trace
    return trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-out",
        type=Path,
        default=ROOT / "docs" / "p04-langchain-trace.json",
    )
    args = parser.parse_args()
    trace_path = args.trace_out.resolve()
    if trace_path.exists():
        parser.error(f"trace already exists; choose another --trace-out: {trace_path}")
    api_key = load_key()
    if not api_key:
        parser.error("OPENAI_API_KEY missing; key value was not printed")
    try:
        trace = asyncio.run(_run(api_key))
    finally:
        del api_key
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("P04 LangChain adapter spike passed for Luna and Sol.")
    for model_name, model_trace in trace["models"].items():
        print(
            f"{model_name}: streaming {model_trace['streaming']['chunk_count']} chunks; "
            f"local tool {model_trace['tool_continuation']['tool_result']}; "
            "stream cancellation confirmed."
        )
    print(f"Sanitized trace: {trace_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SpikeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
