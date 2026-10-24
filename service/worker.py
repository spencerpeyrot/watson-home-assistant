"""Configured durable job worker for Luna and bounded Sol research jobs."""

from __future__ import annotations
from voice.observability import initialize as initialize_observability, shutdown as shutdown_observability

import asyncio
import math
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import UUID, uuid4

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from service.artifacts import LocalArtifactStore
from service.checkpoint import postgres_checkpointer
from service.home_assistant import HomeAssistantLocalBridge, LocalToolSender
from service.internal_tools import WorkerDeviceToolClient
from service.job_graph import TaskGraph
from service.job_worker import DurableJobWorker
from service.postgres import PostgresStore
from service.research_agent import SolDeepAgent
from service.search import OpenAIWebSearch
from service.system_controls import LocalSystemControlsBridge
from service.tool_policy import ToolPolicyRegistry


@dataclass(frozen=True)
class WorkerSettings:
    database_url: str = field(repr=False)
    openai_api_key: str = field(repr=False)
    artifact_root: Path
    luna_model: str = "gpt-6-luna"
    sol_model: str = "gpt-6.1-sol"
    luna_max_output_tokens: int | None = None
    sol_max_output_tokens: int | None = None
    worker_id: UUID = field(default_factory=uuid4)
    lease_seconds: int = 30
    poll_seconds: float = 0.5
    internal_api_url: str | None = None
    worker_token: str | None = field(default=None, repr=False)
    local_light_entities: frozenset[str] = frozenset()

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> WorkerSettings:
        values = os.environ if environ is None else environ
        database_url = values.get("DATABASE_URL", "")
        api_key = values.get("OPENAI_API_KEY", "")
        artifact_root = values.get("CUBE_ARTIFACT_DIR", "")
        if not database_url.strip():
            raise RuntimeError("DATABASE_URL is required")
        if not api_key.strip():
            raise RuntimeError("OPENAI_API_KEY is required")
        if not artifact_root.strip():
            raise RuntimeError("CUBE_ARTIFACT_DIR is required")
        luna_model = values.get("OPENAI_LUNA_MODEL", "gpt-6-luna")
        sol_model = values.get("OPENAI_SOL_MODEL", "gpt-6.1-sol")
        if not luna_model.strip() or not sol_model.strip():
            raise RuntimeError("OpenAI Luna and Sol model names must not be blank")
        luna_max_output_tokens = _optional_token_limit(
            values, "OPENAI_LUNA_MAX_OUTPUT_TOKENS"
        )
        sol_max_output_tokens = _optional_token_limit(
            values, "OPENAI_SOL_MAX_OUTPUT_TOKENS"
        )
        worker_id_value = values.get("CUBE_WORKER_ID", "")
        try:
            worker_id = UUID(worker_id_value) if worker_id_value else uuid4()
        except ValueError:
            raise RuntimeError("CUBE_WORKER_ID must be a UUID") from None
        lease_seconds = _int_setting(values, "CUBE_WORKER_LEASE_SECONDS", 30, minimum=1)
        poll_seconds = _float_setting(values, "CUBE_WORKER_POLL_SECONDS", 0.5, minimum=0.05)
        internal_api_url = values.get("CUBE_INTERNAL_API_URL", "").strip()
        worker_token = values.get("CUBE_WORKER_TOKEN", "")
        raw_entities = values.get("CUBE_LOCAL_LIGHT_ENTITIES", "")
        entity_parts = [part.strip() for part in raw_entities.split(",") if part.strip()]
        if raw_entities.strip() and len(entity_parts) != len(raw_entities.split(",")):
            raise RuntimeError("CUBE_LOCAL_LIGHT_ENTITIES contains an empty entry")
        if len(set(entity_parts)) != len(entity_parts):
            raise RuntimeError("CUBE_LOCAL_LIGHT_ENTITIES contains duplicates")
        if bool(internal_api_url) != bool(worker_token):
            raise RuntimeError(
                "CUBE_INTERNAL_API_URL and CUBE_WORKER_TOKEN must be configured together"
            )
        if entity_parts and not (internal_api_url and worker_token):
            raise RuntimeError(
                "CUBE_LOCAL_LIGHT_ENTITIES requires CUBE_INTERNAL_API_URL and "
                "CUBE_WORKER_TOKEN"
            )
        return cls(
            database_url=database_url,
            openai_api_key=api_key,
            artifact_root=Path(artifact_root).expanduser(),
            luna_model=luna_model,
            sol_model=sol_model,
            luna_max_output_tokens=luna_max_output_tokens,
            sol_max_output_tokens=sol_max_output_tokens,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            poll_seconds=poll_seconds,
            internal_api_url=internal_api_url or None,
            worker_token=worker_token or None,
            local_light_entities=frozenset(entity_parts),
        )


ChatModelFactory = Callable[..., Any]


def build_task_graph(
    *,
    store: PostgresStore,
    checkpointer: Any,
    api_key: str,
    luna_model: str = "gpt-6-luna",
    sol_model: str = "gpt-6.1-sol",
    luna_max_output_tokens: int | None = None,
    sol_max_output_tokens: int | None = None,
    search_client: AsyncOpenAI | None = None,
    chat_model_factory: ChatModelFactory = ChatOpenAI,
    local_tool_sender: LocalToolSender | None = None,
    allowed_light_entities: frozenset[str] = frozenset(),
) -> tuple[TaskGraph, AsyncOpenAI]:
    """Compose production model adapters without making a provider request."""

    if not api_key.strip() or not luna_model.strip() or not sol_model.strip():
        raise ValueError("worker API key and model names must not be blank")
    for ceiling in (luna_max_output_tokens, sol_max_output_tokens):
        if ceiling is not None and not 1 <= ceiling <= 128_000:
            raise ValueError("model output-token ceilings must be between 1 and 128000")
    client = search_client or AsyncOpenAI(api_key=api_key)
    common_model_options = {
        "use_responses_api": True,
        "output_version": "responses/v1",
        "reasoning": {"effort": "low"},
    }
    luna_options = dict(common_model_options)
    sol_options = dict(common_model_options)
    if luna_max_output_tokens is not None:
        luna_options["max_tokens"] = luna_max_output_tokens
    if sol_max_output_tokens is not None:
        sol_options["max_tokens"] = sol_max_output_tokens
    luna = chat_model_factory(model=luna_model, api_key=api_key, **luna_options)
    sol = chat_model_factory(model=sol_model, api_key=api_key, **sol_options)
    search = OpenAIWebSearch(
        client,
        model=luna_model,
        max_output_tokens=luna_max_output_tokens,
    )
    agent = SolDeepAgent(
        model=sol,
        search=search,
        checkpointer=checkpointer,
        model_profile=f"openai:{sol_model}",
    )
    tool_registry = None
    if local_tool_sender is not None or allowed_light_entities:
        if local_tool_sender is None:
            raise ValueError("local tools require a worker-to-device sender")
        tool_registry = ToolPolicyRegistry()
        LocalSystemControlsBridge(deliver=local_tool_sender).register_tools(tool_registry)
        if allowed_light_entities:
            HomeAssistantLocalBridge(
                allowed_light_entities=allowed_light_entities,
                deliver=local_tool_sender,
            ).register_tools(tool_registry)
    graph = TaskGraph(
        luna_model=luna,
        sol_model=sol,
        checkpointer=checkpointer,
        memory_provider=store.list_memories,
        memory_writer=store,
        complex_agent=agent,
        search_adapter=search,
        tool_registry=tool_registry,
        action_ledger=store if tool_registry is not None else None,
        timer_repository=store,
    )
    return graph, client


async def run_worker(settings: WorkerSettings) -> None:
    """Run lease-aware worker until SIGINT/SIGTERM; never migrate app schema."""

    store = PostgresStore(settings.database_url)
    await asyncio.to_thread(initialize_observability, "worker")
    await store.open()
    try:
        async with postgres_checkpointer(settings.database_url) as checkpointer:
            search_client = AsyncOpenAI(api_key=settings.openai_api_key)
            local_tool_client: WorkerDeviceToolClient | None = None
            try:
                if settings.internal_api_url and settings.worker_token:
                    local_tool_client = WorkerDeviceToolClient(
                        settings.internal_api_url,
                        settings.worker_token,
                    )
                graph, _ = build_task_graph(
                    store=store,
                    checkpointer=checkpointer,
                    api_key=settings.openai_api_key,
                    luna_model=settings.luna_model,
                    sol_model=settings.sol_model,
                    luna_max_output_tokens=settings.luna_max_output_tokens,
                    sol_max_output_tokens=settings.sol_max_output_tokens,
                    search_client=search_client,
                    local_tool_sender=(
                        local_tool_client.deliver if local_tool_client is not None else None
                    ),
                    allowed_light_entities=settings.local_light_entities,
                )
                worker = DurableJobWorker(
                    store=store,
                    graph=graph,
                    worker_id=settings.worker_id,
                    artifact_store=LocalArtifactStore(settings.artifact_root),
                    lease_seconds=settings.lease_seconds,
                    poll_seconds=settings.poll_seconds,
                )
                stop = asyncio.Event()
                _install_shutdown_handlers(stop)
                await worker.run(stop)
            finally:
                if local_tool_client is not None:
                    await local_tool_client.close()
                await search_client.close()
    finally:
        await store.close()
        await shutdown_observability()


def _install_shutdown_handlers(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_number, stop.set)
        except (NotImplementedError, RuntimeError):
            continue


def _int_setting(
    values: Mapping[str, str], name: str, default: int, *, minimum: int
) -> int:
    raw = values.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer") from None
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}")
    return value


def _float_setting(
    values: Mapping[str, str], name: str, default: float, *, minimum: float
) -> float:
    raw = values.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be numeric") from None
    if not math.isfinite(value) or value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}")
    return value


def _optional_token_limit(values: Mapping[str, str], name: str) -> int | None:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer") from None
    if not 1 <= value <= 128_000:
        raise RuntimeError(f"{name} must be between 1 and 128000")
    return value


def main() -> int:
    try:
        settings = WorkerSettings.from_environment()
        asyncio.run(run_worker(settings))
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"Worker stopped ({type(error).__name__}); details suppressed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
