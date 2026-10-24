"""Environment-configured durable API with selectable V1 and Muse voice engines."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

from fastapi import FastAPI

from service.artifacts import ArtifactStore, LocalArtifactStore
from service.auth import CredentialAuthenticator, authenticator_from_environment
from service.durable_api import create_durable_app
from service.device_events import DeviceEventHub
from service.elevenlabs_stt import ElevenLabsSTT
from service.elevenlabs_tts import ElevenLabsTTS
from service.live_provider import Connector, connect
from service.muse_gadget import MuseGadgetHost
from service.muse_provider import MuseConversationRegistry, muse_eleven_provider_factory
from service.postgres import PostgresStore
from service.system_controls import LocalSystemControlsBridge
from service.timers import TimerDeliveryWorker
from service.tool_policy import ToolPolicyRegistry
from service.voice_commands import (
    DecisionsClient,
    FallbackDecisionsClient,
    JevDecisionsClient,
    OpenAIDecisionsClient,
    VoiceCommandFlow,
)


DEFAULT_LIVE_INSTRUCTIONS = (
    "You are Cube, a concise voice assistant. Speak naturally. Use client delegation "
    "for tasks that need backend tools, research, or durable work. Do not claim a task "
    "or external action succeeded until a matching backend result confirms it."
    " Current-time questions are handled by the application's clock lookup. "
    "Wait for its verified clock result before answering; never guess the time "
    "from model knowledge. Do not delegate a clock-only question."
)


@dataclass(frozen=True)
class ServerSettings:
    database_url: str = field(repr=False)
    openai_api_key: str = field(repr=False)
    live_instructions: str
    voice_stack: str = "v1"
    artifact_root: Path | None = None
    live_model: str = "gpt-live-1"
    live_voice: str = "marin"
    idle_close_seconds: int = 30
    max_session_seconds: int | None = None
    auto_migrate: bool = False
    muse_sdk_token: str = field(default="", repr=False)
    typesafe_api_key: str = field(default="", repr=False)
    elevenlabs_api_key: str = field(default="", repr=False)
    elevenlabs_voice_id: str = ""
    elevenlabs_stt_model: str = "scribe_v2_realtime"
    elevenlabs_model: str = "eleven_flash_v2_5"
    muse_state_dir: Path = Path("/var/lib/cube-assistant/musegadget")

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> ServerSettings:
        values = os.environ if environ is None else environ
        database_url = values.get("DATABASE_URL", "")
        api_key = values.get("OPENAI_API_KEY", "")
        if not database_url.strip():
            raise RuntimeError("DATABASE_URL is required")

        voice_stack = values.get("CUBE_VOICE_STACK", "v1").strip().casefold()
        if voice_stack not in {"v1", "muse"}:
            raise RuntimeError("CUBE_VOICE_STACK must be v1 or muse")
        if voice_stack in {"v1", "muse"} and not api_key.strip():
            raise RuntimeError(
                "OPENAI_API_KEY is required for Luna Decisions fallback in muse mode "
                "and GPT-Live in v1 mode"
            )

        instructions = values.get("CUBE_LIVE_INSTRUCTIONS", DEFAULT_LIVE_INSTRUCTIONS)
        if voice_stack == "v1" and not instructions.strip():
            raise RuntimeError("CUBE_LIVE_INSTRUCTIONS must not be blank")
        model = values.get("OPENAI_LIVE_MODEL", "gpt-live-1")
        voice = values.get("OPENAI_LIVE_VOICE", "marin")
        if voice_stack == "v1" and (not model.strip() or not voice.strip()):
            raise RuntimeError("OpenAI Live model and voice must not be blank")

        idle_close = _int_setting(values, "CUBE_IDLE_CLOSE_SECONDS", 30, minimum=0)
        max_session = _optional_int_setting(values, "CUBE_MAX_SESSION_SECONDS")
        raw_auto_migrate = values.get("CUBE_AUTO_MIGRATE", "false").casefold()
        if raw_auto_migrate not in {"true", "false"}:
            raise RuntimeError("CUBE_AUTO_MIGRATE must be true or false")
        raw_artifact_root = values.get("CUBE_ARTIFACT_DIR", "")
        muse_sdk_token = values.get("MUSE_SDK_TOKEN", "")
        typesafe_api_key = values.get("TYPESAFE_API_KEY", "")
        elevenlabs_api_key = values.get("ELEVENLABS_API_KEY", "")
        elevenlabs_voice_id = values.get("ELEVENLABS_VOICE_ID", "").strip()
        elevenlabs_stt_model = values.get(
            "ELEVENLABS_STT_MODEL", "scribe_v2_realtime"
        ).strip()
        elevenlabs_model = values.get("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5").strip()
        raw_muse_state_dir = values.get(
            "MUSEGADGET_STATE_DIR", "/var/lib/cube-assistant/musegadget"
        ).strip()
        muse_state_dir = Path(raw_muse_state_dir).expanduser()
        if not muse_state_dir.is_absolute():
            raise RuntimeError("MUSEGADGET_STATE_DIR must be an absolute path")
        if voice_stack == "muse":
            if not muse_sdk_token.strip():
                raise RuntimeError("MUSE_SDK_TOKEN is required when CUBE_VOICE_STACK=muse")
            if re.fullmatch(r"mgst_[A-Za-z0-9_-]{42}[AEIMQUYcgkosw048]", muse_sdk_token.strip()) is None:
                raise RuntimeError("MUSE_SDK_TOKEN has an invalid format")
            if not elevenlabs_api_key.strip():
                raise RuntimeError("ELEVENLABS_API_KEY is required when CUBE_VOICE_STACK=muse")
            if not elevenlabs_voice_id or not elevenlabs_stt_model or not elevenlabs_model:
                raise RuntimeError(
                    "ELEVENLABS_VOICE_ID, ELEVENLABS_STT_MODEL, and "
                    "ELEVENLABS_TTS_MODEL must not be blank"
                )
            if elevenlabs_stt_model != "scribe_v2_realtime":
                raise RuntimeError("ELEVENLABS_STT_MODEL must be scribe_v2_realtime")
            if not typesafe_api_key.strip():
                raise RuntimeError("TYPESAFE_API_KEY is required when CUBE_VOICE_STACK=muse")
        return cls(
            database_url=database_url,
            openai_api_key=api_key,
            live_instructions=instructions,
            voice_stack=voice_stack,
            artifact_root=(
                Path(raw_artifact_root).expanduser() if raw_artifact_root.strip() else None
            ),
            live_model=model,
            live_voice=voice,
            idle_close_seconds=idle_close,
            max_session_seconds=max_session,
            auto_migrate=raw_auto_migrate == "true",
            muse_sdk_token=muse_sdk_token,
            typesafe_api_key=typesafe_api_key,
            elevenlabs_api_key=elevenlabs_api_key,
            elevenlabs_voice_id=elevenlabs_voice_id,
            elevenlabs_stt_model=elevenlabs_stt_model,
            elevenlabs_model=elevenlabs_model,
            muse_state_dir=muse_state_dir,
        )


StoreFactory = Callable[[str], PostgresStore]


def create_app(
    environ: Mapping[str, str] | None = None,
    *,
    store_factory: StoreFactory = PostgresStore,
    live_connector: Connector = connect,
    artifact_store: ArtifactStore | None = None,
    muse_host_factory: Callable[[str], MuseGadgetHost] | None = None,
    elevenlabs_connector: Callable[..., object] | None = None,
    elevenlabs_stt_connector: Callable[..., object] | None = None,
) -> FastAPI:
    """Build server from process settings; do not open DB or provider here.

    Deployments should inject secrets through the process environment or a
    secret manager. This factory never reads `.env` files or connects to OpenAI.
    """

    values = os.environ if environ is None else environ
    settings = ServerSettings.from_environment(values)
    authenticator: CredentialAuthenticator = authenticator_from_environment(values)
    store = store_factory(settings.database_url)
    configured_artifact_store = artifact_store
    if configured_artifact_store is None and settings.artifact_root is not None:
        configured_artifact_store = LocalArtifactStore(settings.artifact_root)
    muse_host: MuseGadgetHost | None = None
    muse_conversations: MuseConversationRegistry | None = None
    event_hub = DeviceEventHub()
    decisions_client: DecisionsClient | None = None
    voice_tool_registry: ToolPolicyRegistry | None = None
    background_services = ()
    if settings.voice_stack == "muse":
        muse_host = (
            muse_host_factory(settings.muse_sdk_token)
            if muse_host_factory is not None
            else MuseGadgetHost(
                settings.muse_sdk_token, state_dir=settings.muse_state_dir
            )
        )
        speech_options = {}
        if elevenlabs_connector is not None:
            speech_options["connector"] = elevenlabs_connector
        speech = ElevenLabsTTS(
            settings.elevenlabs_api_key,
            settings.elevenlabs_voice_id,
            model=settings.elevenlabs_model,
            **speech_options,
        )
        recognizer_options = {}
        if elevenlabs_stt_connector is not None:
            recognizer_options["connector"] = elevenlabs_stt_connector
        recognizer = ElevenLabsSTT(
            settings.elevenlabs_api_key,
            model=settings.elevenlabs_stt_model,
            **recognizer_options,
        )
        muse_conversations = MuseConversationRegistry(
            settings.muse_state_dir / "cube-conversations.sqlite3"
        )
        decisions_client = FallbackDecisionsClient(
            JevDecisionsClient(settings.typesafe_api_key),
            OpenAIDecisionsClient(settings.openai_api_key),
        )
        voice_tool_registry = ToolPolicyRegistry()
        LocalSystemControlsBridge(deliver=event_hub.deliver_local_tool).register_tools(
            voice_tool_registry
        )

        def internal_turn_factory(principal, session):
            assert decisions_client is not None and voice_tool_registry is not None
            return VoiceCommandFlow(
                decisions=decisions_client,
                principal=principal,
                session=session,
                tool_registry=voice_tool_registry,
                ledger=store,
            )

        live_factory = muse_eleven_provider_factory(
            muse_host=muse_host,
            speech=speech,
            recognizer=recognizer,
            conversation_registry=muse_conversations,
            internal_turn_factory=internal_turn_factory,
        )
        background_services = (muse_host.run,)
    else:
        from service.live_composition import openai_live_provider_factory

        live_factory = openai_live_provider_factory(
            api_key=settings.openai_api_key,
            model=settings.live_model,
            voice=settings.live_voice,
            instructions=settings.live_instructions,
            connector=live_connector,
        )
    timer_worker = (
        TimerDeliveryWorker(
            store=store,
            deliver=event_hub.deliver,
            worker_id=uuid4(),
        )
        if isinstance(store, PostgresStore) and settings.voice_stack == "v1"
        else None
    )
    app = create_durable_app(
        store=store,
        authenticate=authenticator.authenticate_http,
        authenticate_websocket=authenticator.authenticate_websocket,
        live_session_factory=live_factory,
        artifact_store=configured_artifact_store,
        timer_worker=timer_worker,
        device_event_hub=event_hub,
        idle_close_seconds=settings.idle_close_seconds,
        max_session_seconds=settings.max_session_seconds,
        manage_store=True,
        apply_migrations_on_startup=settings.auto_migrate,
        background_services=background_services,
    )
    app.state.store = store
    app.state.device_event_hub = event_hub
    app.state.timer_worker = timer_worker
    app.state.artifact_store = configured_artifact_store
    app.state.muse_host = muse_host
    app.state.muse_conversations = muse_conversations
    app.state.decisions_client = decisions_client
    app.state.voice_tool_registry = voice_tool_registry
    app.state.voice_stack = settings.voice_stack
    app.state.live_session_factory = live_factory
    return app


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


def _optional_int_setting(values: Mapping[str, str], name: str) -> int | None:
    raw = values.get(name, "")
    if not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer") from None
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value
