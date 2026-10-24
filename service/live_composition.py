"""Build one server-side GPT-Live provider instance per durable session."""

from __future__ import annotations

from typing import Any

from contracts.models import SessionResponse
from service.live_gateway import VoiceProvider, VoiceProviderFactory
from service.live_provider import Connector, OpenAILiveSession, connect


def openai_live_provider_factory(
    *,
    api_key: str,
    instructions: str,
    model: str = "gpt-live-1",
    voice: str = "marin",
    startup_timeout_seconds: float = 20,
    close_timeout_seconds: float = 15,
    connector: Connector = connect,
) -> VoiceProviderFactory:
    """Create configured Live providers without reading env files or making calls.

    Deployment passes the key from trusted server configuration. The returned
    factory creates isolated provider state for every authenticated session.
    """

    if not api_key.strip():
        raise ValueError("OpenAI API key must not be blank")
    if not model.strip() or not voice.strip() or not instructions.strip():
        raise ValueError("Live model, voice, and instructions must not be blank")
    if startup_timeout_seconds <= 0 or close_timeout_seconds <= 0:
        raise ValueError("Live lifecycle timeouts must be positive")

    def create_provider(_principal: Any, _session: SessionResponse) -> VoiceProvider:
        return OpenAILiveSession(
            api_key,
            model=model,
            voice=voice,
            instructions=instructions,
            startup_timeout_seconds=startup_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            connector=connector,
        )

    return create_provider
