"""Run Meta's Linux Muse Gadget SDK with command execution disabled."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any


_LOGGER = logging.getLogger(__name__)
MUSE_COMMANDS: dict[str, dict[str, Any]] = {}


class MuseGadgetHost:
    """Own one paired SDK connection and expose its active LinkSession."""

    def __init__(
        self, sdk_token: str, state_dir: str | Path = "/var/lib/cube-assistant/musegadget"
    ) -> None:
        if not sdk_token.strip():
            raise ValueError("Muse SDK token must not be blank")
        self._sdk_token = sdk_token
        self._state_dir = Path(state_dir)
        if not self._state_dir.is_absolute():
            raise ValueError("Muse state directory must be absolute")
        self._service: Any | None = None
        self._run_task: asyncio.Task[None] | None = None

    async def run(self, stop: asyncio.Event) -> None:
        """Maintain the SDK connection until application shutdown."""

        os.environ["MUSEGADGET_STATE_DIR"] = str(self._state_dir)
        try:
            service = _restricted_service(self._sdk_token)
        except ImportError:
            raise RuntimeError(
                "Muse voice stack requires the optional muse dependency group"
            ) from None
        self._service = service
        self._run_task = asyncio.create_task(service.run(), name="cube-muse-gadget")
        try:
            await stop.wait()
        finally:
            service.stop()
            await asyncio.gather(self._run_task, return_exceptions=True)
            self._service = None
            self._run_task = None

    async def wait_for_link(self, timeout_seconds: float = 20.0) -> Any:
        """Wait for a registered SDK link, hiding SDK internals from providers."""

        if timeout_seconds <= 0:
            raise ValueError("Muse link timeout must be positive")
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            service = self._service
            task = self._run_task
            if task is not None and task.done():
                raise RuntimeError("Muse gadget service stopped")
            link = getattr(service, "active_link", None) if service is not None else None
            if link is not None and link.registered.is_set():
                return link
            await asyncio.sleep(0.05)
        raise TimeoutError("Muse gadget did not connect before the startup deadline")


def _restricted_service(sdk_token: str) -> Any:
    """Load pinned SDK types lazily and advertise no local commands."""

    from musegadget import __version__
    from musegadget.identity import load_or_create
    from musegadget.link_client import DeviceDescription, LinkSession, Outcome
    from musegadget.service import DEFAULT_NOISE_HOST, Service

    class DenyExecutor:
        @staticmethod
        def run(_command: str, _params: dict, _timeout_ms: int | None = None) -> dict:
            return {"ok": False, "error": "command is not enabled for Cube voice"}

    class RestrictedService(Service):
        @property
        def active_link(self) -> Any | None:
            return self._current

        async def _session(self, vm: dict, pairing: dict) -> tuple[Any, float]:
            # Muse Service currently hardcodes the default shell/file command
            # schemas inside _session. Keep its reconnect and token-refresh loop,
            # but construct the pinned SDK LinkSession with an empty command set.
            device = DeviceDescription(
                node_id=self.identity.node_id,
                display_name=self.display_name,
                version=__version__,
                commands=MUSE_COMMANDS,
            )
            session = LinkSession(
                noise_host=pairing.get("noise_host") or DEFAULT_NOISE_HOST,
                vm_id=vm["vm_id"] or vm["vm_name"],
                vm_auth_token=vm["vm_auth_token"],
                device=device,
                run_command=self.executor.run,
            )
            started = time.monotonic()
            self._current = session
            try:
                outcome = await session.run(self._stop)
            except Exception as error:
                _LOGGER.warning("Muse gadget session failed (%s)", type(error).__name__)
                outcome = Outcome.CLOSED
            finally:
                self._current = None
            lasted = time.monotonic() - (session.registered_at or started)
            return outcome, lasted

    return RestrictedService(
        identity=load_or_create(),
        executor=DenyExecutor(),
        sdk_token=sdk_token,
        display_name="Cube Assistant",
    )
