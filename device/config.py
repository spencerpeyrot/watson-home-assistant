"""Shared non-secret configuration for persistent Pi device state."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def device_state_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Resolve the durable device-data directory without loading dotenv files."""

    values = os.environ if environ is None else environ
    configured = values.get("CUBE_DEVICE_DATA_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "state" / "cube-assistant"
