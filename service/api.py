"""Run Cube API with voice-socket keepalive timeouts disabled."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.port <= 65_535:
        parser.error("port must be from 1 to 65535")

    uvicorn.run(
        "service.server:create_app",
        factory=True,
        host=arguments.host,
        port=arguments.port,
        workers=1,
        # Voice sessions already exchange continuous PCM and own explicit
        # completion/error teardown. Transport keepalive timeouts can abort a
        # healthy long reply while ElevenLabs audio is still streaming.
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
