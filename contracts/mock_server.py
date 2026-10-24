"""Serve one workstream's checked-in contract fixtures on localhost."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).parent
FIXTURES = ROOT / "fixtures"
SCHEMAS = ROOT / "schemas" / "v1"

PROFILES = {
    "device": (
        "audio_frame_metadata.json",
        "audio_output_frame_metadata.json",
        "device_health.json",
        "local_tool_request.json",
        "tool_result.uncertain_write.json",
        "tool_result.verified.json",
    ),
    "voice": (
        "audio_output_frame_metadata.json",
        "job_accepted.json",
        "job_event.progress.json",
        "session_create.json",
        "session_response.json",
        "task_request.json",
        "voice_context_append.json",
        "voice_session_context_append.json",
        "voice_session_event.closed.json",
        "voice_session_event.delegation.json",
        "voice_session_event.error.json",
        "voice_session_event.ready.json",
        "voice_session_event.transcript.json",
    ),
    "orchestrator": (
        "job_accepted.json",
        "job_correction.json",
        "job_event.progress.json",
        "job_snapshot.json",
        "task_request.json",
    ),
    "tools": (
        "api_error.json",
        "local_tool_request.json",
        "tool_result.uncertain_write.json",
        "tool_result.verified.json",
    ),
    "platform": (
        "api_error.json",
        "approval_decision.json",
        "approval_record.json",
        "health_response.json",
        "job_snapshot.json",
        "memory_record.json",
        "memory_write.json",
        "timer_create.json",
        "timer_fired_event.json",
        "timer_list_response.json",
        "timer_record.json",
        "usage_response.json",
    ),
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


class FixtureHandler(BaseHTTPRequestHandler):
    profile = ""

    def do_GET(self) -> None:
        path = unquote(urlsplit(self.path).path)
        if path == "/healthz":
            self._send(200, {"status": "healthy", "profile": self.profile})
            return
        if path == "/_mock/fixtures":
            self._send(200, {"profile": self.profile, "fixtures": PROFILES[self.profile]})
            return
        if path.startswith("/_mock/fixtures/"):
            name = path.removeprefix("/_mock/fixtures/")
            if name not in PROFILES[self.profile]:
                self._send(404, {"error": "fixture_not_available_for_profile"})
                return
            self._send_file(FIXTURES / name)
            return
        if path.startswith("/_mock/schemas/"):
            name = path.removeprefix("/_mock/schemas/") + ".schema.json"
            if Path(name).name != name:
                self._send(404, {"error": "schema_not_found"})
                return
            schema_path = SCHEMAS / name
            if not schema_path.is_file():
                self._send(404, {"error": "schema_not_found"})
                return
            self._send_file(schema_path)
            return
        self._send(404, {"error": "route_not_found"})

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send(404, {"error": "fixture_not_found"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        self.wfile.write(path.read_bytes())

    def _send(self, status: int, value: object) -> None:
        body = _json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    handler = type("ProfileFixtureHandler", (FixtureHandler,), {"profile": args.profile})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"{args.profile} fixture mock listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
