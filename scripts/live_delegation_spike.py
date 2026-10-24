"""Exercise GPT-Live client delegation with synthetic speech and local mock gateway."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from websockets.asyncio.client import ClientConnection, connect

from contracts.models import JobAccepted, SessionCreateRequest, TaskRequest
from scripts.check_openai_access import load_key
from voice.delegation import CommittedDelegation, build_task_request


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_UTTERANCE = (
    "Use the mock task service to calculate fourteen plus twenty-eight, then tell me the result."
)
SPEECH_FRAMES = 2400  # 100 ms at 24 kHz mono.
TRAILING_SILENCE_SECONDS = 1.0
TRANSCRIPT_QUIET_SECONDS = 1.0
OUTPUT_QUIET_SECONDS = 3.5


class SpikeError(RuntimeError):
    """A sanitized failure message safe to print."""


def _free_loopback_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(
    base_url: str,
    path: str,
    *,
    token: str | None = None,
    method: str = "GET",
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = Request(base_url + path, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=3) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        try:
            payload = json.loads(error.read())
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None
        return error.code, payload
    except (URLError, TimeoutError) as error:
        raise SpikeError(f"local mock gateway unavailable ({type(error).__name__})") from None


def _start_mock_gateway() -> tuple[subprocess.Popen[str], str, str, dict[str, str], Any]:
    port = _free_loopback_port()
    tokens = {
        "device": secrets.token_urlsafe(32),
        "owner": secrets.token_urlsafe(32),
        "operator": secrets.token_urlsafe(32),
    }
    # Do not pass the OpenAI key or unrelated user environment into the mock process.
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT),
        "CUBE_MOCK_DEVICE_ID": "cube-01",
        "CUBE_MOCK_DEVICE_TOKEN": tokens["device"],
        "CUBE_MOCK_OWNER_TOKEN": tokens["owner"],
        "CUBE_MOCK_OPERATOR_TOKEN": tokens["operator"],
    }
    log = tempfile.TemporaryFile(mode="w+t")
    process = subprocess.Popen(
        [sys.executable, "-m", "service.mock_server", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(80):
        if process.poll() is not None:
            process.wait()
            log.close()
            raise SpikeError("local mock gateway exited during startup")
        try:
            status, _ = _http_json(base_url, "/healthz")
            if status == 200:
                return process, base_url, tokens["device"], tokens, log
        except SpikeError:
            pass
        time.sleep(0.1)
    process.terminate()
    process.wait(timeout=5)
    log.close()
    raise SpikeError("local mock gateway did not become ready")


def _stop_mock_gateway(process: subprocess.Popen[str], tokens: dict[str, str], log: Any) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    log.seek(0)
    output = log.read()
    log.close()
    if any(token in output for token in tokens.values()):
        raise SpikeError("mock credential appeared in gateway logs")


def _create_synthetic_audio(directory: Path) -> Path:
    if sys.platform != "darwin":
        raise SpikeError("provide --audio on non-macOS hosts")
    source = directory / "synthetic-request.aiff"
    output = directory / "synthetic-request.wav"
    try:
        subprocess.run(
            ["say", "-o", str(source), SYNTHETIC_UTTERANCE],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@24000", "-c", "1", str(source), str(output)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SpikeError(f"could not create synthetic speech ({type(error).__name__})") from None
    return output


def _validate_audio(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            valid = (
                audio.getnchannels() == 1
                and audio.getsampwidth() == 2
                and audio.getframerate() == 24_000
                and audio.getnframes() > 0
            )
    except (OSError, wave.Error) as error:
        raise SpikeError(f"audio file could not be read ({type(error).__name__})") from None
    if not valid:
        raise SpikeError("audio must be mono PCM16 WAV at 24 kHz")


async def _read_events(connection: ClientConnection, queue: asyncio.Queue[Any]) -> None:
    try:
        async for message in connection:
            try:
                await queue.put((time.monotonic(), json.loads(message)))
            except (json.JSONDecodeError, TypeError):
                await queue.put(SpikeError("GPT-Live returned an invalid event"))
    except Exception:
        await queue.put(SpikeError("GPT-Live event stream ended unexpectedly"))
    finally:
        await queue.put(None)


async def _send_audio(connection: ClientConnection, path: Path) -> None:
    with wave.open(str(path), "rb") as audio:
        rate = audio.getframerate()
        while chunk := audio.readframes(SPEECH_FRAMES):
            if len(chunk) % 2:
                raise SpikeError("PCM16 chunk ended on a partial sample")
            await connection.send(
                json.dumps(
                    {
                        "type": "session.input_audio.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    },
                    separators=(",", ":"),
                )
            )
            await asyncio.sleep(len(chunk) / (rate * 2))
        silence_frames = int(rate * TRAILING_SILENCE_SECONDS / SPEECH_FRAMES)
        silence = bytes(SPEECH_FRAMES * 2)
        for _ in range(silence_frames):
            await connection.send(
                json.dumps(
                    {
                        "type": "session.input_audio.append",
                        "audio": base64.b64encode(silence).decode("ascii"),
                    },
                    separators=(",", ":"),
                )
            )
            await asyncio.sleep(SPEECH_FRAMES / rate)


def _event_payload(event: Any) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise SpikeError("GPT-Live returned an invalid event")
    return event


def _join_transcript_fragments(fragments: list[str]) -> str:
    transcript = ""
    for fragment in fragments:
        if transcript and transcript[-1] in ".!?" and fragment[:1].isupper():
            transcript += " "
        transcript += fragment
    return transcript.strip()


async def _send_event(connection: ClientConnection, payload: dict[str, Any]) -> None:
    await connection.send(json.dumps(payload, separators=(",", ":")))


def _make_task_request(
    transcript: str,
    delegation_id: str,
    offset_ms: int,
    session_id: uuid.UUID,
    conversation_id: uuid.UUID,
) -> TaskRequest:
    return build_task_request(
        CommittedDelegation(
            delegation_id=delegation_id,
            offset_ms=offset_ms,
            transcript=transcript,
            committed_at=datetime.now(timezone.utc),
        ),
        device_id="cube-01",
        conversation_id=conversation_id,
        session_id=session_id,
        context_revision=0,
        locale="en-US",
        timezone_name="America/New_York",
    )


def _post_task(base_url: str, token: str, body: TaskRequest) -> JobAccepted:
    status, payload = _http_json(
        base_url,
        "/v1/jobs",
        token=token,
        method="POST",
        body=body.model_dump(mode="json"),
    )
    if status != 202 or not isinstance(payload, dict):
        raise SpikeError(f"mock task service rejected request (HTTP {status})")
    try:
        return JobAccepted.model_validate(payload)
    except Exception:
        raise SpikeError("mock task service returned an invalid JobAccepted message") from None


async def _run_live_session(
    audio_path: Path,
    api_key: str,
    base_url: str,
    device_token: str,
    trace: dict[str, Any],
) -> None:
    conversation_id = uuid.uuid4()
    status, session_response = await asyncio.to_thread(
        _http_json,
        base_url,
        "/v1/sessions",
        token=device_token,
        method="POST",
        body=SessionCreateRequest(
            conversation_id=conversation_id,
            locale="en-US",
            timezone="America/New_York",
        ).model_dump(mode="json"),
    )
    if status != 201:
        raise SpikeError(f"mock session service rejected request (HTTP {status})")
    app_session_id = uuid.UUID(session_response["session_id"])
    trace["application_session_id"] = str(app_session_id)
    trace["conversation_id"] = str(conversation_id)

    session_config = {
        "model": "gpt-live-1",
        "instructions": (
            "This is a client-delegation integration test. Do not greet. "
            "For every user request, delegate to the client and say nothing while waiting. "
            "After the client returns, speak only this exact sentence: 'The mock task service accepted the request.' "
            "Do not imply that the requested task ran."
        ),
        "audio": {
            "format": {"type": "audio/pcm", "rate": 24_000},
            "output": {"voice": "marin"},
        },
        "delegation": {"type": "client"},
    }

    try:
        async with connect(
            "wss://api.openai.com/v1/live/sessions",
            additional_headers={"Authorization": f"Bearer {api_key}"},
            open_timeout=20,
            close_timeout=5,
            max_size=2**20,
        ) as connection:
            queue: asyncio.Queue[Any] = asyncio.Queue()
            reader = asyncio.create_task(_read_events(connection, queue))
            sender: asyncio.Task[None] | None = None
            try:
                await _send_event(
                    connection,
                    {
                        "type": "session.start",
                        "event_id": "p03_session_start",
                        "session": session_config,
                    },
                )
                while not trace.get("session_started"):
                    item = await asyncio.wait_for(queue.get(), timeout=20)
                    if item is None or isinstance(item, SpikeError):
                        raise SpikeError("GPT-Live closed before session.started")
                    _, event = item
                    payload = _event_payload(event)
                    event_type = payload.get("type", "unknown")
                    trace["event_order"].append(event_type)
                    if event_type == "error":
                        raise SpikeError("GPT-Live rejected session startup")
                    if event_type == "session.started":
                        trace["provider_session_id"] = payload.get("session", {}).get("id")
                        trace["session_started"] = True

                sender = asyncio.create_task(_send_audio(connection, audio_path))
                transcript_fragments: list[dict[str, Any]] = []
                delegation: dict[str, Any] | None = None
                last_input_delta = 0.0
                overall_deadline = time.monotonic() + 35
                while True:
                    if sender.done():
                        await sender
                    if (
                        sender.done()
                        and delegation
                        and last_input_delta
                        and time.monotonic() - last_input_delta >= TRANSCRIPT_QUIET_SECONDS
                    ):
                        break
                    remaining = overall_deadline - time.monotonic()
                    if remaining <= 0:
                        raise SpikeError("timed out waiting for transcript and client delegation")
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=min(0.3, remaining))
                    except TimeoutError:
                        continue
                    if item is None or isinstance(item, SpikeError):
                        raise SpikeError("GPT-Live event stream ended before delegation")
                    _, event = item
                    payload = _event_payload(event)
                    event_type = payload.get("type", "unknown")
                    trace["event_order"].append(event_type)
                    if event_type == "error":
                        raise SpikeError("GPT-Live rejected audio or delegation")
                    if event_type == "session.input_transcript.delta":
                        transcript_fragments.append(
                            {
                                "delta": payload.get("delta", ""),
                                "start_ms": payload.get("start_ms"),
                                "end_ms": payload.get("end_ms"),
                            }
                        )
                        last_input_delta = time.monotonic()
                    elif event_type == "session.delegation.created":
                        raw_delegation = payload.get("delegation", {})
                        if raw_delegation.get("target") == "client":
                            delegation = {
                                "id": raw_delegation.get("id"),
                                "target": raw_delegation.get("target"),
                                "offset_ms": payload.get("offset_ms"),
                            }
                if not delegation or not delegation.get("id") or delegation.get("offset_ms") is None:
                    raise SpikeError("no usable client delegation event received")

                # A client delegation can arrive before the last transcript fragment.
                # Use the complete, quieted single-turn audio transcript; never synthesize text from the prompt.
                transcript = "".join(fragment["delta"] for fragment in transcript_fragments).strip()
                if not transcript:
                    raise SpikeError("no input transcript arrived before delegation")
                if trace.get("synthetic_audio") and not transcript.rstrip(".!?").lower().endswith("result"):
                    raise SpikeError("synthetic speech transcript ended before its final word")
                trace["input_transcript"] = transcript
                trace["transcript_fragments"] = transcript_fragments
                trace["delegation"] = delegation

                task_request = _make_task_request(
                    transcript,
                    delegation["id"],
                    delegation["offset_ms"],
                    app_session_id,
                    conversation_id,
                )
                accepted = await asyncio.to_thread(
                    _post_task,
                    base_url,
                    device_token,
                    task_request,
                )
                trace["task_request"] = task_request.model_dump(mode="json")
                trace["job_accepted"] = accepted.model_dump(mode="json")
                result_text = "The mock task service accepted the request."
                output_fragments: list[str] = []
                result_acknowledged = False
                instruction_acknowledged = False
                output_quiet_at: float | None = None
                response_deadline = time.monotonic() + 18
                await _send_event(
                    connection,
                    {
                        "type": "session.commentary.append",
                        "event_id": "p03_mock_result",
                        "delegation_id": delegation["id"],
                        "content": result_text,
                    },
                )
                await _send_event(
                    connection,
                    {
                        "type": "session.instructions.append",
                        "event_id": "p03_speak_result",
                        "delegation_id": None,
                        "content": (
                            "The verified client result is ready. Immediately tell the user exactly: "
                            "'The mock task service accepted the request.' Do not imply the task ran."
                        ),
                    },
                )
                result_sent_at = time.monotonic()
                trace["result_sent"] = result_text
                trace["output_audio_delta_count"] = 0
                trace["output_audio_bytes"] = 0
                while time.monotonic() < response_deadline:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=0.3)
                    except TimeoutError:
                        if (
                            result_acknowledged
                            and output_fragments
                            and output_quiet_at
                            and time.monotonic() - output_quiet_at >= OUTPUT_QUIET_SECONDS
                        ):
                            break
                        continue
                    if item is None or isinstance(item, SpikeError):
                        break
                    received_at, event = item
                    payload = _event_payload(event)
                    event_type = payload.get("type", "unknown")
                    trace["event_order"].append(event_type)
                    if event_type == "error":
                        raise SpikeError("GPT-Live rejected the mock result")
                    if event_type == "session.commentary.appended" and payload.get("client_event_id") == "p03_mock_result":
                        result_acknowledged = True
                    elif event_type == "session.instructions.appended" and payload.get("client_event_id") == "p03_speak_result":
                        instruction_acknowledged = True
                    elif event_type == "session.output_transcript.delta":
                        if received_at >= result_sent_at:
                            output_fragments.append(payload.get("delta", ""))
                            output_quiet_at = time.monotonic()
                    elif event_type == "session.output_audio.delta":
                        if received_at >= result_sent_at:
                            audio_bytes = base64.b64decode(payload.get("delta", ""), validate=True)
                            trace["output_audio_delta_count"] += 1
                            trace["output_audio_bytes"] += len(audio_bytes)
                    elif event_type == "session.closed":
                        trace["session_closed"] = True
                        trace["final_usage"] = payload.get("usage")
                        break
                output_transcript = _join_transcript_fragments(output_fragments)
                if not result_acknowledged:
                    raise SpikeError("GPT-Live did not acknowledge the mock result")
                if not instruction_acknowledged:
                    raise SpikeError("GPT-Live did not acknowledge the result instruction")
                if "accept" not in output_transcript.lower():
                    detail = repr(output_transcript) if trace.get("synthetic_audio") else f"{len(output_transcript)} characters"
                    raise SpikeError(f"GPT-Live returned an incomplete assistant transcript ({detail})")
                if not trace["output_audio_bytes"]:
                    raise SpikeError("GPT-Live returned no audio for the mock result")
                trace["assistant_transcript"] = output_transcript
                trace["result_acknowledged"] = result_acknowledged
                trace["instruction_acknowledged"] = instruction_acknowledged

                if not trace.get("session_closed"):
                    await _send_event(
                        connection,
                        {"type": "session.close", "event_id": "p03_session_close"},
                    )
                    close_deadline = time.monotonic() + 15
                    while time.monotonic() < close_deadline:
                        try:
                            item = await asyncio.wait_for(queue.get(), timeout=0.5)
                        except TimeoutError:
                            continue
                        if item is None or isinstance(item, SpikeError):
                            break
                        _, event = item
                        payload = _event_payload(event)
                        event_type = payload.get("type", "unknown")
                        trace["event_order"].append(event_type)
                        if event_type == "session.closed":
                            trace["session_closed"] = True
                            trace["final_usage"] = payload.get("usage")
                            break
                if not trace.get("session_closed"):
                    raise SpikeError("GPT-Live session.closed was not received")
            finally:
                if sender is not None and not sender.done():
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
    except SpikeError:
        raise
    except Exception as error:
        raise SpikeError(f"GPT-Live spike failed ({type(error).__name__})") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, help="mono PCM16 WAV at 24 kHz; default uses synthetic macOS speech")
    parser.add_argument(
        "--trace-out",
        type=Path,
        default=ROOT / "docs" / "p03-delegation-trace.json",
    )
    args = parser.parse_args()
    trace_path = args.trace_out.resolve()
    if trace_path.exists():
        parser.error(f"trace already exists; choose another --trace-out: {trace_path}")
    api_key = load_key()
    if not api_key:
        parser.error("OPENAI_API_KEY missing; key value was not printed")

    trace: dict[str, Any] = {
        "schema_version": 1,
        "test": "synthetic_audio_client_delegation",
        "synthetic_audio": args.audio is None,
        "live_model": "gpt-live-1",
        "backend": "local authenticated mock gateway",
        "event_order": [],
        "raw_audio_retained": False,
    }
    with tempfile.TemporaryDirectory(prefix="cube-p03-live-") as temp_dir:
        audio_path = args.audio.resolve() if args.audio else _create_synthetic_audio(Path(temp_dir))
        _validate_audio(audio_path)
        process, base_url, device_token, tokens, log = _start_mock_gateway()
        try:
            asyncio.run(_run_live_session(audio_path, api_key, base_url, device_token, trace))
        finally:
            _stop_mock_gateway(process, tokens, log)
    del api_key
    trace["recorded_at"] = datetime.now(timezone.utc).isoformat()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("P03 client delegation passed.")
    print(f"Transcript: {trace['input_transcript']}")
    print(f"Mock job: {trace['job_accepted']['job_id']}; result returned to GPT-Live.")
    print(f"Assistant transcript: {trace['assistant_transcript']}")
    print(f"Sanitized trace: {trace_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SpikeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
