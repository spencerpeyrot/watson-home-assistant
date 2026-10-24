"""Measure sustained PiAudioRuntime capture and silence playback without saving audio."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from device.alsa import AlsaPlayback, AudioDeviceError, PcmFormat
from device.audio_runtime import PiAudioRuntime


DEFAULT_PCM = "plughw:CARD=seeed2micvoicec,DEV=0"


class _CountingPlayback(AlsaPlayback):
    def __init__(self, pcm: str, audio_format: PcmFormat) -> None:
        super().__init__(pcm, audio_format)
        self.frames_written = 0

    async def write_frame(self, payload: bytes) -> None:
        await super().write_frame(payload)
        self.frames_written += 1


async def _run(*, capture_pcm: str, playback_pcm: str, duration_seconds: float) -> dict[str, object]:
    audio_format = PcmFormat()
    playback = _CountingPlayback(playback_pcm, audio_format)
    runtime = PiAudioRuntime(
        capture_pcm=capture_pcm,
        playback_pcm=playback_pcm,
        audio_format=audio_format,
        playback_stream=playback,
    )
    session_id = uuid4()
    capture_frames = 0
    capture_task: asyncio.Task[None] | None = None
    accepted_playback_frames = 0
    started_at: float | None = None

    async def drain_capture() -> None:
        nonlocal capture_frames
        while await runtime.next_capture_frame() is not None:
            # Deliberately discard frame bytes; never persist or inspect microphone audio.
            capture_frames += 1

    try:
        await runtime.start(session_id)
        capture_task = asyncio.create_task(drain_capture(), name="cube-soak-capture-drain")
        frame_seconds = audio_format.frame_duration_ms / 1000
        silence_frame = bytes(audio_format.frame_bytes)
        started_at = time.monotonic()
        deadline = started_at + duration_seconds
        next_frame_at = started_at

        while next_frame_at < deadline:
            if runtime.failure is not None:
                raise AudioDeviceError(runtime.failure)
            if await runtime.queue_playback_frame(session_id, silence_frame):
                accepted_playback_frames += 1
            else:
                raise AudioDeviceError("playback session closed during soak")
            next_frame_at += frame_seconds
            await asyncio.sleep(max(0, next_frame_at - time.monotonic()))

        # Let the last silent frames leave the bounded queue before cleanup.
        await asyncio.sleep(frame_seconds * 2)
        if runtime.failure is not None:
            raise AudioDeviceError(runtime.failure)
    finally:
        await runtime.stop(session_id)
        if capture_task is not None:
            await asyncio.gather(capture_task, return_exceptions=True)

    elapsed_seconds = time.monotonic() - started_at if started_at is not None else 0
    capture_stats, playback_stats = await runtime.stats()
    result: dict[str, object] = {
        "schema_version": 1,
        "capture_pcm": capture_pcm,
        "playback_pcm": playback_pcm,
        "sample_rate_hz": audio_format.sample_rate_hz,
        "channels": audio_format.channels,
        "frame_duration_ms": audio_format.frame_duration_ms,
        "requested_duration_seconds": duration_seconds,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "capture_frames_drained": capture_frames,
        "capture_frames_expected_approx": round(duration_seconds / (audio_format.frame_duration_ms / 1000)),
        "capture_queue_dropped_frames": capture_stats.dropped_overflow_frames,
        "capture_queue_remaining_frames": capture_stats.queued_frames,
        "playback_silence_frames_accepted": accepted_playback_frames,
        "playback_silence_frames_written": playback.frames_written,
        "playback_queue_dropped_frames": playback_stats.dropped_overflow_frames,
        "playback_queue_remaining_frames": playback_stats.queued_frames,
        "microphone_audio_persisted": False,
        "provider_calls": 0,
        "passed": (
            capture_frames > 0
            and playback.frames_written > 0
            and capture_stats.dropped_overflow_frames == 0
            and playback_stats.dropped_overflow_frames == 0
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-pcm", default=DEFAULT_PCM)
    parser.add_argument("--playback-pcm", default=DEFAULT_PCM)
    parser.add_argument("--duration-seconds", type=float, default=180)
    args = parser.parse_args()
    if not 1 <= args.duration_seconds <= 1800:
        parser.error("duration must be between 1 and 1800 seconds")
    try:
        result = asyncio.run(
            _run(
                capture_pcm=args.capture_pcm,
                playback_pcm=args.playback_pcm,
                duration_seconds=args.duration_seconds,
            )
        )
    except (AudioDeviceError, OSError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
