"""Capture HAT microphone while playing a local PCM16 speech sample through HAT."""

from __future__ import annotations

import argparse
import array
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any


RATE_HZ = 24_000
CHANNELS = 1
SAMPLE_WIDTH = 2
DEFAULT_PCM = "plughw:CARD=seeed2micvoicec,DEV=0"


class ProbeError(RuntimeError):
    """Sanitized probe failure."""


def _wav_info(path: Path) -> tuple[float, bytes]:
    try:
        with wave.open(str(path), "rb") as source:
            valid = (
                source.getnchannels() == CHANNELS
                and source.getsampwidth() == SAMPLE_WIDTH
                and source.getframerate() == RATE_HZ
                and source.getnframes() > 0
            )
            duration = source.getnframes() / RATE_HZ
            if not valid:
                raise ProbeError("input must be mono PCM16 WAV at 24 kHz")
            return duration, source.readframes(source.getnframes())
    except (OSError, wave.Error) as error:
        raise ProbeError(f"input WAV could not be read ({type(error).__name__})") from None


def _dbfs(samples: array.array[int]) -> float | None:
    if not samples:
        return None
    square_sum = sum(sample * sample for sample in samples)
    rms = math.sqrt(square_sum / len(samples))
    if rms == 0:
        return None
    return round(20 * math.log10(rms / 32768), 1)


def _measure_capture(path: Path, lead_in: float, playback_seconds: float) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as capture:
            if (
                capture.getnchannels() != CHANNELS
                or capture.getsampwidth() != SAMPLE_WIDTH
                or capture.getframerate() != RATE_HZ
            ):
                raise ProbeError("capture format differs from requested PCM16 mono 24 kHz")
            frames = capture.getnframes()
            samples = array.array("h", capture.readframes(frames))
    except (OSError, wave.Error) as error:
        raise ProbeError(f"capture WAV could not be read ({type(error).__name__})") from None
    if sys.byteorder == "big":
        samples.byteswap()

    def window(start_seconds: float, end_seconds: float) -> array.array[int]:
        start = max(0, min(len(samples), int(start_seconds * RATE_HZ)))
        end = max(start, min(len(samples), int(end_seconds * RATE_HZ)))
        return samples[start:end]

    baseline = window(0, lead_in)
    during = window(lead_in, lead_in + playback_seconds)
    after = window(lead_in + playback_seconds, len(samples) / RATE_HZ)
    peak = max((abs(sample) for sample in during), default=0)
    near_full_scale = sum(abs(sample) >= 32_000 for sample in during)
    full_scale = sum(abs(sample) >= 32_767 for sample in during)
    return {
        "capture_seconds": round(len(samples) / RATE_HZ, 2),
        "baseline_rms_dbfs": _dbfs(baseline),
        "playback_rms_dbfs": _dbfs(during),
        "tail_rms_dbfs": _dbfs(after),
        "playback_peak_dbfs": round(20 * math.log10(peak / 32768), 1) if peak else None,
        "playback_near_full_scale_samples": near_full_scale,
        "playback_full_scale_samples": full_scale,
        "playback_sample_count": len(during),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    audio_path = args.audio.resolve()
    playback_seconds, _ = _wav_info(audio_path)
    capture_seconds = math.ceil(args.lead_in + playback_seconds + args.tail)
    if capture_seconds < 1:
        raise ProbeError("capture duration must be at least one second")

    with tempfile.TemporaryDirectory(prefix="cube-p05-duplex-") as temp_dir:
        capture_path = Path(temp_dir) / "microphone-capture.wav"
        try:
            recorder = subprocess.Popen(
                [
                    "arecord",
                    "-q",
                    "-D",
                    args.capture_pcm,
                    "-f",
                    "S16_LE",
                    "-r",
                    str(RATE_HZ),
                    "-c",
                    str(CHANNELS),
                    "-d",
                    str(capture_seconds),
                    str(capture_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise ProbeError(f"arecord could not start ({type(error).__name__})") from None

        try:
            time.sleep(args.lead_in)
            if recorder.poll() is not None:
                raise ProbeError("arecord stopped before playback")
            try:
                playback = subprocess.run(
                    ["aplay", "-q", "-D", args.playback_pcm, str(audio_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=playback_seconds + 10,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise ProbeError(f"aplay failed ({type(error).__name__})") from None
            if playback.returncode:
                raise ProbeError(f"aplay exited with status {playback.returncode}")
            try:
                recorder.wait(timeout=capture_seconds + 5)
            except subprocess.TimeoutExpired:
                recorder.terminate()
                recorder.wait(timeout=5)
                raise ProbeError("arecord did not finish within its capture window") from None
            if recorder.returncode:
                raise ProbeError(f"arecord exited with status {recorder.returncode}")
        finally:
            if recorder.poll() is None:
                recorder.terminate()
                recorder.wait(timeout=5)

        metrics = _measure_capture(capture_path, args.lead_in, playback_seconds)
        if args.keep_capture:
            destination = args.keep_capture.resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(capture_path, destination)
            metrics["capture_path"] = str(destination)
        return {
            "schema_version": 1,
            "capture_pcm": args.capture_pcm,
            "playback_pcm": args.playback_pcm,
            "sample_rate_hz": RATE_HZ,
            "input_playback_seconds": round(playback_seconds, 2),
            "lead_in_seconds": args.lead_in,
            "tail_seconds_requested": args.tail,
            "mixer_changed": False,
            "capture_retained": args.keep_capture is not None,
            **metrics,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="mono PCM16 WAV at 24 kHz")
    parser.add_argument("--capture-pcm", default=DEFAULT_PCM)
    parser.add_argument("--playback-pcm", default=DEFAULT_PCM)
    parser.add_argument("--lead-in", type=float, default=0.5, help="quiet capture lead-in seconds")
    parser.add_argument("--tail", type=float, default=1.0, help="quiet capture tail seconds")
    parser.add_argument("--keep-capture", type=Path, help="optional path to retain microphone WAV")
    args = parser.parse_args()
    if args.lead_in < 0 or args.tail < 0:
        parser.error("lead-in and tail must be non-negative")
    try:
        result = _run(args)
    except ProbeError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
