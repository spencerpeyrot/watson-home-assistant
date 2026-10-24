"""Bounded PCM level counters; no audio is retained or exported."""

from array import array
from math import sqrt
import sys


class PcmLevelMeter:
    def __init__(self) -> None:
        self.frames = 0
        self.samples = 0
        self.energy = 0
        self.peak = 0
        self.clipped = 0

    def add(self, payload: bytes) -> None:
        samples = array("h")
        samples.frombytes(payload)
        if sys.byteorder != "little":
            samples.byteswap()
        self.frames += 1
        self.samples += len(samples)
        for sample in samples:
            magnitude = abs(sample)
            self.energy += sample * sample
            self.peak = max(self.peak, magnitude)
            self.clipped += int(magnitude >= 32760)

    def snapshot(self) -> dict[str, int | float]:
        return {
            "input_frames": self.frames,
            "pcm_rms": round(sqrt(self.energy / self.samples), 2) if self.samples else 0.0,
            "pcm_peak": self.peak,
            "pcm_clipped_samples": self.clipped,
        }
