"""ONNX-backed openWakeWord adapter for the Pi's 24 kHz PCM capture frames."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from fractions import Fraction
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


class OpenWakeWordDetector:
    """Adapt 20 ms mono PCM16 frames to openWakeWord's 80 ms, 16 kHz input."""

    MODEL_FRAME_DURATION_MS = 80
    MODEL_SAMPLE_RATE_HZ = 16_000

    def __init__(
        self,
        *,
        model_path: str | Path,
        melspectrogram_model_path: str | Path,
        embedding_model_path: str | Path,
        input_sample_rate_hz: int = 24_000,
        input_channels: int = 1,
        input_frame_duration_ms: int = 20,
        threshold: float = 0.5,
        confirmation_windows: int = 3,
    ) -> None:
        if input_sample_rate_hz <= 0 or input_channels != 1:
            raise ValueError("wake detection requires a positive sample rate and mono audio")
        if input_frame_duration_ms <= 0 or self.MODEL_FRAME_DURATION_MS % input_frame_duration_ms:
            raise ValueError("capture frame duration must divide the 80 ms model window")
        if input_sample_rate_hz * input_frame_duration_ms % 1000:
            raise ValueError("capture frame duration must align to whole PCM samples")
        if not 0 < threshold <= 1:
            raise ValueError("wake threshold must be greater than 0 and at most 1")
        if isinstance(confirmation_windows, bool) or not isinstance(confirmation_windows, int) or confirmation_windows < 1:
            raise ValueError("wake confirmation windows must be a positive integer")

        self._model_path = self._require_onnx_file(model_path, "wake-word")
        melspectrogram_path = self._require_onnx_file(
            melspectrogram_model_path, "mel-spectrogram"
        )
        embedding_path = self._require_onnx_file(
            embedding_model_path, "embedding"
        )

        try:
            import numpy as np
            from openwakeword.model import Model
            from scipy.signal import resample_poly
        except ImportError as error:
            raise RuntimeError(
                "wake-word dependencies are missing; install the wakeword group"
            ) from error

        self._np = np
        self._resample_poly = resample_poly
        self._model_name = self._model_path.stem
        self._threshold = threshold
        self._confirmation_windows = confirmation_windows
        self._qualifying_windows = 0
        self._last_prediction_at: float | None = None
        self._score_history: deque[float] = deque(maxlen=16)
        self._diagnostic_started = time.monotonic()
        self._diagnostic_windows = 0
        self._diagnostic_peak = 0.0
        self._diagnostic_near = 0
        self._diagnostic_triggers = 0
        self._last_near_log = 0.0
        self._last_detection_metadata: dict[str, float | int] = {}
        self._input_sample_rate_hz = input_sample_rate_hz
        self._input_frame_duration_ms = input_frame_duration_ms
        self._frames_per_window = self.MODEL_FRAME_DURATION_MS // input_frame_duration_ms
        self._frame_bytes = input_sample_rate_hz * input_frame_duration_ms // 1000 * 2
        sample_ratio = Fraction(self.MODEL_SAMPLE_RATE_HZ, input_sample_rate_hz)
        self._resample_up = sample_ratio.numerator
        self._resample_down = sample_ratio.denominator
        self._window = bytearray()
        self._model = Model(
            wakeword_models=[str(self._model_path)],
            inference_framework="onnx",
            melspec_model_path=str(melspectrogram_path),
            embedding_model_path=str(embedding_path),
            ncpu=1,
        )

    async def __call__(self, pcm_frame: bytes) -> bool:
        """Require consecutive qualifying 80 ms windows before triggering."""

        if len(pcm_frame) != self._frame_bytes:
            raise ValueError(f"PCM frame must contain exactly {self._frame_bytes} bytes")
        self._window.extend(pcm_frame)
        if len(self._window) < self._frame_bytes * self._frames_per_window:
            return False

        window = bytes(self._window)
        self._window.clear()
        return await asyncio.to_thread(self._predict, window)

    def _predict(self, pcm_window: bytes) -> bool:
        samples = self._np.frombuffer(pcm_window, dtype="<i2")
        audio_16khz = self._resample_poly(
            samples, self._resample_up, self._resample_down
        )
        expected_samples = self.MODEL_SAMPLE_RATE_HZ * self.MODEL_FRAME_DURATION_MS // 1000
        if audio_16khz.size != expected_samples:
            raise RuntimeError("wake-word resampling returned an unexpected window size")
        audio_16khz = self._np.clip(
            self._np.rint(audio_16khz), -32768, 32767
        ).astype(self._np.int16)
        predictions = self._model.predict(audio_16khz)
        score = float(predictions.get(self._model_name, 0.0))
        now = time.monotonic()
        # Evaluation pauses during startup/playback. Do not combine candidates
        # across a pause into one confirmed wake.
        if self._last_prediction_at is not None and now - self._last_prediction_at > 0.24:
            self._qualifying_windows = 0
        self._last_prediction_at = now
        self._qualifying_windows = (
            self._qualifying_windows + 1 if score >= self._threshold else 0
        )
        triggered = self._qualifying_windows >= self._confirmation_windows
        self._record_score(score, triggered=triggered)
        if not triggered:
            return False

        # Clear old wake audio before the voice session returns to idle capture.
        self._model.reset()
        self._score_history.clear()
        self._qualifying_windows = 0
        return True

    @property
    def last_detection_metadata(self) -> dict[str, float | int]:
        """Return numeric trigger diagnostics, never audio or recognized text."""
        return dict(self._last_detection_metadata)

    def _record_score(self, score: float, *, triggered: bool) -> None:
        now = time.monotonic()
        near_floor = self._threshold * 0.5
        self._score_history.append(round(score, 4))
        self._diagnostic_windows += 1
        self._diagnostic_peak = max(self._diagnostic_peak, score)
        self._diagnostic_near += int(score >= near_floor)
        self._diagnostic_triggers += int(triggered)
        if triggered:
            self._last_detection_metadata = {
                "wake_score": round(score, 4),
                "wake_threshold": self._threshold,
                "wake_history_peak": max(self._score_history),
                "wake_history_windows": len(self._score_history),
                "wake_confirmation_windows": self._confirmation_windows,
            }
        # At most one near-threshold record per second; every actual trigger
        # records the preceding 16 evaluated windows (up to 1.28 s of scores).
        # These are numeric model outputs only, not a microphone recording.
        if triggered or (score >= near_floor and now - self._last_near_log >= 1.0):
            _LOGGER.info(
                "wake score diagnostic trigger=%s score=%.4f threshold=%.4f "
                "confirmation=%d/%d history=%s",
                triggered, score, self._threshold, self._qualifying_windows,
                self._confirmation_windows, list(self._score_history),
            )
            self._last_near_log = now
        if now - self._diagnostic_started >= 30.0:
            _LOGGER.info(
                "wake score summary elapsed_seconds=%.1f windows=%d peak=%.4f "
                "near_floor=%.4f near_windows=%d triggers=%d",
                now - self._diagnostic_started, self._diagnostic_windows,
                self._diagnostic_peak, near_floor, self._diagnostic_near,
                self._diagnostic_triggers,
            )
            self._diagnostic_started = now
            self._diagnostic_windows = 0
            self._diagnostic_peak = 0.0
            self._diagnostic_near = 0
            self._diagnostic_triggers = 0

    @staticmethod
    def _require_onnx_file(path: str | Path, label: str) -> Path:
        resolved = Path(path).expanduser()
        if resolved.suffix.lower() != ".onnx" or not resolved.is_file():
            raise ValueError(f"{label} model must be an existing ONNX file")
        return resolved
