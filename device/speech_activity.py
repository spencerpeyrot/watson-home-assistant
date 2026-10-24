"""Local WebRTC VAD candidate for 24 kHz Pi capture frames."""

from __future__ import annotations

from collections.abc import Callable


class WebRtcSpeechActivity:
    """Classify local speech and mark an endpoint after trailing silence.

    Capture uses mono PCM16 at 24 kHz in 20 ms frames. WebRTC VAD accepts
    32 kHz, so each frame is resampled locally before inference. This is a
    candidate policy; acoustic calibration is required before deployment.
    """

    INPUT_SAMPLE_RATE_HZ = 24_000
    VAD_SAMPLE_RATE_HZ = 32_000
    FRAME_DURATION_MS = 20

    def __init__(
        self,
        *,
        mode: int = 2,
        minimum_speech_ms: int = 60,
        endpoint_silence_ms: int = 600,
    ) -> None:
        if mode not in range(4):
            raise ValueError("WebRTC VAD mode must be from 0 to 3")
        if minimum_speech_ms < self.FRAME_DURATION_MS:
            raise ValueError("minimum speech must be at least one frame")
        if endpoint_silence_ms < self.FRAME_DURATION_MS:
            raise ValueError("endpoint silence must be at least one frame")
        if minimum_speech_ms % self.FRAME_DURATION_MS:
            raise ValueError("minimum speech must align to 20 ms frames")
        if endpoint_silence_ms % self.FRAME_DURATION_MS:
            raise ValueError("endpoint silence must align to 20 ms frames")

        try:
            import numpy as np
            import webrtcvad
            from scipy.signal import resample_poly
        except ImportError as error:
            raise RuntimeError(
                "speech activity dependencies are missing; install the wakeword group"
            ) from error

        self._np = np
        self._resample_poly: Callable[..., object] = resample_poly
        self._vad_mode = mode
        self._vad = webrtcvad.Vad(mode)
        self._frame_bytes = (
            self.INPUT_SAMPLE_RATE_HZ * 2 * self.FRAME_DURATION_MS // 1000
        )
        self._minimum_speech_frames = minimum_speech_ms // self.FRAME_DURATION_MS
        self._endpoint_silence_frames = endpoint_silence_ms // self.FRAME_DURATION_MS
        self._candidate_speech_frames = 0
        self._speech_open = False
        self._silence_frames = 0
        self._endpoint_pending = False
        self._cached_payload: bytes | None = None
        self._cached_endpoint_checked = True

    def __call__(self, payload: bytes) -> bool:
        """Return accepted speech after the configured onset threshold."""

        return self.speech_detect(payload)

    def speech_detect(self, payload: bytes) -> bool:
        """Callable matching `device.session_control.SpeechDetector`."""

        raw = self._validate_frame(payload)
        decision = self._classify(raw)
        self._cached_payload = raw
        self._cached_endpoint_checked = False
        return decision

    def turn_end_detect(self, payload: bytes) -> bool:
        """Consume the matching speech decision and emit each endpoint once."""

        raw = self._validate_frame(payload)
        if raw != self._cached_payload or self._cached_endpoint_checked:
            self._classify(raw)
        self._cached_payload = raw
        self._cached_endpoint_checked = True
        ended = self._endpoint_pending
        self._endpoint_pending = False
        return ended

    def reset(self) -> None:
        """Clear endpoint history after a session is discarded or replaced."""

        import webrtcvad

        self._vad = webrtcvad.Vad(self._vad_mode)
        self._candidate_speech_frames = 0
        self._speech_open = False
        self._silence_frames = 0
        self._endpoint_pending = False
        self._cached_payload = None
        self._cached_endpoint_checked = True

    def _validate_frame(self, payload: bytes) -> bytes:
        raw = bytes(payload)
        if len(raw) != self._frame_bytes:
            raise ValueError(f"PCM frame must contain exactly {self._frame_bytes} bytes")
        return raw

    def _classify(self, payload: bytes) -> bool:
        samples = self._np.frombuffer(payload, dtype="<i2")
        converted = self._resample_poly(samples, 4, 3)
        vad_samples = self._np.clip(
            self._np.rint(converted), -32768, 32767
        ).astype("<i2", copy=False)
        voiced = self._vad.is_speech(
            vad_samples.tobytes(), self.VAD_SAMPLE_RATE_HZ
        )

        if voiced:
            self._silence_frames = 0
            if not self._speech_open:
                self._candidate_speech_frames += 1
                if self._candidate_speech_frames >= self._minimum_speech_frames:
                    self._speech_open = True
                    self._candidate_speech_frames = 0
        else:
            self._candidate_speech_frames = 0
            if self._speech_open:
                self._silence_frames += 1
                if self._silence_frames >= self._endpoint_silence_frames:
                    self._speech_open = False
                    self._silence_frames = 0
                    self._endpoint_pending = True

        return self._speech_open
