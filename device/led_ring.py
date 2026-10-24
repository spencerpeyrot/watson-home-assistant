"""Low-brightness voice status patterns for the Pi 5 WS2812B ring."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from enum import StrEnum
from typing import Protocol


class VoiceIndicatorState(StrEnum):
    OFF = "off"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class VoiceIndicator(Protocol):
    def set_state(self, state: VoiceIndicatorState) -> None: ...

    def set_audio_level(self, level: float) -> None: ...

    async def run(self, stop_event: asyncio.Event) -> None: ...


_LOGGER = logging.getLogger(__name__)


class Pi5VoiceRing:
    """Drive eight GRB pixels on GPIO13 through Adafruit's Pi 5 PIO driver."""

    PIXEL_COUNT = 8
    MAX_BRIGHTNESS = 0.20
    FRAME_INTERVAL_SECONDS = 0.04

    def __init__(self) -> None:
        try:
            import board
            from adafruit_raspberry_pi5_neopixel_write import neopixel_write
        except ImportError as error:
            raise RuntimeError(
                "Pi 5 NeoPixel import failed; install the led dependency group "
                f"({type(error).__name__}: {error})"
            ) from error

        self._pin = board.D13
        self._write_pixels = neopixel_write
        self._state = VoiceIndicatorState.OFF
        self._audio_level = 0.0

    def set_state(self, state: VoiceIndicatorState) -> None:
        self._state = state
        if state != VoiceIndicatorState.SPEAKING:
            self._audio_level = 0.0

    def set_audio_level(self, level: float) -> None:
        """Set normalized live PCM RMS for speech cadence brightness."""

        self._audio_level = min(1.0, max(0.0, level)) if math.isfinite(level) else 0.0

    async def run(self, stop_event: asyncio.Event) -> None:
        """Render patterns until shutdown, then clear all pixels."""

        started_at = time.monotonic()
        first_frame = True
        try:
            while not stop_event.is_set():
                elapsed = time.monotonic() - started_at
                colors = self._frame(self._state, elapsed)
                await asyncio.to_thread(self._show, colors)
                if first_frame:
                    _LOGGER.info("voice status ring active on GPIO13 at 20% output")
                    first_frame = False
                try:
                    await asyncio.wait_for(
                        stop_event.wait(), timeout=self.FRAME_INTERVAL_SECONDS
                    )
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as error:
            _LOGGER.warning("voice status ring stopped (%s)", type(error).__name__)
        finally:
            try:
                await asyncio.to_thread(
                    self._show, [(0, 0, 0)] * self.PIXEL_COUNT
                )
            except Exception as error:
                _LOGGER.warning(
                    "voice status ring cleanup failed (%s)", type(error).__name__
                )

    def _frame(
        self, state: VoiceIndicatorState, elapsed: float
    ) -> list[tuple[int, int, int]]:
        if state == VoiceIndicatorState.OFF:
            return [(0, 0, 0)] * self.PIXEL_COUNT

        if state == VoiceIndicatorState.LISTENING:
            return [(255, 255, 255)] * self.PIXEL_COUNT

        if state == VoiceIndicatorState.THINKING:
            pulse = 0.08 + 0.92 * (0.5 - 0.5 * math.cos(elapsed * math.tau / 1.8))
            return [(round(255 * pulse),) * 3] * self.PIXEL_COUNT

        # Spoken PCM arrives in short frames. RMS makes brightness rise and
        # fall with syllables while the output cap keeps white dim.
        pulse = min(1.0, math.sqrt(self._audio_level / 0.18))
        return [(round(255 * pulse),) * 3] * self.PIXEL_COUNT

    def _show(self, colors: list[tuple[int, int, int]]) -> None:
        """Scale RGB channels and serialize in the ring's tested GRB order."""

        data = bytearray()
        for red, green, blue in colors:
            data.extend(
                (
                    round(green * self.MAX_BRIGHTNESS),
                    round(red * self.MAX_BRIGHTNESS),
                    round(blue * self.MAX_BRIGHTNESS),
                )
            )
        self._write_pixels(self._pin, data)
