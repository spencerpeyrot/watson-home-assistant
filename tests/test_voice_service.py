from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from contracts.models import SessionResponse
from device.voice_service import compose_pi_voice_service
from voice.job_orchestrator import VoiceJobCoordinator


CONVERSATION_ID = UUID("00000000-0000-4000-8000-000000000d01")
SESSION_ID = UUID("00000000-0000-4000-8000-000000000d02")


class PiVoiceServiceCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_shares_job_links_and_closes_clients_without_starting_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            async def wake_detect(_payload: bytes) -> bool:
                return False

            def never_detect(_payload: bytes) -> bool:
                return False

            with patch.dict(os.environ, {"CUBE_DEVICE_DATA_DIR": directory}):
                service = await compose_pi_voice_service(
                    gateway_url="https://gateway.example",
                    device_id="cube-01",
                    device_token="t" * 40,
                    conversation_id=CONVERSATION_ID,
                    locale="en-US",
                    timezone_name="America/New_York",
                    capture_pcm="plughw:CARD=seeed2micvoicec,DEV=0",
                    playback_pcm="plughw:CARD=seeed2micvoicec,DEV=0",
                    wake_detect=wake_detect,
                    speech_detect=never_detect,
                    interrupt_detect=never_detect,
                    turn_end_detect=never_detect,
                )

            try:
                self.assertIs(
                    service.runtime._idle_notification_check,
                    service.job_notifications,
                )
                self.assertFalse(service.runtime._controller._speech_updates_idle)
                response = SessionResponse(
                    session_id=SESSION_ID,
                    conversation_id=CONVERSATION_ID,
                    state="ready",
                    opened_at=datetime.now(timezone.utc),
                    idle_close_seconds=30,
                )
                callbacks = service.runtime._session_callbacks_factory(response)
                self.assertIsInstance(callbacks, VoiceJobCoordinator)
                self.assertEqual(
                    callbacks._jobs._link_store._path,
                    str(service.job_notifications.link_database_path),
                )
                await callbacks.close()

                stop_event = asyncio.Event()
                stop_event.set()
                await service.run(stop_event)

                self.assertTrue(service.job_notifications._jobs._client.is_closed)
                self.assertTrue(service._session_client._client.is_closed)
                self.assertEqual(
                    service.job_notifications.link_database_path,
                    Path(directory) / "voice-job-links.sqlite3",
                )
            finally:
                if not service._closed:
                    stop_event = asyncio.Event()
                    stop_event.set()
                    await service.run(stop_event)
