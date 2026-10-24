from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from device.home_assistant import PiHomeAssistantExecutor
from device.local_tools import LocalToolInbox
from device.timer_client import TimerClientSettings, run


class TimerClientSettingsTests(unittest.IsolatedAsyncioTestCase):
    def test_reads_process_mapping_and_hides_device_token(self) -> None:
        token = "d" * 40
        settings = TimerClientSettings.from_environment(
            {
                "CUBE_GATEWAY_URL": "https://cube.example/",
                "CUBE_DEVICE_ID": "cube-01",
                "CUBE_DEVICE_TOKEN": token,
                "CUBE_TIMER_PLAYBACK_PCM": "plughw:CARD=seeed2micvoicec,DEV=0",
                "CUBE_DEVICE_DATA_DIR": "/tmp/cube-device-test",
            }
        )

        self.assertEqual(settings.gateway_url, "https://cube.example/")
        self.assertEqual(settings.device_id, "cube-01")
        self.assertEqual(settings.device_token, token)
        self.assertEqual(settings.playback_pcm, "plughw:CARD=seeed2micvoicec,DEV=0")
        self.assertEqual(settings.data_dir, Path("/tmp/cube-device-test"))
        self.assertNotIn(token, repr(settings))

    def test_optional_home_assistant_settings_are_scoped_and_redacted(self) -> None:
        device_token = "d" * 40
        ha_token = "h" * 40
        settings = TimerClientSettings.from_environment(
            {
                "CUBE_GATEWAY_URL": "https://cube.example",
                "CUBE_DEVICE_ID": "cube-01",
                "CUBE_DEVICE_TOKEN": device_token,
                "CUBE_TIMER_PLAYBACK_PCM": "hw:2,0",
                "CUBE_HOME_ASSISTANT_URL": "https://ha.example",
                "CUBE_HOME_ASSISTANT_TOKEN": ha_token,
                "CUBE_LOCAL_LIGHT_ENTITIES": "light.office, light.living_room",
            }
        )

        self.assertEqual(settings.home_assistant_url, "https://ha.example")
        self.assertEqual(settings.home_assistant_token, ha_token)
        self.assertEqual(
            settings.local_light_entities,
            frozenset({"light.office", "light.living_room"}),
        )
        self.assertNotIn(device_token, repr(settings))
        self.assertNotIn(ha_token, repr(settings))

    def test_home_assistant_settings_require_complete_valid_allowlist(self) -> None:
        base = {
            "CUBE_GATEWAY_URL": "https://cube.example",
            "CUBE_DEVICE_ID": "cube-01",
            "CUBE_DEVICE_TOKEN": "d" * 40,
            "CUBE_TIMER_PLAYBACK_PCM": "hw:2,0",
        }
        with self.assertRaisesRegex(RuntimeError, "must be configured together"):
            TimerClientSettings.from_environment(
                {**base, "CUBE_HOME_ASSISTANT_URL": "https://ha.example"}
            )
        complete = {
            **base,
            "CUBE_HOME_ASSISTANT_URL": "https://ha.example",
            "CUBE_HOME_ASSISTANT_TOKEN": "h" * 40,
        }
        with self.assertRaisesRegex(RuntimeError, "contains duplicates"):
            TimerClientSettings.from_environment(
                {**complete, "CUBE_LOCAL_LIGHT_ENTITIES": "light.office,light.office"}
            )
        with self.assertRaisesRegex(RuntimeError, "must contain light entity IDs"):
            TimerClientSettings.from_environment(
                {**complete, "CUBE_LOCAL_LIGHT_ENTITIES": "switch.office"}
            )

    async def test_run_composes_local_tool_inbox_and_closes_home_assistant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = TimerClientSettings.from_environment(
                {
                    "CUBE_GATEWAY_URL": "https://cube.example",
                    "CUBE_DEVICE_ID": "cube-01",
                    "CUBE_DEVICE_TOKEN": "d" * 40,
                    "CUBE_TIMER_PLAYBACK_PCM": "hw:2,0",
                    "CUBE_DEVICE_DATA_DIR": temporary,
                    "CUBE_HOME_ASSISTANT_URL": "https://ha.example",
                    "CUBE_HOME_ASSISTANT_TOKEN": "h" * 40,
                    "CUBE_LOCAL_LIGHT_ENTITIES": "light.office",
                }
            )
            client = AsyncMock()
            captured: dict[str, object] = {}

            async def fake_event_client(**kwargs: object) -> None:
                captured.update(kwargs)

            with (
                patch("device.timer_client.HomeAssistantClient", return_value=client) as factory,
                patch(
                    "device.timer_client.run_timer_event_client",
                    new=AsyncMock(side_effect=fake_event_client),
                ) as run_event_client,
            ):
                await run(settings)

            factory.assert_called_once_with(
                "https://ha.example",
                "h" * 40,
                allowed_light_entities=frozenset({"light.office"}),
            )
            run_event_client.assert_awaited_once()
            self.assertIsInstance(captured["local_tool_inbox"], LocalToolInbox)
            self.assertIsInstance(captured["execute_tool"], PiHomeAssistantExecutor)
            client.close.assert_awaited_once()

    def test_requires_explicit_pcm_and_tls_for_remote_gateway(self) -> None:
        base = {
            "CUBE_GATEWAY_URL": "https://cube.example",
            "CUBE_DEVICE_ID": "cube-01",
            "CUBE_DEVICE_TOKEN": "d" * 40,
        }
        with self.assertRaisesRegex(RuntimeError, "CUBE_TIMER_PLAYBACK_PCM"):
            TimerClientSettings.from_environment(base)

        with self.assertRaisesRegex(RuntimeError, "CUBE_GATEWAY_URL"):
            TimerClientSettings.from_environment(
                {
                    **base,
                    "CUBE_GATEWAY_URL": "http://gateway.example",
                    "CUBE_TIMER_PLAYBACK_PCM": "hw:2,0",
                }
            )

    def test_blank_data_dir_uses_user_state_directory(self) -> None:
        settings = TimerClientSettings.from_environment(
            {
                "CUBE_GATEWAY_URL": "https://cube.example",
                "CUBE_DEVICE_ID": "cube-01",
                "CUBE_DEVICE_TOKEN": "d" * 40,
                "CUBE_TIMER_PLAYBACK_PCM": "hw:2,0",
                "CUBE_DEVICE_DATA_DIR": "",
            }
        )

        self.assertEqual(settings.data_dir, Path.home() / ".local" / "state" / "cube-assistant")


if __name__ == "__main__":
    unittest.main()
