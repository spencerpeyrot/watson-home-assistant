from __future__ import annotations

import unittest
from pathlib import Path

from device.config import device_state_dir


class DeviceConfigTests(unittest.TestCase):
    def test_explicit_device_state_directory_expands_home_and_trims_space(self) -> None:
        self.assertEqual(
            device_state_dir({"CUBE_DEVICE_DATA_DIR": " ~/cube-state "}),
            Path.home() / "cube-state",
        )

    def test_default_device_state_directory_uses_user_state_location(self) -> None:
        self.assertEqual(
            device_state_dir({}),
            Path.home() / ".local" / "state" / "cube-assistant",
        )


if __name__ == "__main__":
    unittest.main()
