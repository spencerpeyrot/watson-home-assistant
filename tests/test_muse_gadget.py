from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from musegadget.identity import Identity

from service.muse_gadget import MUSE_COMMANDS, MuseGadgetHost, _restricted_service


class MuseGadgetHostTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_for_link_requires_registration(self):
        host = MuseGadgetHost("sdk-token")
        registered = asyncio.Event()
        link = SimpleNamespace(registered=registered)
        host._service = SimpleNamespace(active_link=link)

        with self.assertRaises(TimeoutError):
            await host.wait_for_link(0.01)
        registered.set()
        self.assertIs(await host.wait_for_link(0.1), link)

    async def test_sdk_service_advertises_no_host_commands_and_denies_exec(self):
        identity = Identity("02:00:00:00:00:01")
        with patch("musegadget.identity.load_or_create", return_value=identity):
            service = _restricted_service("sdk-token")

        self.assertEqual(MUSE_COMMANDS, {})
        result = service.executor.run("system.run", {"command": "touch /tmp/nope"})
        self.assertFalse(result["ok"])
        self.assertIn("not enabled", result["error"])


if __name__ == "__main__":
    unittest.main()
