from __future__ import annotations

import unittest
from unittest.mock import patch

from service.api import main


class ApiRunnerTests(unittest.TestCase):
    @patch("service.api.uvicorn.run")
    def test_disables_transport_keepalive_cutoff_for_voice_sessions(self, run) -> None:
        self.assertEqual(main(["--host", "127.0.0.1", "--port", "8081"]), 0)

        run.assert_called_once_with(
            "service.server:create_app",
            factory=True,
            host="127.0.0.1",
            port=8081,
            workers=1,
            ws_ping_interval=None,
            ws_ping_timeout=None,
        )


if __name__ == "__main__":
    unittest.main()
