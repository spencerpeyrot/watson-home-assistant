from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.release_switch import switch_release


class ReleaseSwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.releases = self.root / "releases"
        self.releases.mkdir()
        (self.releases / "rev-a").mkdir()
        (self.releases / "rev-b").mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_first_switch_and_rollback_replace_symlink_atomically(self) -> None:
        self.assertIsNone(switch_release(self.root, "rev-a"))
        self.assertEqual(
            (self.root / "current").resolve(), (self.releases / "rev-a").resolve()
        )

        self.assertEqual(switch_release(self.root, "rev-b"), "rev-a")
        self.assertEqual(
            (self.root / "current").resolve(), (self.releases / "rev-b").resolve()
        )

        self.assertEqual(switch_release(self.root, "rev-a"), "rev-b")
        self.assertEqual(
            (self.root / "current").resolve(), (self.releases / "rev-a").resolve()
        )
        self.assertEqual(list(self.root.glob(".current-*.tmp")), [])

    def test_switch_to_current_release_is_idempotent(self) -> None:
        switch_release(self.root, "rev-a")

        self.assertEqual(switch_release(self.root, "rev-a"), "rev-a")
        self.assertEqual(
            (self.root / "current").resolve(), (self.releases / "rev-a").resolve()
        )

    def test_rejects_path_traversal_and_missing_release(self) -> None:
        with self.assertRaisesRegex(ValueError, "simple directory"):
            switch_release(self.root, "../outside")
        with self.assertRaisesRegex(ValueError, "missing or invalid"):
            switch_release(self.root, "rev-c")
        self.assertFalse((self.root / "current").exists())

    def test_refuses_regular_or_broken_current_path(self) -> None:
        current = self.root / "current"
        current.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not a symlink"):
            switch_release(self.root, "rev-a")
        self.assertEqual(current.read_text(encoding="utf-8"), "preserve")

        current.unlink()
        current.symlink_to("releases/missing", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "broken"):
            switch_release(self.root, "rev-a")


if __name__ == "__main__":
    unittest.main()
