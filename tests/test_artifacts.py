from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from service.artifacts import ArtifactAlreadyExists, ArtifactNotFound, LocalArtifactStore


OWNER = UUID("00000000-0000-4000-8000-000000000040")
OTHER_OWNER = UUID("00000000-0000-4000-8000-000000000041")
ARTIFACT = UUID("00000000-0000-4000-8000-000000000042")


class LocalArtifactStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_atomic_immutable_owner_scoped_round_trip_and_delete(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cube-artifacts-") as temporary:
            store = LocalArtifactStore(Path(temporary) / "artifacts")
            content = b"short research artifact"

            metadata = await store.put(OWNER, ARTIFACT, content)

            self.assertEqual(metadata.byte_size, len(content))
            self.assertEqual(metadata.sha256, hashlib.sha256(content).hexdigest())
            self.assertEqual(await store.get(OWNER, ARTIFACT), content)
            with self.assertRaises(ArtifactNotFound):
                await store.get(OTHER_OWNER, ARTIFACT)
            self.assertTrue(await store.delete(OWNER, ARTIFACT))
            self.assertFalse(await store.delete(OWNER, ARTIFACT))

    async def test_artifact_id_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cube-artifacts-") as temporary:
            store = LocalArtifactStore(Path(temporary) / "artifacts")
            await store.put(OWNER, ARTIFACT, b"first")

            with self.assertRaises(ArtifactAlreadyExists):
                await store.put(OWNER, ARTIFACT, b"replacement")

            self.assertEqual(await store.get(OWNER, ARTIFACT), b"first")


if __name__ == "__main__":
    unittest.main()
