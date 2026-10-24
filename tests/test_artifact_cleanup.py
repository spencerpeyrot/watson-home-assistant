from __future__ import annotations

import unittest
from uuid import UUID

from service.artifact_cleanup import ArtifactCleanupWorker
from service.artifacts import ArtifactError


OWNER = UUID("00000000-0000-4000-8000-000000000701")
ARTIFACT_A = UUID("00000000-0000-4000-8000-000000000702")
ARTIFACT_B = UUID("00000000-0000-4000-8000-000000000703")


class FakeStore:
    def __init__(self, pending):
        self.pending = list(pending)
        self.purged = []

    async def list_deleted_artifacts(self, *, limit):
        return tuple(self.pending[:limit])

    async def purge_deleted_artifact(self, *, owner_id, artifact_id):
        assert owner_id == OWNER
        self.purged.append(artifact_id)
        self.pending.remove((owner_id, artifact_id))
        return True


class FakeArtifactStore:
    def __init__(self, failures=()):
        self.failures = set(failures)
        self.deleted = []

    async def delete(self, owner_id, artifact_id):
        assert owner_id == OWNER
        self.deleted.append(artifact_id)
        if artifact_id in self.failures:
            raise ArtifactError("storage unavailable")
        return True


class ArtifactCleanupWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_deletes_blobs_before_purging_tombstones(self) -> None:
        store = FakeStore(((OWNER, ARTIFACT_A), (OWNER, ARTIFACT_B)))
        artifacts = FakeArtifactStore()
        worker = ArtifactCleanupWorker(store=store, artifact_store=artifacts)

        cleaned = await worker.run_once()

        self.assertEqual(cleaned, 2)
        self.assertEqual(artifacts.deleted, [ARTIFACT_A, ARTIFACT_B])
        self.assertEqual(store.purged, [ARTIFACT_A, ARTIFACT_B])
        self.assertEqual(store.pending, [])

    async def test_storage_failure_keeps_tombstone_for_retry(self) -> None:
        store = FakeStore(((OWNER, ARTIFACT_A),))
        artifacts = FakeArtifactStore(failures=(ARTIFACT_A,))
        worker = ArtifactCleanupWorker(store=store, artifact_store=artifacts)

        cleaned = await worker.run_once()

        self.assertEqual(cleaned, 0)
        self.assertEqual(store.purged, [])
        self.assertEqual(store.pending, [(OWNER, ARTIFACT_A)])


if __name__ == "__main__":
    unittest.main()
