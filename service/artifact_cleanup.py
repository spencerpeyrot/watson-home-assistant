"""Retryable cleanup for blobs hidden by soft-deleted metadata."""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from service.artifacts import ArtifactError, ArtifactStore
from service.postgres import PostgresStore
from voice.observability import start_session

_LOGGER = logging.getLogger(__name__)


class ArtifactCleanupWorker:
    """Delete blobs first, then purge tombstones; retries are idempotent."""

    def __init__(
        self,
        *,
        store: PostgresStore,
        artifact_store: ArtifactStore,
        poll_seconds: float = 5.0,
        batch_size: int = 100,
    ) -> None:
        if poll_seconds <= 0 or not 1 <= batch_size <= 500:
            raise ValueError("artifact cleanup limits are invalid")
        self._store = store
        self._artifact_store = artifact_store
        self._poll_seconds = poll_seconds
        self._batch_size = batch_size

    async def run_once(self) -> int:
        """Clean one pending batch; retain tombstones when blob deletion fails."""

        trace = start_session(uuid4(), "artifact.cleanup", batch_size=self._batch_size)
        cleaned = 0
        try:
            pending = await self._store.list_deleted_artifacts(limit=self._batch_size)
            for owner_id, artifact_id in pending:
                with trace.operation("artifact.delete", artifact_id=artifact_id) as span:
                    try:
                        await self._artifact_store.delete(owner_id, artifact_id)
                    except ArtifactError as error:
                        span.update(status="failed", failure_stage="delete_blob",
                                    error_type=type(error).__name__, error_category="artifact_delete_failed")
                        continue
                    if await self._store.purge_deleted_artifact(
                        owner_id=owner_id, artifact_id=artifact_id
                    ):
                        cleaned += 1
                        span.update(status="succeeded")
                    else:
                        span.update(status="failed", failure_stage="purge_metadata",
                                    error_category="artifact_metadata_not_purged")
            trace.update(status="succeeded", cleanup_count=cleaned)
            return cleaned
        except Exception as error:
            trace.update(status="failed", failure_stage="list_or_purge",
                         error_type=type(error).__name__, error_category="artifact_cleanup_failed")
            raise
        finally:
            trace.end()

    async def run(self, stop_event: asyncio.Event) -> None:
        """Poll until stopped; unresolved tombstones remain eligible on restart."""

        while not stop_event.is_set():
            try:
                cleaned = await self.run_once()
            except Exception as error:
                _LOGGER.warning("artifact cleanup pass failed (%s)", type(error).__name__)
                cleaned = 0
            if cleaned:
                continue
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._poll_seconds)
            except TimeoutError:
                pass
