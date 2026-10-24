"""Lease-aware worker for LangGraph task execution."""

from __future__ import annotations

import asyncio
import logging
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from contracts.models import ArtifactRef, UsageMeasurement
from service.artifacts import ArtifactStore
from service.job_graph import TaskGraph, TaskGraphResult
from service.postgres import ClaimedJob, LeaseLost, PostgresStore
from voice.observability import start_session


_LOGGER = logging.getLogger(__name__)


class DurableJobWorker:
    """Claim, execute, heartbeat, and persist one job result at a time."""

    def __init__(
        self,
        *,
        store: PostgresStore,
        graph: TaskGraph,
        worker_id: UUID,
        artifact_store: ArtifactStore | None = None,
        lease_seconds: int = 30,
        poll_seconds: float = 0.5,
    ) -> None:
        if lease_seconds < 1 or poll_seconds <= 0:
            raise ValueError("worker lease and poll intervals must be positive")
        self._store = store
        self._graph = graph
        self._worker_id = worker_id
        self._artifact_store = artifact_store
        self._lease_seconds = lease_seconds
        self._poll_seconds = poll_seconds

    async def run_once(self) -> bool:
        """Process one available job; return false when queue is empty."""

        job = await self._store.claim_next_job(
            worker_id=self._worker_id, lease_seconds=self._lease_seconds
        )
        if job is None:
            return False
        trace = start_session(job.request.session_id, "job.lifecycle", job_id=job.job_id,
                              delegation_id=job.request.delegation_id, attempt=job.attempt_count)
        trace.io(input=job.request.request_text)
        try:
            interrupted_usage: list[UsageMeasurement] = []
            with trace.activate():
                work = asyncio.create_task(
                    self._graph.run(
                        job.request,
                        job_id=job.job_id,
                        owner_id=job.owner_id,
                        on_partial_usage=interrupted_usage.append,
                    )
                )
            heartbeat = asyncio.create_task(self._heartbeat(job, work))
            persist_abandoned_usage = False
            try:
                done, _ = await asyncio.wait(
                    (work, heartbeat), return_when=asyncio.FIRST_COMPLETED
                )
                if heartbeat in done:
                    await heartbeat
                result = await work
                trace.io(output=result.response)
            except LeaseLost:
                trace.event("job.lease_lost", status="abandoned")
                persist_abandoned_usage = True
                return True
            except asyncio.CancelledError:
                trace.event("job.cancelled", status="cancelled")
                persist_abandoned_usage = True
                raise
            except Exception as error:
                trace.event("job.execution_failed", status="failed", error_type=type(error).__name__)
                measured_usage = getattr(error, "measured_usage", None)
                if not isinstance(measured_usage, UsageMeasurement):
                    measured_usage = None
                try:
                    await self._store.append_job_event(
                        owner_id=job.owner_id,
                        job_id=job.job_id,
                        kind="status",
                        status="failed",
                        summary="The assistant could not complete this task.",
                        error_code="INTERNAL_ERROR",
                        worker_id=self._worker_id,
                        measured_usage=measured_usage,
                        usage_provider="openai" if measured_usage is not None else None,
                    )
                except LeaseLost:
                    pass
                return True
            finally:
                for task in (work, heartbeat):
                    if not task.done():
                        task.cancel()
                results = await asyncio.gather(work, heartbeat, return_exceptions=True)
                if persist_abandoned_usage:
                    usage = _measured_usage_from_task_result(results[0])
                    if usage is None and interrupted_usage:
                        usage = interrupted_usage[-1]
                    if usage is not None:
                        try:
                            await self._store.record_usage_once(
                                owner_id=job.owner_id,
                                session_id=job.request.session_id,
                                job_id=job.job_id,
                                usage_id=uuid5(
                                    NAMESPACE_URL,
                                    f"cube-assistant/job-attempt-usage/"
                                    f"{job.job_id}/{job.attempt_count}/openai",
                                ),
                                provider="openai",
                                measurement=usage,
                            )
                        except Exception as error:
                            _LOGGER.warning(
                                "abandoned job usage persistence failed (%s)",
                                type(error).__name__,
                            )
            artifact_ref: ArtifactRef | None = None
            artifact_registered = False
            try:
                if result.report is not None:
                    artifact_ref = await self._store_report(job, result)
                    artifact_registered = True
                await self._store.append_job_event(
                    owner_id=job.owner_id,
                    job_id=job.job_id,
                    kind="result",
                    status="succeeded",
                    summary=f"[{result.route}] {result.response.strip()}"[:2000],
                    worker_id=self._worker_id,
                    artifact_refs=(artifact_ref,) if artifact_ref is not None else (),
                    sources=result.sources,
                    measured_usage=result.measured_usage,
                    usage_provider="openai" if result.measured_usage is not None else None,
                )
                trace.event("job.result_persisted", status="succeeded", route=result.route)
            except LeaseLost:
                # A stale worker must not publish a result after another worker reclaimed.
                if artifact_ref is not None:
                    await self._discard_report(job, artifact_ref, registered=artifact_registered)
            except Exception as error:
                trace.event("job.publication_failed", status="failed", error_type=type(error).__name__)
                if artifact_ref is not None:
                    await self._discard_report(job, artifact_ref, registered=artifact_registered)
                _LOGGER.warning("job result persistence failed (%s)", type(error).__name__)
                try:
                    await self._store.append_job_event(
                        owner_id=job.owner_id,
                        job_id=job.job_id,
                        kind="status",
                        status="failed",
                        summary="The assistant could not save this task result.",
                        error_code="INTERNAL_ERROR",
                        worker_id=self._worker_id,
                        measured_usage=result.measured_usage,
                        usage_provider=(
                            "openai" if result.measured_usage is not None else None
                        ),
                    )
                except LeaseLost:
                    pass
            return True
        finally:
            trace.end()

    async def _store_report(self, job: ClaimedJob, result: TaskGraphResult) -> ArtifactRef:
        if self._artifact_store is None:
            raise RuntimeError("artifact storage is required for a generated report")
        artifact_ref = ArtifactRef(
            artifact_id=uuid4(),
            label="Task report",
            media_type="text/markdown; charset=utf-8",
        )
        metadata = await self._artifact_store.put(
            job.owner_id, artifact_ref.artifact_id, result.report.encode("utf-8")
        )
        try:
            await self._store.register_artifact(
                owner_id=job.owner_id,
                job_id=job.job_id,
                artifact_ref=artifact_ref,
                storage_key=artifact_ref.artifact_id.hex,
                byte_size=metadata.byte_size,
                sha256=metadata.sha256,
            )
        except Exception:
            try:
                await self._artifact_store.delete(job.owner_id, artifact_ref.artifact_id)
            except Exception as cleanup_error:
                _LOGGER.warning("unregistered artifact cleanup failed (%s)", type(cleanup_error).__name__)
            raise
        return artifact_ref

    async def _discard_report(
        self, job: ClaimedJob, artifact_ref: ArtifactRef, *, registered: bool
    ) -> None:
        if self._artifact_store is None:
            return
        try:
            if registered:
                marked = await self._store.delete_artifact_metadata(
                    owner_id=job.owner_id, artifact_id=artifact_ref.artifact_id
                )
                if not marked:
                    return
            await self._artifact_store.delete(job.owner_id, artifact_ref.artifact_id)
        except Exception as error:
            _LOGGER.warning("unpublished artifact cleanup failed (%s)", type(error).__name__)

    async def run(self, stop_event: asyncio.Event) -> None:
        """Poll until stopped; lease expiry recovers work after process loss."""

        while not stop_event.is_set():
            if await self.run_once():
                continue
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._poll_seconds)
            except TimeoutError:
                pass

    async def _heartbeat(self, job: ClaimedJob, work: asyncio.Task) -> None:
        interval = max(0.1, self._lease_seconds / 3)
        while not work.done():
            await asyncio.sleep(interval)
            if work.done():
                return
            renewed = await self._store.renew_job_lease(
                job_id=job.job_id,
                worker_id=self._worker_id,
                lease_seconds=self._lease_seconds,
            )
            if not renewed:
                work.cancel()
                raise LeaseLost("job lease could not be renewed")


def _measured_usage_from_task_result(result) -> UsageMeasurement | None:
    """Extract confirmed usage from a result or an interrupted graph exception."""

    measured_usage = getattr(result, "measured_usage", None)
    if measured_usage is None and isinstance(result, TaskGraphResult):
        measured_usage = result.measured_usage
    return measured_usage if isinstance(measured_usage, UsageMeasurement) else None
