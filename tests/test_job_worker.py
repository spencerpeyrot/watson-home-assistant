from __future__ import annotations

import asyncio
import hashlib
import unittest
from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, UUID, uuid5

from contracts.models import JobSnapshot, SourceRef, TaskRequest, UsageMeasurement
from service.artifacts import ArtifactMetadata
from service.job_graph import JobGraphError, TaskGraphResult
from service.job_worker import DurableJobWorker
from service.postgres import ClaimedJob, LeaseLost


OWNER_ID = UUID("00000000-0000-4000-8000-000000000801")
JOB_ID = UUID("00000000-0000-4000-8000-000000000802")
WORKER_ID = UUID("00000000-0000-4000-8000-000000000803")


def make_claimed_job() -> ClaimedJob:
    now = datetime.now(timezone.utc)
    request = TaskRequest(
        device_id="cube-01",
        conversation_id=UUID("00000000-0000-4000-8000-000000000804"),
        session_id=UUID("00000000-0000-4000-8000-000000000805"),
        turn_id=UUID("00000000-0000-4000-8000-000000000806"),
        delegation_id="delegation-report",
        trace_id=UUID("00000000-0000-4000-8000-000000000807"),
        request_text="Research and report the answer.",
        context_revision=0,
        locale="en-US",
        timezone="America/New_York",
        idempotency_key="report-job",
    )
    return ClaimedJob(
        job_id=JOB_ID,
        owner_id=OWNER_ID,
        worker_id=WORKER_ID,
        request=request,
        snapshot=JobSnapshot(
            job_id=JOB_ID,
            status="running",
            context_revision=0,
            last_sequence=1,
            summary="Working.",
        ),
        lease_expires_at=now + timedelta(seconds=30),
        attempt_count=1,
    )


class FakeGraph:
    async def run(self, *_args, **_kwargs):
        return TaskGraphResult(
            route="complex",
            response="Report completed.",
            report="# Report\n\nSaved findings.",
            sources=(SourceRef(title="Source", url="https://example.com"),),
            measured_usage=UsageMeasurement(input_tokens=31, output_tokens=12),
        )


class FailedGraph:
    async def run(self, *_args, **_kwargs):
        raise JobGraphError(
            "model task failed",
            measured_usage=UsageMeasurement(input_tokens=17, output_tokens=4),
        )


class CancelledGraph:
    def __init__(self):
        self.started = asyncio.Event()

    async def run(self, *_args, on_partial_usage=None, **_kwargs):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            measured_usage = UsageMeasurement(input_tokens=29, output_tokens=7)
            error.measured_usage = measured_usage
            if on_partial_usage is not None:
                on_partial_usage(measured_usage)
            raise


class FakeArtifactStore:
    def __init__(self):
        self.blobs: dict[tuple[UUID, UUID], bytes] = {}

    async def put(self, owner_id, artifact_id, content):
        self.blobs[(owner_id, artifact_id)] = bytes(content)
        return ArtifactMetadata(
            byte_size=len(content), sha256=hashlib.sha256(content).hexdigest()
        )

    async def get(self, owner_id, artifact_id):
        return self.blobs[(owner_id, artifact_id)]

    async def delete(self, owner_id, artifact_id):
        return self.blobs.pop((owner_id, artifact_id), None) is not None


class FakeJobStore:
    def __init__(self, *, lose_result_lease: bool = False, lose_lease: bool = False):
        self.job = make_claimed_job()
        self.events: list[dict] = []
        self.artifacts: list[dict] = []
        self.deleted_artifacts: list[UUID] = []
        self.lose_result_lease = lose_result_lease
        self.lose_lease = lose_lease
        self.usage_records: list[dict] = []

    async def claim_next_job(self, **_kwargs):
        job, self.job = self.job, None
        return job

    async def renew_job_lease(self, **_kwargs):
        return not self.lose_lease

    async def record_usage_once(self, **kwargs):
        self.usage_records.append(kwargs)

    async def register_artifact(self, **kwargs):
        self.artifacts.append(kwargs)

    async def delete_artifact_metadata(self, **_kwargs):
        self.deleted_artifacts.append(_kwargs["artifact_id"])
        return True

    async def append_job_event(self, **kwargs):
        if self.lose_result_lease and kwargs.get("status") == "succeeded":
            raise LeaseLost("stale worker")
        self.events.append(kwargs)


class DurableJobWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_graph_persists_confirmed_partial_usage(self):
        store = FakeJobStore()
        worker = DurableJobWorker(store=store, graph=FailedGraph(), worker_id=WORKER_ID)

        self.assertTrue(await worker.run_once())

        self.assertEqual(store.events[0]["status"], "failed")
        self.assertEqual(
            store.events[0]["measured_usage"],
            UsageMeasurement(input_tokens=17, output_tokens=4),
        )
        self.assertEqual(store.events[0]["usage_provider"], "openai")

    async def test_lost_lease_persists_confirmed_usage_from_cancelled_graph(self):
        store = FakeJobStore(lose_lease=True)
        graph = CancelledGraph()
        worker = DurableJobWorker(
            store=store, graph=graph, worker_id=WORKER_ID, lease_seconds=1
        )

        self.assertTrue(await worker.run_once())

        self.assertEqual(len(store.usage_records), 1)
        record = store.usage_records[0]
        self.assertEqual(record["owner_id"], OWNER_ID)
        self.assertEqual(record["job_id"], JOB_ID)
        self.assertEqual(record["session_id"], make_claimed_job().request.session_id)
        self.assertEqual(record["provider"], "openai")
        self.assertEqual(record["measurement"], UsageMeasurement(input_tokens=29, output_tokens=7))
        self.assertEqual(
            record["usage_id"],
            uuid5(
                NAMESPACE_URL,
                f"cube-assistant/job-attempt-usage/{JOB_ID}/1/openai",
            ),
        )

    async def test_worker_shutdown_persists_confirmed_usage_before_propagating_cancel(self):
        store = FakeJobStore()
        graph = CancelledGraph()
        worker = DurableJobWorker(store=store, graph=graph, worker_id=WORKER_ID)
        run = asyncio.create_task(worker.run_once())
        await asyncio.wait_for(graph.started.wait(), timeout=2)
        run.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await run

        self.assertEqual(len(store.usage_records), 1)
        self.assertEqual(
            store.usage_records[0]["measurement"],
            UsageMeasurement(input_tokens=29, output_tokens=7),
        )

    async def test_persists_report_then_emits_sources_artifacts_and_usage(self) -> None:
        store = FakeJobStore()
        artifacts = FakeArtifactStore()
        worker = DurableJobWorker(
            store=store,
            graph=FakeGraph(),
            worker_id=WORKER_ID,
            artifact_store=artifacts,
        )

        self.assertTrue(await worker.run_once())

        self.assertEqual(len(store.artifacts), 1)
        registered = store.artifacts[0]
        ref = registered["artifact_ref"]
        self.assertEqual(registered["owner_id"], OWNER_ID)
        self.assertEqual(registered["job_id"], JOB_ID)
        self.assertEqual(artifacts.blobs[(OWNER_ID, ref.artifact_id)], b"# Report\n\nSaved findings.")
        event = store.events[0]
        self.assertEqual(event["status"], "succeeded")
        self.assertEqual(event["artifact_refs"], (ref,))
        self.assertEqual(str(event["sources"][0].url), "https://example.com/")
        self.assertEqual(event["measured_usage"].input_tokens, 31)
        self.assertEqual(event["usage_provider"], "openai")

    async def test_report_without_artifact_store_fails_closed(self) -> None:
        store = FakeJobStore()
        worker = DurableJobWorker(store=store, graph=FakeGraph(), worker_id=WORKER_ID)

        with self.assertLogs("service.job_worker", level="WARNING"):
            self.assertTrue(await worker.run_once())

        self.assertEqual(store.artifacts, [])
        self.assertEqual(store.events[0]["status"], "failed")
        self.assertEqual(store.events[0]["error_code"], "INTERNAL_ERROR")

    async def test_stale_worker_tombstones_and_removes_unpublished_report(self) -> None:
        store = FakeJobStore(lose_result_lease=True)
        artifacts = FakeArtifactStore()
        worker = DurableJobWorker(
            store=store,
            graph=FakeGraph(),
            worker_id=WORKER_ID,
            artifact_store=artifacts,
        )

        self.assertTrue(await worker.run_once())

        self.assertEqual(store.events, [])
        self.assertEqual(len(store.deleted_artifacts), 1)
        self.assertEqual(artifacts.blobs, {})


if __name__ == "__main__":
    unittest.main()
