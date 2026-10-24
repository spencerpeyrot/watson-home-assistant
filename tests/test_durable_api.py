from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import Request
from fastapi.testclient import TestClient

from contracts.models import (
    JobSnapshot,
    LocalToolRequest,
    MemoryRecord,
    MemoryWriteRequest,
    TaskRequest,
    ToolResult,
    UsageSummary,
)
from service.durable_api import AuthenticatedPrincipal, create_durable_app
from service.device_events import DeviceEventUnavailable
from service.postgres import _digest_json


OWNER_ID = UUID("00000000-0000-4000-8000-000000000101")
JOB_ID = UUID("00000000-0000-4000-8000-000000000102")
JOB_REQUEST = TaskRequest(
    device_id="cube-01",
    conversation_id=UUID("00000000-0000-4000-8000-000000000103"),
    session_id=UUID("00000000-0000-4000-8000-000000000104"),
    turn_id=UUID("00000000-0000-4000-8000-000000000105"),
    delegation_id="delegation-internal-api",
    trace_id=UUID("00000000-0000-4000-8000-000000000106"),
    request_text="Turn on desk light.",
    context_revision=0,
    locale="en-US",
    timezone="America/New_York",
    idempotency_key="internal-tool-test",
)


class FakeStore:
    def __init__(self) -> None:
        self.correction_owner: UUID | None = None
        self.correction_key: str | None = None
        self.event_owner: UUID | None = None
        self.health_error = False
        self.memory_owner: UUID | None = None
        self.memory_idempotency_key: str | None = None
        self.memory: MemoryRecord | None = None
        self.usage_owner: UUID | None = None
        self.usage_period_start = None
        self.usage_period_end = None
        self.sessions_owner: UUID | None = None
        self.jobs_owner: UUID | None = None
        self.job_device_id = "cube-01"
        self.job_read_owner: UUID | None = None
        self.job_cancel_owner: UUID | None = None
        self.closed_session_owner: UUID | None = None
        self.closed_session_device: str | None = None

    async def get_job_request(self, *, owner_id, job_id):
        return JOB_REQUEST if job_id == JOB_ID and owner_id == OWNER_ID else None

    async def healthcheck(self) -> bool:
        if self.health_error:
            raise RuntimeError("database internals stay private")
        return True

    async def correct_job(self, *, owner_id, job_id, request, idempotency_key):
        self.correction_owner = owner_id
        self.correction_key = idempotency_key
        return JobSnapshot(
            job_id=job_id,
            status="queued",
            context_revision=request.context_revision,
            last_sequence=3,
            summary="Correction accepted and requeued.",
        )

    async def get_job_device_id(self, *, owner_id, job_id):
        return self.job_device_id if job_id == JOB_ID else None

    async def get_job(self, *, owner_id, job_id):
        self.job_read_owner = owner_id
        if job_id != JOB_ID:
            return None
        return JobSnapshot(
            job_id=job_id,
            status="running",
            context_revision=0,
            last_sequence=1,
            summary="Working.",
        )

    async def cancel_job(self, *, owner_id, job_id, idempotency_key):
        self.job_cancel_owner = owner_id
        self.correction_key = idempotency_key
        return JobSnapshot(
            job_id=job_id,
            status="cancelled",
            context_revision=0,
            last_sequence=2,
            summary="Cancelled.",
        )

    async def list_job_events(self, *, owner_id, job_id):
        return ()

    async def list_timers(self, *, owner_id):
        raise AssertionError("operator must be denied before repository access")

    async def list_sessions(self, *, owner_id, limit=100):
        self.sessions_owner = owner_id
        return ()

    async def list_jobs(self, *, owner_id, limit=100):
        self.jobs_owner = owner_id
        return ()

    async def close_session(self, *, owner_id, session_id, device_id=None):
        self.closed_session_owner = owner_id
        self.closed_session_device = device_id
        return False

    async def list_memories(self, *, owner_id):
        self.memory_owner = owner_id
        return (self.memory,) if self.memory is not None else ()

    async def write_memory(self, *, owner_id, request, idempotency_key):
        from datetime import datetime, timezone

        from uuid import uuid4

        self.memory_owner = owner_id
        self.memory_idempotency_key = idempotency_key
        now = datetime.now(timezone.utc)
        self.memory = MemoryRecord(
            memory_id=uuid4(),
            category=request.category,
            content=request.content,
            source_turn_id=request.source_turn_id,
            created_at=now,
            updated_at=now,
        )
        return self.memory, False

    async def update_memory_idempotent(
        self, *, owner_id, memory_id, request, idempotency_key
    ):
        from datetime import datetime, timezone

        self.memory_owner = owner_id
        self.memory_idempotency_key = idempotency_key
        if self.memory is None or self.memory.memory_id != memory_id:
            return None, False
        self.memory = self.memory.model_copy(
            update={
                "content": request.content,
                "source_turn_id": request.source_turn_id,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        return self.memory, False

    async def delete_memory_idempotent(self, *, owner_id, memory_id, idempotency_key):
        self.memory_owner = owner_id
        self.memory_idempotency_key = idempotency_key
        if self.memory is not None and self.memory.memory_id == memory_id:
            self.memory = None

    async def list_owner_job_events(self, *, owner_id, after_job_id=None, after_sequence=None):
        self.event_owner = owner_id
        return ()

    async def list_usage(self, *, owner_id, session_id=None):
        self.usage_owner = owner_id
        return ()

    async def summarize_usage(self, *, owner_id, period_start, period_end):
        self.usage_owner = owner_id
        self.usage_period_start = period_start
        self.usage_period_end = period_end
        return UsageSummary(period_start=period_start)


async def authenticate(request: Request) -> AuthenticatedPrincipal:
    token = request.headers.get("Authorization", "")
    if token == "Bearer operator-test":
        return AuthenticatedPrincipal(role="operator", owner_id=None)
    if token == "Bearer owner-test":
        return AuthenticatedPrincipal(role="owner", owner_id=OWNER_ID)
    if token == "Bearer worker-test":
        return AuthenticatedPrincipal(role="worker", owner_id=OWNER_ID, device_id="cube-01")
    return AuthenticatedPrincipal(role="device", owner_id=OWNER_ID, device_id="cube-01")


class FakeEventHub:
    def __init__(self, *, fail: bool = False):
        self.request: LocalToolRequest | None = None
        self.fail = fail

    async def deliver_local_tool(self, request: LocalToolRequest) -> ToolResult:
        self.request = request
        if self.fail:
            raise DeviceEventUnavailable("device is offline")
        return ToolResult(
            action_id=request.action_id,
            trace_id=request.trace_id,
            device_id=request.device_id,
            conversation_id=request.conversation_id,
            session_id=request.session_id,
            delegation_id=request.delegation_id,
            context_revision=request.context_revision,
            job_id=request.job_id,
            turn_id=request.turn_id,
            status="succeeded",
            verified=True,
            retryable=False,
        )


def internal_tool_request(**overrides) -> LocalToolRequest:
    now = datetime.now(timezone.utc)
    values = {
        "action_id": UUID("00000000-0000-4000-8000-000000000107"),
        "trace_id": JOB_REQUEST.trace_id,
        "device_id": JOB_REQUEST.device_id,
        "conversation_id": JOB_REQUEST.conversation_id,
        "session_id": JOB_REQUEST.session_id,
        "delegation_id": JOB_REQUEST.delegation_id,
        "job_id": JOB_ID,
        "turn_id": JOB_REQUEST.turn_id,
        "context_revision": JOB_REQUEST.context_revision,
        "tool_name": "set_light_state",
        "payload": {"entity_id": "light.desk", "state": "on"},
        "issued_at": now,
        "expires_at": now + timedelta(seconds=15),
    }
    values.update(overrides)
    return LocalToolRequest(**values)


class DurableApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeStore()
        self.client = TestClient(create_durable_app(store=self.store, authenticate=authenticate))

    def tearDown(self) -> None:
        self.client.close()

    def test_operator_health_does_not_require_owner_mapping(self) -> None:
        response = self.client.get(
            "/v1/health", headers={"Authorization": "Bearer operator-test"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_operator_health_reports_database_failure_without_leaking_error(self) -> None:
        self.store.health_error = True

        response = self.client.get(
            "/v1/health", headers={"Authorization": "Bearer operator-test"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unhealthy")
        self.assertNotIn("database internals", response.text)

    def test_operator_health_reports_stopped_background_service(self) -> None:
        async def broken_service(_stop):
            raise RuntimeError("private service detail")

        client = TestClient(create_durable_app(
            store=self.store, authenticate=authenticate,
            background_services=(broken_service,),
        ))
        with client:
            response = client.get("/v1/health", headers={"Authorization": "Bearer operator-test"})
        self.assertEqual(response.json()["status"], "degraded")
        self.assertEqual(response.json()["dependencies"]["cube-background-service-0"], "unavailable")
        self.assertNotIn("private service detail", response.text)

    def test_operator_cannot_use_owner_routes(self) -> None:
        response = self.client.get(
            "/v1/timers", headers={"Authorization": "Bearer operator-test"}
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "PERMISSION_DENIED")

    def test_global_event_stream_is_owner_scoped_and_resumable(self) -> None:
        response = self.client.get("/v1/events", headers={"Authorization": "Bearer owner-test"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(self.store.event_owner, OWNER_ID)

    def test_worker_local_tool_route_checks_scope_and_delivers_correlated_result(self):
        hub = FakeEventHub()
        client = TestClient(
            create_durable_app(
                store=self.store,
                authenticate=authenticate,
                device_event_hub=hub,  # type: ignore[arg-type]
            )
        )
        try:
            response = client.post(
                "/internal/v1/local-tools",
                headers={"Authorization": "Bearer worker-test"},
                json=internal_tool_request().model_dump(mode="json"),
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "succeeded")
            self.assertEqual(hub.request.action_id, UUID("00000000-0000-4000-8000-000000000107"))

            owner_response = client.post(
                "/internal/v1/local-tools",
                headers={"Authorization": "Bearer owner-test"},
                json=internal_tool_request().model_dump(mode="json"),
            )
            self.assertEqual(owner_response.status_code, 403)

            wrong_device_response = client.post(
                "/internal/v1/local-tools",
                headers={"Authorization": "Bearer worker-test"},
                json=internal_tool_request(device_id="cube-02").model_dump(mode="json"),
            )
            self.assertEqual(wrong_device_response.status_code, 403)

            stale_response = client.post(
                "/internal/v1/local-tools",
                headers={"Authorization": "Bearer worker-test"},
                json=internal_tool_request(context_revision=1).model_dump(mode="json"),
            )
            self.assertEqual(stale_response.status_code, 409)
        finally:
            client.close()

    def test_worker_local_tool_route_sanitizes_disconnected_device(self):
        client = TestClient(
            create_durable_app(
                store=self.store,
                authenticate=authenticate,
                device_event_hub=FakeEventHub(fail=True),  # type: ignore[arg-type]
            )
        )
        try:
            response = client.post(
                "/internal/v1/local-tools",
                headers={"Authorization": "Bearer worker-test"},
                json=internal_tool_request().model_dump(mode="json"),
            )
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"]["code"], "DEVICE_OFFLINE")
            self.assertNotIn("device is offline", response.text)
        finally:
            client.close()

    def test_dashboard_list_routes_are_owner_scoped(self) -> None:
        headers = {"Authorization": "Bearer owner-test"}
        sessions = self.client.get("/v1/sessions?limit=50", headers=headers)
        jobs = self.client.get("/v1/jobs?limit=50", headers=headers)

        self.assertEqual(sessions.status_code, 200)
        self.assertEqual(sessions.json()["sessions"], [])
        self.assertEqual(jobs.status_code, 200)
        self.assertEqual(jobs.json()["jobs"], [])
        self.assertEqual(self.store.sessions_owner, OWNER_ID)
        self.assertEqual(self.store.jobs_owner, OWNER_ID)

    def test_device_session_close_is_bound_to_authenticated_device(self) -> None:
        response = self.client.delete(
            "/v1/sessions/00000000-0000-4000-8000-000000000111",
            headers={"Authorization": "Bearer device-test"},
        )

        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.store.closed_session_owner, OWNER_ID)
        self.assertEqual(self.store.closed_session_device, "cube-01")

    def test_device_session_close_requires_device_mapping(self) -> None:
        async def authenticate_unmapped(_request: Request) -> AuthenticatedPrincipal:
            return AuthenticatedPrincipal(role="device", owner_id=OWNER_ID)

        self.client.app.dependency_overrides[authenticate] = authenticate_unmapped
        try:
            response = self.client.delete(
                "/v1/sessions/00000000-0000-4000-8000-000000000111",
                headers={"Authorization": "Bearer device-test"},
            )
        finally:
            self.client.app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 403)
        self.assertIsNone(self.store.closed_session_owner)
        self.assertEqual(
            self.client.get("/v1/jobs", headers={"Authorization": "Bearer operator-test"}).status_code,
            403,
        )

    def test_dashboard_is_same_origin_and_does_not_persist_owner_token(self) -> None:
        page = self.client.get("/dashboard")
        script = self.client.get("/dashboard/assets/app.js")

        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.headers["cache-control"], "no-store")
        self.assertIn("frame-ancestors 'none'", page.headers["content-security-policy"])
        self.assertIn("Application-side monthly/job budget limits are not configured", page.text)
        self.assertEqual(script.status_code, 200)
        self.assertNotIn("localStorage", script.text)
        self.assertNotIn("sessionStorage", script.text)

    def test_global_event_stream_rejects_malformed_cursor(self) -> None:
        response = self.client.get(
            "/v1/events",
            headers={"Authorization": "Bearer owner-test", "Last-Event-ID": "bad-cursor"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")

    def test_correction_uses_authenticated_owner_and_idempotency_header(self) -> None:
        response = self.client.post(
            f"/v1/jobs/{JOB_ID}/corrections",
            headers={"Authorization": "Bearer owner-test", "Idempotency-Key": "correction-1"},
            json={"context_revision": 1, "request_text": "Use the corrected request."},
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.store.correction_owner, OWNER_ID)
        self.assertEqual(self.store.correction_key, "correction-1")
        self.assertEqual(response.json()["context_revision"], 1)

    def test_device_can_manage_only_its_own_job(self) -> None:
        device_headers = {"Authorization": "Bearer device-test"}
        owned = self.client.get(f"/v1/jobs/{JOB_ID}", headers=device_headers)
        foreign_id = UUID("00000000-0000-4000-8000-000000000199")
        foreign = self.client.get(f"/v1/jobs/{foreign_id}", headers=device_headers)
        foreign_cancel = self.client.post(
            f"/v1/jobs/{foreign_id}/cancel",
            headers={**device_headers, "Idempotency-Key": "foreign-device-cancel"},
        )
        self.assertEqual(foreign_cancel.status_code, 404)
        self.assertIsNone(self.store.job_cancel_owner)
        cancelled = self.client.post(
            f"/v1/jobs/{JOB_ID}/cancel",
            headers={**device_headers, "Idempotency-Key": "device-cancel-1"},
        )

        self.assertEqual(owned.status_code, 200)
        self.assertEqual(owned.json()["job_id"], str(JOB_ID))
        self.assertEqual(foreign.status_code, 404)
        self.assertEqual(cancelled.status_code, 202)
        self.assertEqual(self.store.job_cancel_owner, OWNER_ID)
        self.assertEqual(self.store.correction_key, "device-cancel-1")

    def test_unmapped_device_cannot_read_or_manage_jobs(self) -> None:
        async def authenticate_unmapped(_request: Request) -> AuthenticatedPrincipal:
            return AuthenticatedPrincipal(role="device", owner_id=OWNER_ID)

        self.client.app.dependency_overrides[authenticate] = authenticate_unmapped
        try:
            response = self.client.post(
                f"/v1/jobs/{JOB_ID}/cancel",
                headers={"Authorization": "Bearer device-test", "Idempotency-Key": "unmapped-cancel"},
            )
        finally:
            self.client.app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 403)
        self.assertIsNone(self.store.job_cancel_owner)

    def test_missing_idempotency_header_returns_contract_error(self) -> None:
        response = self.client.post(
            f"/v1/jobs/{JOB_ID}/corrections",
            headers={"Authorization": "Bearer owner-test"},
            json={"context_revision": 1, "request_text": "Corrected request."},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")

    def test_idempotency_digest_normalizes_nested_uuid_and_contract_model(self) -> None:
        request = MemoryWriteRequest(
            category="preference",
            content="Tea with lemon",
            source_turn_id=UUID("00000000-0000-4000-8000-000000000103"),
        )

        actual = _digest_json({"memory_id": JOB_ID, "request": request})
        expected = _digest_json(
            {
                "memory_id": str(JOB_ID),
                "request": request.model_dump(mode="json"),
            }
        )

        self.assertEqual(actual, expected)

    def test_memory_routes_use_owner_scope_and_require_idempotency(self) -> None:
        source_turn_id = UUID("00000000-0000-4000-8000-000000000104")
        headers = {
            "Authorization": "Bearer owner-test",
            "Idempotency-Key": "memory-write-1",
        }
        written = self.client.post(
            "/v1/memories",
            headers=headers,
            json={
                "category": "preference",
                "content": "Tea with lemon.",
                "source_turn_id": str(source_turn_id),
            },
        )

        self.assertEqual(written.status_code, 201)
        memory_id = written.json()["memory_id"]
        self.assertEqual(self.store.memory_owner, OWNER_ID)
        self.assertEqual(self.store.memory_idempotency_key, "memory-write-1")
        listed = self.client.get("/v1/memories", headers={"Authorization": "Bearer owner-test"})
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["memories"][0]["content"], "Tea with lemon.")

        updated = self.client.patch(
            f"/v1/memories/{memory_id}",
            headers={**headers, "Idempotency-Key": "memory-update-1"},
            json={"content": "Mint tea.", "source_turn_id": str(source_turn_id)},
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["content"], "Mint tea.")

        deleted = self.client.delete(
            f"/v1/memories/{memory_id}",
            headers={**headers, "Idempotency-Key": "memory-delete-1"},
        )
        self.assertEqual(deleted.status_code, 204)
        self.assertIsNone(self.store.memory)
        self.assertEqual(self.store.memory_owner, OWNER_ID)

    def test_memory_mutation_without_idempotency_key_fails(self) -> None:
        response = self.client.post(
            "/v1/memories",
            headers={"Authorization": "Bearer owner-test"},
            json={
                "category": "fact",
                "content": "Likes tea.",
                "source_turn_id": "00000000-0000-4000-8000-000000000104",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")

    def test_usage_route_returns_monthly_owner_summary(self) -> None:
        response = self.client.get("/v1/usage", headers={"Authorization": "Bearer owner-test"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.usage_owner, OWNER_ID)
        self.assertEqual(response.json()["summary"]["input_tokens"], 0)
        self.assertIsNone(response.json()["summary"]["estimated_cost_usd"])
        self.assertEqual(self.store.usage_period_start.day, 1)
        self.assertEqual(
            self.store.usage_period_end,
            datetime.fromisoformat(response.json()["as_of"].replace("Z", "+00:00")),
        )


if __name__ == "__main__":
    unittest.main()
