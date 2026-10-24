from __future__ import annotations

import os
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from fastapi.testclient import TestClient

from contracts.models import SessionCreateRequest, TimerCreateRequest
from service.mock_server import create_app


class MockTimerRouteTests(unittest.TestCase):
    def test_timer_routes_enforce_scopes_and_idempotency(self) -> None:
        environment = {
            "CUBE_MOCK_DEVICE_ID": "cube-01",
            "CUBE_MOCK_DEVICE_TOKEN": "d" * 40,
            "CUBE_MOCK_OWNER_TOKEN": "o" * 40,
            "CUBE_MOCK_OPERATOR_TOKEN": "p" * 40,
        }
        request = TimerCreateRequest(
            due_at=datetime.now(timezone.utc) + timedelta(minutes=2),
            message="Tea is ready.",
            idempotency_key="timer-mock-1",
        )
        headers = {"Authorization": f"Bearer {environment['CUBE_MOCK_DEVICE_TOKEN']}"}
        with patch.dict(os.environ, environment, clear=False):
            with TestClient(create_app()) as client:
                first = client.post("/v1/timers", json=request.model_dump(mode="json"), headers=headers)
                replay = client.post("/v1/timers", json=request.model_dump(mode="json"), headers=headers)
                changed = client.post(
                    "/v1/timers",
                    json=request.model_copy(update={"message": "Coffee is ready."}).model_dump(mode="json"),
                    headers=headers,
                )
                self.assertEqual(first.status_code, 201)
                self.assertEqual(replay.json()["timer_id"], first.json()["timer_id"])
                self.assertEqual(changed.status_code, 409)

                owner_headers = {"Authorization": f"Bearer {environment['CUBE_MOCK_OWNER_TOKEN']}"}
                self.assertEqual(client.get("/v1/timers", headers=headers).status_code, 403)
                listed = client.get("/v1/timers", headers=owner_headers)
                self.assertEqual(listed.status_code, 200)
                self.assertEqual(len(listed.json()["timers"]), 1)
                timer_id = first.json()["timer_id"]
                self.assertEqual(client.delete(f"/v1/timers/{timer_id}", headers=owner_headers).status_code, 204)
                self.assertEqual(client.get("/v1/timers", headers=owner_headers).json()["timers"][0]["status"], "cancelled")

    def test_dashboard_list_routes_cover_mock_sessions_and_jobs(self) -> None:
        environment = {
            "CUBE_MOCK_DEVICE_ID": "cube-01",
            "CUBE_MOCK_DEVICE_TOKEN": "d" * 40,
            "CUBE_MOCK_OWNER_TOKEN": "o" * 40,
            "CUBE_MOCK_OPERATOR_TOKEN": "p" * 40,
        }
        device_headers = {"Authorization": f"Bearer {environment['CUBE_MOCK_DEVICE_TOKEN']}"}
        owner_headers = {"Authorization": f"Bearer {environment['CUBE_MOCK_OWNER_TOKEN']}"}
        session_request = SessionCreateRequest(
            conversation_id=uuid4(), locale="en-US", timezone="America/New_York"
        )
        task_request = json.loads(
            (Path(__file__).resolve().parents[1] / "contracts" / "fixtures" / "task_request.json")
            .read_text(encoding="utf-8")
        )
        with patch.dict(os.environ, environment, clear=False):
            with TestClient(create_app()) as client:
                page = client.get("/dashboard")
                self.assertEqual(page.status_code, 200)
                self.assertEqual(page.headers["cache-control"], "no-store")
                self.assertEqual(client.get("/dashboard/assets/app.js").status_code, 200)
                created = client.post(
                    "/v1/sessions",
                    json=session_request.model_dump(mode="json"),
                    headers=device_headers,
                )
                self.assertEqual(created.status_code, 201)
                sessions = client.get("/v1/sessions", headers=owner_headers)
                self.assertEqual(sessions.status_code, 200)
                self.assertEqual(sessions.json()["sessions"][0]["state"], "ready")

                accepted = client.post("/v1/jobs", json=task_request, headers=device_headers)
                self.assertEqual(accepted.status_code, 202)
                jobs = client.get("/v1/jobs", headers=owner_headers)
                self.assertEqual(jobs.status_code, 200)
                self.assertEqual(jobs.json()["jobs"][0]["job_id"], accepted.json()["job_id"])


if __name__ == "__main__":
    unittest.main()
