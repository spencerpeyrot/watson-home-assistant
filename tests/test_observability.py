from __future__ import annotations

import os
import asyncio
import unittest
from uuid import uuid4

from service.elevenlabs_diagnostics import transport_diagnostics
from voice.observability import Observation, _content, _metadata


class FakeSpan:
    def __init__(self) -> None:
        self.updates = []
        self.children = []
        self.ended = False

    def update(self, **values) -> None:
        self.updates.append(values)

    def start_observation(self, **_values):
        child = FakeSpan()
        self.children.append(child)
        return child

    def end(self) -> None:
        self.ended = True


class ObservabilityTests(unittest.TestCase):
    def test_content_redacts_before_truncating(self) -> None:
        secret = "sensitive-observability-test-secret"
        key = "CUBE_OBSERVABILITY_TEST_PASSWORD"
        os.environ[key] = secret
        try:
            self.assertEqual(_content("x" * 3_999 + secret), "x" * 3_999 + "[redacted] [truncated]")
        finally:
            del os.environ[key]

    def test_metadata_masks_and_bounds_strings(self) -> None:
        secret = "metadata-observability-test-secret"
        os.environ["CUBE_OBSERVABILITY_TEST_TOKEN"] = secret
        try:
            result = _metadata({"provider_request_id": secret, "session_id": uuid4(), "unknown": "drop"})
        finally:
            del os.environ["CUBE_OBSERVABILITY_TEST_TOKEN"]
        self.assertEqual(result["provider_request_id"], "[redacted]")
        self.assertNotIn("unknown", result)
        self.assertEqual(len(result["session_id"]), 36)

    def test_operation_preserves_safe_provider_error_metadata(self) -> None:
        span = FakeSpan()
        observation = Observation(span, session_id=uuid4())

        class ProviderFailure(Exception):
            observability = {
                "provider": "elevenlabs", "operation": "stt", "failure_stage": "session_start",
                "error_category": "rate_limit", "provider_error_code": "too_many_requests",
                "retryable": True, "provider_request_id": "req-123",
            }

        with self.assertRaises(ProviderFailure):
            with observation.operation("speech.recognize"):
                raise ProviderFailure("sanitized")

        update = span.children[0].updates[-1]
        self.assertEqual(update["metadata"]["failure_stage"], "session_start")
        self.assertEqual(update["metadata"]["error_category"], "rate_limit")
        self.assertEqual(update["metadata"]["provider_request_id"], "req-123")
        self.assertEqual(update["metadata"]["status"], "failed")
        self.assertEqual(update["level"], "ERROR")

    def test_cancellation_has_cancelled_status(self) -> None:
        span = FakeSpan()
        observation = Observation(span, session_id=uuid4())
        with self.assertRaises(asyncio.CancelledError):
            with observation.operation("work"):
                raise asyncio.CancelledError()
        self.assertEqual(span.children[0].updates[-1]["metadata"]["status"], "cancelled")
        self.assertEqual(span.children[0].updates[-1]["level"], "WARNING")

    def test_event_status_sets_observation_level(self) -> None:
        span = FakeSpan()
        observation = Observation(span, session_id=uuid4())
        observation.event("provider.failed", status="failed", error_type="OSError")
        child = span.children[0]
        self.assertEqual(child.updates[-1]["metadata"]["status"], "failed")
        self.assertEqual(child.updates[-1]["level"], "ERROR")
        self.assertTrue(child.ended)

    def test_transport_diagnostics_keep_only_safe_status_and_request_id(self) -> None:
        class Failure(Exception):
            status_code = 503
            response = type("Response", (), {
                "status_code": 503, "headers": {"x-request-id": "req/123"}
            })()

        details = transport_diagnostics(Failure())
        self.assertEqual(details["http_status"], 503)
        self.assertEqual(details["error_category"], "provider_unavailable")
        self.assertTrue(details["retryable"])
        self.assertEqual(details["provider_request_id"], "req/123")
        self.assertNotIn("secret", details["provider_request_id"])


if __name__ == "__main__":
    unittest.main()
