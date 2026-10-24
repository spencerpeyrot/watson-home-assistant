from __future__ import annotations

import hashlib
import json
import unittest
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from service.postgres import IdempotencyConflict
from service.tool_policy import (
    ActionClaim,
    ApprovalGrant,
    ToolOutcome,
    ToolPolicyRegistry,
)


OWNER = UUID("00000000-0000-4000-8000-000000000201")
ACTION = UUID("00000000-0000-4000-8000-000000000202")
OTHER_ACTION = UUID("00000000-0000-4000-8000-000000000203")
TRACE = UUID("00000000-0000-4000-8000-000000000204")
APPROVAL = UUID("00000000-0000-4000-8000-000000000205")


class LightPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity_id: str
    state: str


class FakeLedger:
    def __init__(self) -> None:
        self.rows: dict[tuple[UUID, str], dict[str, Any]] = {}
        self.claims = 0

    async def claim_action(self, **values) -> ActionClaim:
        self.claims += 1
        key = (values["owner_id"], values["idempotency_key"])
        encoded = json.dumps(values["payload"], sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        row = self.rows.get(key)
        if row is None:
            self.rows[key] = {
                "action_id": values["action_id"],
                "digest": digest,
                "state": "executing",
                "result": None,
            }
            return ActionClaim("execute", values["action_id"], "executing")
        if row["digest"] != digest:
            raise IdempotencyConflict("key reused with another payload")
        if row["state"] == "succeeded":
            return ActionClaim("completed", row["action_id"], "succeeded", row["result"])
        return ActionClaim("blocked", row["action_id"], row["state"])

    async def complete_action(
        self, *, owner_id: UUID, action_id: UUID, result: Any, verified: bool
    ) -> None:
        assert verified
        row = next(row for row in self.rows.values() if row["action_id"] == action_id)
        row["state"] = "succeeded"
        row["result"] = result

    async def mark_action_uncertain(self, *, owner_id: UUID, action_id: UUID) -> None:
        row = next(row for row in self.rows.values() if row["action_id"] == action_id)
        row["state"] = "needs_reconciliation"


class ToolPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_and_invalid_tools_never_reach_executor(self) -> None:
        calls: list[LightPayload] = []

        async def execute(payload: BaseModel) -> ToolOutcome:
            calls.append(payload)  # type: ignore[arg-type]
            return ToolOutcome(verified=True, result={"ok": True})

        registry = ToolPolicyRegistry()
        registry.register(
            name="set_light",
            payload_model=LightPayload,
            risk="automatic",
            execute=execute,
        )
        ledger = FakeLedger()
        unknown = await registry.dispatch(
            owner_id=OWNER,
            tool_name="unknown",
            payload={},
            action_id=ACTION,
            trace_id=TRACE,
            context_revision=0,
            idempotency_key="write-1",
            ledger=ledger,
        )
        invalid = await registry.dispatch(
            owner_id=OWNER,
            tool_name="set_light",
            payload={"entity_id": "light.desk", "state": "on", "extra": True},
            action_id=ACTION,
            trace_id=TRACE,
            context_revision=0,
            idempotency_key="write-1",
            ledger=ledger,
        )

        self.assertEqual(unknown.status, "denied")
        self.assertEqual(unknown.error_code, "ACTION_DENIED")
        self.assertEqual(invalid.status, "denied")
        self.assertEqual(invalid.error_code, "INVALID_REQUEST")
        self.assertEqual(calls, [])
        self.assertEqual(ledger.claims, 0)

    async def test_approval_binds_action_payload_revision_and_expiry(self) -> None:
        async def execute(_payload: BaseModel) -> ToolOutcome:
            self.fail("approval checks must happen before execution")

        registry = ToolPolicyRegistry()
        registry.register(
            name="unlock_door",
            payload_model=LightPayload,
            risk="approval_required",
            execute=execute,
        )
        payload = {"entity_id": "lock.front", "state": "unlock"}
        now = datetime.now(timezone.utc)
        pending = registry.authorize(
            tool_name="unlock_door",
            payload=payload,
            action_id=ACTION,
            context_revision=4,
        )
        self.assertEqual(pending.status, "approval_required")
        assert pending.payload_digest is not None
        expected_digest = hashlib.sha256(
            json.dumps(
                {
                    "tool_name": "unlock_door",
                    "resource_key": "unlock_door",
                    "payload": payload,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.assertEqual(pending.payload_digest, expected_digest)
        grant = ApprovalGrant(
            approval_id=APPROVAL,
            action_id=ACTION,
            payload_digest=pending.payload_digest,
            context_revision=4,
            status="approved",
            expires_at=now + timedelta(minutes=1),
        )
        accepted = registry.authorize(
            tool_name="unlock_door",
            payload=payload,
            action_id=ACTION,
            context_revision=4,
            approval=grant,
            now=now,
        )
        stale = registry.authorize(
            tool_name="unlock_door",
            payload=payload,
            action_id=OTHER_ACTION,
            context_revision=4,
            approval=grant,
            now=now,
        )
        expired = registry.authorize(
            tool_name="unlock_door",
            payload=payload,
            action_id=ACTION,
            context_revision=4,
            approval=ApprovalGrant(
                approval_id=APPROVAL,
                action_id=ACTION,
                payload_digest=pending.payload_digest,
                context_revision=4,
                status="approved",
                expires_at=now,
            ),
            now=now,
        )

        self.assertEqual(accepted.status, "allowed")
        self.assertEqual(stale.error_code, "APPROVAL_PAYLOAD_MISMATCH")
        self.assertEqual(expired.error_code, "APPROVAL_EXPIRED")

    async def test_completed_write_reuses_result_without_second_execution(self) -> None:
        calls: list[str] = []

        async def execute(payload: BaseModel) -> ToolOutcome:
            calls.append(payload.model_dump()["state"])
            return ToolOutcome(verified=True, result={"observed_state": "on"})

        registry = ToolPolicyRegistry()
        registry.register(
            name="set_light",
            payload_model=LightPayload,
            risk="automatic",
            execute=execute,
            resource_key=lambda payload: payload.model_dump()["entity_id"],
        )
        ledger = FakeLedger()
        values = {
            "owner_id": OWNER,
            "tool_name": "set_light",
            "payload": {"entity_id": "light.desk", "state": "on"},
            "trace_id": TRACE,
            "context_revision": 0,
            "idempotency_key": "set-light-on-1",
            "ledger": ledger,
        }
        first = await registry.dispatch(action_id=ACTION, **values)
        second = await registry.dispatch(action_id=OTHER_ACTION, **values)

        self.assertEqual(first.status, "succeeded")
        self.assertEqual(second.status, "succeeded")
        self.assertEqual(second.action_id, ACTION)
        self.assertTrue(second.reused)
        self.assertEqual(second.result, {"observed_state": "on"})
        self.assertEqual(calls, ["on"])

    async def test_uncertain_write_requires_reconciliation_before_retry(self) -> None:
        calls = 0

        async def execute(_payload: BaseModel) -> ToolOutcome:
            nonlocal calls
            calls += 1
            raise TimeoutError("device response lost after write")

        registry = ToolPolicyRegistry()
        registry.register(
            name="set_light",
            payload_model=LightPayload,
            risk="automatic",
            execute=execute,
        )
        ledger = FakeLedger()
        values = {
            "owner_id": OWNER,
            "tool_name": "set_light",
            "payload": {"entity_id": "light.desk", "state": "off"},
            "trace_id": TRACE,
            "context_revision": 0,
            "idempotency_key": "set-light-off-uncertain",
            "ledger": ledger,
        }

        with self.assertRaises(TimeoutError):
            await registry.dispatch(action_id=ACTION, **values)
        retry = await registry.dispatch(action_id=OTHER_ACTION, **values)

        self.assertEqual(retry.status, "needs_reconciliation")
        self.assertEqual(calls, 1)

    async def test_unverified_write_cannot_complete_ledger(self) -> None:
        calls = 0

        async def execute(_payload: BaseModel) -> ToolOutcome:
            nonlocal calls
            calls += 1
            return ToolOutcome(verified=False, result={"request_sent": True})

        registry = ToolPolicyRegistry()
        registry.register(
            name="set_light",
            payload_model=LightPayload,
            risk="automatic",
            execute=execute,
        )
        ledger = FakeLedger()
        values = {
            "owner_id": OWNER,
            "tool_name": "set_light",
            "payload": {"entity_id": "light.desk", "state": "on"},
            "trace_id": TRACE,
            "context_revision": 0,
            "idempotency_key": "set-light-unverified",
            "ledger": ledger,
        }

        result = await registry.dispatch(action_id=ACTION, **values)
        retry = await registry.dispatch(action_id=OTHER_ACTION, **values)

        self.assertFalse(result.verified)
        self.assertEqual(result.status, "needs_reconciliation")
        self.assertEqual(retry.status, "needs_reconciliation")
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
