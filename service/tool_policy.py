"""Typed tool allowlist, approval binding, and idempotent dispatch boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ValidationError


Risk = Literal["read_only", "automatic", "approval_required"]
DecisionStatus = Literal["allowed", "approval_required", "denied"]
DispatchStatus = Literal[
    "succeeded", "denied", "approval_required", "needs_reconciliation"
]


@dataclass(frozen=True)
class ApprovalGrant:
    approval_id: UUID
    action_id: UUID
    payload_digest: str
    context_revision: int
    status: Literal["pending", "approved", "denied", "expired"]
    expires_at: datetime


@dataclass(frozen=True)
class PolicyDecision:
    status: DecisionStatus
    tool_name: str | None = None
    payload: BaseModel | None = None
    payload_digest: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class ActionClaim:
    """Result of atomically claiming a write's owner/idempotency key."""

    outcome: Literal["execute", "completed", "blocked"]
    action_id: UUID
    state: str
    result: Any | None = None


@dataclass(frozen=True)
class DispatchResult:
    status: DispatchStatus
    action_id: UUID
    verified: bool
    result: Any | None = None
    error_code: str | None = None
    reused: bool = False


@dataclass(frozen=True)
class ToolOutcome:
    """Executor result; writes complete only after verification."""

    verified: bool
    result: Any | None = None


@dataclass(frozen=True)
class LocalToolContext:
    """Authenticated voice/job scope needed for a Pi-local tool request."""

    device_id: str
    conversation_id: UUID
    session_id: UUID
    delegation_id: str
    turn_id: UUID
    expires_at: datetime


@dataclass(frozen=True)
class ToolExecutionContext:
    """Action identity plus device/session scope for contextual executors."""

    action_id: UUID
    trace_id: UUID
    job_id: UUID | None
    context_revision: int
    issued_at: datetime
    local: LocalToolContext


class ActionLedger(Protocol):
    async def claim_action(
        self,
        *,
        owner_id: UUID,
        action_id: UUID,
        trace_id: UUID,
        job_id: UUID | None,
        tool_name: str,
        resource_key: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> ActionClaim: ...

    async def complete_action(
        self, *, owner_id: UUID, action_id: UUID, result: Any, verified: bool
    ) -> None: ...

    async def mark_action_uncertain(self, *, owner_id: UUID, action_id: UUID) -> None: ...


@dataclass(frozen=True)
class _ToolDefinition:
    name: str
    description: str
    payload_model: type[BaseModel]
    risk: Risk
    execute: Callable[[BaseModel], Awaitable[ToolOutcome]] | None
    execute_contextual: (
        Callable[[BaseModel, ToolExecutionContext], Awaitable[ToolOutcome]] | None
    )
    resource_key: Callable[[BaseModel], str]


class ToolPolicyRegistry:
    """Only registered, schema-valid tools can reach an executor."""

    def __init__(self) -> None:
        self._tools: dict[str, _ToolDefinition] = {}

    def register(
        self,
        *,
        name: str,
        description: str = "",
        payload_model: type[BaseModel],
        risk: Risk,
        execute: Callable[[BaseModel], Awaitable[ToolOutcome]] | None = None,
        execute_contextual: (
            Callable[[BaseModel, ToolExecutionContext], Awaitable[ToolOutcome]] | None
        ) = None,
        resource_key: Callable[[BaseModel], str] | None = None,
    ) -> None:
        if not name.strip() or name != name.strip() or name in self._tools:
            raise ValueError("tool name must be nonblank and unique")
        if risk not in {"read_only", "automatic", "approval_required"}:
            raise ValueError("tool risk class is invalid")
        if (execute is None) == (execute_contextual is None):
            raise ValueError("provide exactly one tool executor")
        self._tools[name] = _ToolDefinition(
            name=name,
            description=description.strip(),
            payload_model=payload_model,
            risk=risk,
            execute=execute,
            execute_contextual=execute_contextual,
            resource_key=resource_key or (lambda _payload: name),
        )

    def tool_specs(self) -> tuple[tuple[str, str, type[BaseModel]], ...]:
        """Return model-facing names, descriptions, and strict argument schemas."""

        return tuple(
            (name, definition.description, definition.payload_model)
            for name, definition in self._tools.items()
        )

    def authorize(
        self,
        *,
        tool_name: str,
        payload: dict[str, Any],
        action_id: UUID,
        context_revision: int,
        approval: ApprovalGrant | None = None,
        now: datetime | None = None,
    ) -> PolicyDecision:
        """Validate tool inputs and bind any required approval to exact action."""

        definition = self._tools.get(tool_name)
        if definition is None:
            return PolicyDecision(status="denied", error_code="ACTION_DENIED")
        try:
            validated = definition.payload_model.model_validate(payload)
        except ValidationError:
            return PolicyDecision(status="denied", error_code="INVALID_REQUEST")
        serialized = validated.model_dump(mode="json")
        try:
            resource_key = definition.resource_key(validated)
        except (TypeError, ValueError):
            return PolicyDecision(status="denied", error_code="INVALID_REQUEST")
        if not isinstance(resource_key, str) or not resource_key.strip():
            return PolicyDecision(status="denied", error_code="INVALID_REQUEST")
        canonical = json.dumps(
            {
                "tool_name": tool_name,
                "resource_key": resource_key,
                "payload": serialized,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        if definition.risk != "approval_required":
            return PolicyDecision(
                status="allowed",
                tool_name=tool_name,
                payload=validated,
                payload_digest=digest,
            )
        if approval is None:
            return PolicyDecision(
                status="approval_required",
                tool_name=tool_name,
                payload=validated,
                payload_digest=digest,
            )
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None or current_time.utcoffset() is None:
            raise ValueError("policy time must be timezone-aware")
        if approval.expires_at.tzinfo is None or approval.expires_at.utcoffset() is None:
            raise ValueError("approval expiry must be timezone-aware")
        if approval.status == "expired" or approval.expires_at <= current_time:
            return PolicyDecision(status="denied", error_code="APPROVAL_EXPIRED")
        if approval.status != "approved":
            return PolicyDecision(status="denied", error_code="ACTION_DENIED")
        if (
            approval.action_id != action_id
            or approval.context_revision != context_revision
            or approval.payload_digest != digest
        ):
            return PolicyDecision(
                status="denied", error_code="APPROVAL_PAYLOAD_MISMATCH"
            )
        return PolicyDecision(
            status="allowed",
            tool_name=tool_name,
            payload=validated,
            payload_digest=digest,
        )

    async def dispatch(
        self,
        *,
        owner_id: UUID,
        tool_name: str,
        payload: dict[str, Any],
        action_id: UUID,
        trace_id: UUID,
        context_revision: int,
        idempotency_key: str,
        ledger: ActionLedger | None,
        job_id: UUID | None = None,
        approval: ApprovalGrant | None = None,
        now: datetime | None = None,
        local_context: LocalToolContext | None = None,
    ) -> DispatchResult:
        """Authorize, durably claim writes, then invoke exactly one executor."""

        decision = self.authorize(
            tool_name=tool_name,
            payload=payload,
            action_id=action_id,
            context_revision=context_revision,
            approval=approval,
            now=now,
        )
        if decision.status != "allowed":
            return DispatchResult(
                status=decision.status,
                action_id=action_id,
                verified=False,
                error_code=decision.error_code,
            )
        definition = self._tools[tool_name]
        assert decision.payload is not None
        dispatch_time = now or datetime.now(timezone.utc)
        if dispatch_time.tzinfo is None or dispatch_time.utcoffset() is None:
            raise ValueError("dispatch time must be timezone-aware")
        if definition.execute_contextual is not None:
            if local_context is None:
                return DispatchResult(
                    status="denied",
                    action_id=action_id,
                    verified=False,
                    error_code="INVALID_REQUEST",
                )
            if (
                not local_context.device_id.strip()
                or not local_context.delegation_id.strip()
                or local_context.expires_at.tzinfo is None
                or local_context.expires_at.utcoffset() is None
                or local_context.expires_at <= dispatch_time
            ):
                return DispatchResult(
                    status="denied",
                    action_id=action_id,
                    verified=False,
                    error_code="ACTION_DENIED",
                )

        async def execute_tool() -> ToolOutcome:
            if definition.execute_contextual is not None:
                assert local_context is not None
                return await definition.execute_contextual(
                    decision.payload,
                    ToolExecutionContext(
                        action_id=action_id,
                        trace_id=trace_id,
                        job_id=job_id,
                        context_revision=context_revision,
                        issued_at=dispatch_time,
                        local=local_context,
                    ),
                )
            assert definition.execute is not None
            return await definition.execute(decision.payload)

        if definition.risk == "read_only":
            outcome = await execute_tool()
            return DispatchResult(
                status="succeeded" if outcome.verified else "needs_reconciliation",
                action_id=action_id,
                verified=outcome.verified,
                result=outcome.result,
            )
        if ledger is None:
            raise ValueError("write-capable tools require an action ledger")
        if not idempotency_key.strip():
            raise ValueError("write-capable tools require an idempotency key")
        payload_data = decision.payload.model_dump(mode="json")
        claim = await ledger.claim_action(
            owner_id=owner_id,
            action_id=action_id,
            trace_id=trace_id,
            job_id=job_id,
            tool_name=tool_name,
            resource_key=definition.resource_key(decision.payload),
            payload=payload_data,
            idempotency_key=idempotency_key,
        )
        if claim.outcome == "completed":
            return DispatchResult(
                status="succeeded",
                action_id=claim.action_id,
                verified=True,
                result=claim.result,
                reused=True,
            )
        if claim.outcome == "blocked":
            return DispatchResult(
                status="needs_reconciliation",
                action_id=claim.action_id,
                verified=False,
                error_code="RESOURCE_CONFLICT",
            )
        try:
            outcome = await execute_tool()
        except BaseException:
            await ledger.mark_action_uncertain(
                owner_id=owner_id, action_id=claim.action_id
            )
            raise
        if not outcome.verified:
            await ledger.mark_action_uncertain(
                owner_id=owner_id, action_id=claim.action_id
            )
            return DispatchResult(
                status="needs_reconciliation",
                action_id=claim.action_id,
                verified=False,
                result=outcome.result,
            )
        await ledger.complete_action(
            owner_id=owner_id,
            action_id=claim.action_id,
            result=outcome.result,
            verified=True,
        )
        return DispatchResult(
            status="succeeded",
            action_id=claim.action_id,
            verified=True,
            result=outcome.result,
        )
