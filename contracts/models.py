"""Internal, versioned application contracts. Provider payloads stay in adapters."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, model_validator


class ContractModel(BaseModel):
    """Strict base for messages crossing service boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RecentTurn(ContractModel):
    turn_id: UUID
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1)
    committed_at: AwareDatetime


class TaskRequest(ContractModel):
    """Committed user task; owner identity comes from gateway authentication."""

    schema_version: Literal[1] = 1
    device_id: str = Field(min_length=1)
    conversation_id: UUID
    session_id: UUID
    turn_id: UUID
    delegation_id: str = Field(min_length=1)
    trace_id: UUID
    request_text: str = Field(min_length=1)
    context_revision: int = Field(ge=0)
    recent_turns: tuple[RecentTurn, ...] = ()
    referenced_job_ids: tuple[UUID, ...] = ()
    locale: str = Field(min_length=2)
    timezone: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1, max_length=200)


class SessionCreateRequest(ContractModel):
    schema_version: Literal[1] = 1
    conversation_id: UUID
    locale: str = Field(min_length=2)
    timezone: str = Field(min_length=1)


class SessionResponse(ContractModel):
    schema_version: Literal[1] = 1
    session_id: UUID
    conversation_id: UUID
    state: Literal["connecting", "starting", "ready", "closing", "closed"]
    opened_at: AwareDatetime
    idle_close_seconds: int = Field(ge=0)
    max_session_seconds: int | None = Field(default=None, gt=0)


class SessionListResponse(ContractModel):
    schema_version: Literal[1] = 1
    sessions: tuple[SessionResponse, ...] = ()


JobStatusValue = Literal[
    "queued",
    "running",
    "waiting_for_approval",
    "succeeded",
    "failed",
    "cancelled",
    "needs_reconciliation",
]

JobEventKind = Literal[
    "status",
    "progress",
    "correction_acknowledged",
    "approval_requested",
    "artifact_ready",
    "result",
]


class ArtifactRef(ContractModel):
    artifact_id: UUID
    label: str = Field(min_length=1)
    media_type: str = Field(min_length=1)


class SourceRef(ContractModel):
    title: str = Field(min_length=1)
    url: HttpUrl
    retrieved_at: AwareDatetime | None = None


class UsageMeasurement(ContractModel):
    """Measured or estimated usage; units explicit to avoid mixed totals."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    connected_seconds: Decimal | None = Field(default=None, ge=0)
    cost_usd: Decimal | None = Field(default=None, ge=0)
    cost_kind: Literal["measured", "estimated"] | None = None

    @model_validator(mode="after")
    def require_cost_kind_with_cost(self) -> UsageMeasurement:
        if (self.cost_usd is None) != (self.cost_kind is None):
            raise ValueError("cost_usd and cost_kind must be supplied together")
        return self


class UsageRecord(ContractModel):
    """One persisted provider measurement with ownership scope and timestamp."""

    schema_version: Literal[1] = 1
    usage_id: UUID
    session_id: UUID | None = None
    job_id: UUID | None = None
    provider: str = Field(min_length=1)
    measurement: UsageMeasurement
    observed_at: AwareDatetime


class UsageSummary(ContractModel):
    """Owner-scoped current-period totals, with estimated and measured cost split."""

    period_start: AwareDatetime
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    connected_seconds: Decimal = Field(default=Decimal("0"), ge=0)
    measured_cost_usd: Decimal | None = Field(default=None, ge=0)
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)


class JobEvent(ContractModel):
    schema_version: Literal[1] = 1
    device_id: str = Field(min_length=1)
    job_id: UUID
    event_id: UUID
    sequence: int = Field(ge=1)
    kind: JobEventKind
    status: JobStatusValue
    conversation_id: UUID
    session_id: UUID
    turn_id: UUID
    delegation_id: str = Field(min_length=1)
    trace_id: UUID
    parent_job_id: UUID | None = None
    context_revision: int = Field(ge=0)
    summary: str = Field(min_length=1)
    timestamp: AwareDatetime
    approval_id: UUID | None = None
    artifact_refs: tuple[ArtifactRef, ...] = ()
    sources: tuple[SourceRef, ...] = ()
    error_code: str | None = None
    measured_usage: UsageMeasurement | None = None


class JobAccepted(ContractModel):
    schema_version: Literal[1] = 1
    job_id: UUID
    status: Literal["queued", "running"]
    trace_id: UUID
    deduplicated: bool
    first_event_sequence: int = Field(ge=1)


class JobSnapshot(ContractModel):
    schema_version: Literal[1] = 1
    job_id: UUID
    status: JobStatusValue
    context_revision: int = Field(ge=0)
    last_sequence: int = Field(ge=0)
    summary: str = Field(min_length=1)
    artifact_refs: tuple[ArtifactRef, ...] = ()
    sources: tuple[SourceRef, ...] = ()
    error_code: str | None = None


class JobListResponse(ContractModel):
    schema_version: Literal[1] = 1
    jobs: tuple[JobSnapshot, ...] = ()


class JobCorrectionRequest(ContractModel):
    schema_version: Literal[1] = 1
    context_revision: int = Field(ge=1)
    request_text: str = Field(min_length=1)
    recent_turns: tuple[RecentTurn, ...] = ()


class ApprovalDecisionRequest(ContractModel):
    schema_version: Literal[1] = 1
    decision: Literal["approve", "deny"]
    payload_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    context_revision: int = Field(ge=0)


class ApprovalRecord(ContractModel):
    schema_version: Literal[1] = 1
    approval_id: UUID
    action_id: UUID
    action_summary: str = Field(min_length=1)
    payload_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    context_revision: int = Field(ge=0)
    status: Literal["pending", "approved", "denied", "expired"]
    expires_at: AwareDatetime


class ApprovalListResponse(ContractModel):
    schema_version: Literal[1] = 1
    approvals: tuple[ApprovalRecord, ...] = ()


class ApprovalDecisionResponse(ContractModel):
    schema_version: Literal[1] = 1
    approval_id: UUID
    status: Literal["approved", "denied"]
    decided_at: AwareDatetime


class MemoryRecord(ContractModel):
    schema_version: Literal[1] = 1
    memory_id: UUID
    category: Literal["preference", "fact"]
    content: str = Field(min_length=1)
    source_turn_id: UUID
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def updated_after_creation(self) -> MemoryRecord:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be on or after created_at")
        return self


class MemoryWriteRequest(ContractModel):
    schema_version: Literal[1] = 1
    category: Literal["preference", "fact"]
    content: str = Field(min_length=1)
    source_turn_id: UUID


class MemoryUpdateRequest(ContractModel):
    schema_version: Literal[1] = 1
    content: str = Field(min_length=1)
    source_turn_id: UUID


class MemoryListResponse(ContractModel):
    schema_version: Literal[1] = 1
    memories: tuple[MemoryRecord, ...] = ()


class UsageResponse(ContractModel):
    schema_version: Literal[1] = 1
    measured_usage: tuple[UsageRecord, ...] = ()
    summary: UsageSummary
    as_of: AwareDatetime


class TimerCreateRequest(ContractModel):
    schema_version: Literal[1] = 1
    conversation_id: UUID | None = None
    due_at: AwareDatetime
    message: str = Field(min_length=1, max_length=500)
    idempotency_key: str = Field(min_length=1, max_length=200)


TimerStatus = Literal["scheduled", "firing", "delivered", "cancelled", "failed"]


class TimerRecord(ContractModel):
    schema_version: Literal[1] = 1
    timer_id: UUID
    conversation_id: UUID | None = None
    due_at: AwareDatetime
    message: str = Field(min_length=1, max_length=500)
    status: TimerStatus
    created_at: AwareDatetime
    delivered_at: AwareDatetime | None = None


class TimerListResponse(ContractModel):
    schema_version: Literal[1] = 1
    timers: tuple[TimerRecord, ...] = ()


class TimerFiredEvent(ContractModel):
    """Stable event ID makes timer notification delivery deduplicable."""

    schema_version: Literal[1] = 1
    event_id: UUID
    timer_id: UUID
    device_id: str = Field(min_length=1)
    conversation_id: UUID | None = None
    due_at: AwareDatetime
    fired_at: AwareDatetime
    message: str = Field(min_length=1, max_length=500)
    attempt_count: int = Field(ge=1)


class TimerEventAck(ContractModel):
    """Device acknowledgement after it accepts a timer notification."""

    schema_version: Literal[1] = 1
    event_id: UUID


class HealthResponse(ContractModel):
    schema_version: Literal[1] = 1
    status: Literal["healthy", "degraded", "unhealthy"]
    service: str = Field(min_length=1)
    observed_at: AwareDatetime
    dependencies: dict[str, Literal["healthy", "degraded", "unavailable"]] = Field(
        default_factory=dict
    )


class DeviceHealthEvent(ContractModel):
    schema_version: Literal[1] = 1
    device_id: str = Field(min_length=1)
    trace_id: UUID
    observed_at: AwareDatetime
    audio_ready: bool
    session_state: Literal["idle", "connecting", "active", "closing"]
    muted: bool
    temperature_c: Decimal | None = None
    playback_state: Literal["stopped", "buffering", "playing"]


class PlaybackMeterEvent(ContractModel):
    schema_version: Literal[1] = 1
    device_id: str = Field(min_length=1)
    session_id: UUID
    sequence: int = Field(ge=0)
    played_at: AwareDatetime
    rms_level: Decimal = Field(ge=0, le=1)
    peak_level: Decimal = Field(ge=0, le=1)


class LocalToolRequest(ContractModel):
    schema_version: Literal[1] = 1
    action_id: UUID
    trace_id: UUID
    device_id: str = Field(min_length=1)
    conversation_id: UUID
    session_id: UUID
    delegation_id: str = Field(min_length=1)
    job_id: UUID | None = None
    turn_id: UUID
    context_revision: int = Field(ge=0)
    tool_name: str = Field(min_length=1)
    payload: dict[str, Any]
    issued_at: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def expires_after_issue(self) -> LocalToolRequest:
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be after issued_at")
        return self


class AudioFrameMetadata(ContractModel):
    schema_version: Literal[1] = 1
    session_id: UUID
    sequence: int = Field(ge=0)
    captured_at: AwareDatetime
    encoding: Literal["pcm_s16le"]
    sample_rate_hz: int = Field(gt=0)
    channels: int = Field(gt=0)
    frame_duration_ms: int = Field(gt=0)


class AudioTurnEnd(ContractModel):
    """Mark the final captured frame and VAD speech length for one input turn."""

    type: Literal["voice_audio_turn_end"] = "voice_audio_turn_end"
    schema_version: Literal[1] = 1
    session_id: UUID
    turn_id: UUID
    last_audio_sequence: int = Field(ge=0)
    vad_speech_ms: int = Field(default=0, ge=0, le=43_700)


class AudioOutputFrameMetadata(ContractModel):
    """PCM playback frame relayed from the gateway to the device."""

    schema_version: Literal[1] = 1
    session_id: UUID
    sequence: int = Field(ge=0)
    encoding: Literal["pcm_s16le"]
    sample_rate_hz: Literal[24_000] = 24_000
    channels: Literal[1] = 1
    payload_bytes: int = Field(ge=2, le=9_600)

    @model_validator(mode="after")
    def even_pcm_payload(self) -> AudioOutputFrameMetadata:
        if self.payload_bytes % 2:
            raise ValueError("PCM output payload byte count must be even")
        return self


class VoiceSessionEvent(ContractModel):
    """Provider-neutral control event for one authenticated voice session."""

    schema_version: Literal[1] = 1
    type: Literal[
        "session_ready",
        "transcript_delta",
        "assistant_processing",
        "assistant_response_started",
        "assistant_response_complete",
        "assistant_turn_ignored",
        "delegation_created",
        "session_error",
        "session_closed",
    ]
    session_id: UUID
    half_duplex: bool | None = None
    reply_expectation: Literal[
        "none", "open_answer", "confirmation", "closing"
    ] | None = None
    direction: Literal["user", "assistant"] | None = None
    delta: str | None = None
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    delegation_id: str | None = None
    target: Literal["client", "responses", "server"] | None = None
    offset_ms: int | None = Field(default=None, ge=0)
    code: Literal[
        "INVALID_REQUEST",
        "DEPENDENCY_UNAVAILABLE",
        "DEPENDENCY_TIMEOUT",
        "INTERNAL_ERROR",
    ] | None = None

    @model_validator(mode="after")
    def validate_event_fields(self) -> VoiceSessionEvent:
        fields = {
            "half_duplex": self.half_duplex,
            "reply_expectation": self.reply_expectation,
            "direction": self.direction,
            "delta": self.delta,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "delegation_id": self.delegation_id,
            "target": self.target,
            "offset_ms": self.offset_ms,
            "code": self.code,
        }
        allowed: dict[str, set[str]] = {
            "session_ready": {"half_duplex"},
            "transcript_delta": {"direction", "delta", "start_ms", "end_ms"},
            "assistant_processing": set(),
            "assistant_response_started": set(),
            "assistant_response_complete": {"reply_expectation"},
            "assistant_turn_ignored": set(),
            "delegation_created": {"delegation_id", "target", "offset_ms"},
            "session_error": {"code"},
            "session_closed": set(),
        }
        required: dict[str, set[str]] = {
            "session_ready": set(),
            "transcript_delta": {"direction", "delta"},
            "assistant_processing": set(),
            "assistant_response_started": set(),
            "assistant_response_complete": set(),
            "assistant_turn_ignored": set(),
            "delegation_created": {"delegation_id", "target", "offset_ms"},
            "session_error": {"code"},
            "session_closed": set(),
        }
        present = {name for name, value in fields.items() if value is not None}
        if present - allowed[self.type] or not required[self.type] <= present:
            raise ValueError("voice event fields do not match its type")
        if self.type == "transcript_delta":
            if not self.delta or (self.start_ms is not None and self.end_ms is not None and self.end_ms < self.start_ms):
                raise ValueError("transcript delta or timing is invalid")
            if (self.start_ms is None) != (self.end_ms is None):
                raise ValueError("transcript timing must include both endpoints")
        if self.type == "delegation_created" and (
            self.delegation_id is None or not self.delegation_id.strip()
        ):
            raise ValueError("delegation ID must not be blank")
        return self


class VoiceContextAppend(ContractModel):
    """Bounded device-to-gateway context update for a known delegation."""

    schema_version: Literal[1] = 1
    session_id: UUID
    delegation_id: str = Field(min_length=1, max_length=200)
    kind: Literal["commentary", "thinking"]
    content: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_context_size(self) -> VoiceContextAppend:
        if not self.delegation_id.strip() or len(self.content.encode("utf-8")) > 500:
            raise ValueError("voice context must be correlated and at most 500 UTF-8 bytes")
        return self


class VoiceSessionContextAppend(ContractModel):
    """Bounded device-to-gateway context update for the active session."""

    schema_version: Literal[1] = 1
    session_id: UUID
    kind: Literal["commentary", "thinking", "instructions"]
    content: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_context_size(self) -> VoiceSessionContextAppend:
        if len(self.content.encode("utf-8")) > 500:
            raise ValueError("voice session context must fit 500 UTF-8 bytes")
        return self


ErrorCode = Literal[
    "AUTHENTICATION_REQUIRED",
    "PERMISSION_DENIED",
    "INVALID_REQUEST",
    "UNSUPPORTED_SCHEMA_VERSION",
    "RESOURCE_NOT_FOUND",
    "RESOURCE_CONFLICT",
    "IDEMPOTENCY_CONFLICT",
    "STALE_CONTEXT_REVISION",
    "APPROVAL_EXPIRED",
    "APPROVAL_PAYLOAD_MISMATCH",
    "ACTION_DENIED",
    "RATE_LIMITED",
    "BUDGET_EXCEEDED",
    "DEPENDENCY_UNAVAILABLE",
    "DEPENDENCY_TIMEOUT",
    "DEVICE_OFFLINE",
    "JOB_NOT_CANCELLABLE",
    "INTERNAL_ERROR",
]


class ApiError(ContractModel):
    code: ErrorCode
    message: str = Field(min_length=1)
    retryable: bool
    details: dict[str, Any] | None = None


class ApiErrorResponse(ContractModel):
    schema_version: Literal[1] = 1
    trace_id: UUID
    error: ApiError


ToolStatus = Literal["succeeded", "failed", "denied", "needs_reconciliation"]


class ToolResult(ContractModel):
    schema_version: Literal[1] = 1
    action_id: UUID
    trace_id: UUID
    device_id: str = Field(min_length=1)
    conversation_id: UUID
    session_id: UUID
    delegation_id: str = Field(min_length=1)
    context_revision: int = Field(ge=0)
    job_id: UUID | None = None
    turn_id: UUID | None = None
    status: ToolStatus
    verified: bool
    observed_state: dict[str, Any] | None = None
    sources: tuple[SourceRef, ...] = ()
    error_code: str | None = None
    retryable: bool
    completed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def verified_success(self) -> ToolResult:
        if self.status == "succeeded" and not self.verified:
            raise ValueError("successful tool results must be verified")
        return self
