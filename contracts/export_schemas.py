"""Export checked-in JSON Schema from the Pydantic contract models."""

import json
from pathlib import Path

from .models import (
    ApiErrorResponse,
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    ApprovalListResponse,
    ApprovalRecord,
    AudioFrameMetadata,
    AudioOutputFrameMetadata,
    AudioTurnEnd,
    DeviceHealthEvent,
    HealthResponse,
    JobAccepted,
    JobCorrectionRequest,
    JobEvent,
    JobListResponse,
    JobSnapshot,
    LocalToolRequest,
    MemoryRecord,
    MemoryListResponse,
    MemoryUpdateRequest,
    MemoryWriteRequest,
    SessionCreateRequest,
    SessionListResponse,
    SessionResponse,
    PlaybackMeterEvent,
    TaskRequest,
    TimerCreateRequest,
    TimerFiredEvent,
    TimerEventAck,
    TimerListResponse,
    TimerRecord,
    ToolResult,
    UsageRecord,
    UsageSummary,
    UsageResponse,
    VoiceContextAppend,
    VoiceSessionContextAppend,
    VoiceSessionEvent,
)


SCHEMAS_DIR = Path(__file__).parent / "schemas" / "v1"
SCHEMA_MODELS = (
    TaskRequest,
    SessionCreateRequest,
    SessionResponse,
    SessionListResponse,
    JobAccepted,
    JobSnapshot,
    JobListResponse,
    JobCorrectionRequest,
    JobEvent,
    ApprovalDecisionRequest,
    ApprovalRecord,
    ApprovalListResponse,
    ApprovalDecisionResponse,
    MemoryRecord,
    MemoryWriteRequest,
    MemoryUpdateRequest,
    MemoryListResponse,
    UsageRecord,
    UsageSummary,
    UsageResponse,
    TimerCreateRequest,
    TimerListResponse,
    TimerRecord,
    TimerFiredEvent,
    TimerEventAck,
    HealthResponse,
    DeviceHealthEvent,
    PlaybackMeterEvent,
    LocalToolRequest,
    AudioFrameMetadata,
    AudioOutputFrameMetadata,
    AudioTurnEnd,
    VoiceSessionEvent,
    VoiceContextAppend,
    VoiceSessionContextAppend,
    ToolResult,
    ApiErrorResponse,
)


def main() -> None:
    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    for model in SCHEMA_MODELS:
        schema = model.model_json_schema(mode="serialization")
        schema["$id"] = f"https://cube-assistant.invalid/contracts/v1/{model.__name__}.schema.json"
        (SCHEMAS_DIR / f"{model.__name__}.schema.json").write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
