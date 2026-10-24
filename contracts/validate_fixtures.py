"""Validate checked-in JSON fixtures against their Pydantic contracts."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from .export_schemas import SCHEMAS_DIR, SCHEMA_MODELS
from .models import (
    ApiErrorResponse,
    ApprovalDecisionRequest,
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
    MemoryWriteRequest,
    SessionCreateRequest,
    SessionListResponse,
    SessionResponse,
    TaskRequest,
    TimerCreateRequest,
    TimerEventAck,
    TimerFiredEvent,
    TimerListResponse,
    TimerRecord,
    ToolResult,
    UsageResponse,
    VoiceContextAppend,
    VoiceSessionContextAppend,
    VoiceSessionEvent,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_MODELS: dict[str, type[BaseModel]] = {
    "api_error.json": ApiErrorResponse,
    "approval_decision.json": ApprovalDecisionRequest,
    "approval_record.json": ApprovalRecord,
    "audio_frame_metadata.json": AudioFrameMetadata,
    "audio_output_frame_metadata.json": AudioOutputFrameMetadata,
    "audio_turn_end.json": AudioTurnEnd,
    "device_health.json": DeviceHealthEvent,
    "health_response.json": HealthResponse,
    "job_accepted.json": JobAccepted,
    "job_correction.json": JobCorrectionRequest,
    "job_event.progress.json": JobEvent,
    "job_list_response.json": JobListResponse,
    "job_snapshot.json": JobSnapshot,
    "local_tool_request.json": LocalToolRequest,
    "memory_record.json": MemoryRecord,
    "memory_write.json": MemoryWriteRequest,
    "session_create.json": SessionCreateRequest,
    "session_list_response.json": SessionListResponse,
    "session_response.json": SessionResponse,
    "task_request.json": TaskRequest,
    "timer_create.json": TimerCreateRequest,
    "timer_event_ack.json": TimerEventAck,
    "timer_fired_event.json": TimerFiredEvent,
    "timer_list_response.json": TimerListResponse,
    "timer_record.json": TimerRecord,
    "tool_result.uncertain_write.json": ToolResult,
    "tool_result.verified.json": ToolResult,
    "usage_response.json": UsageResponse,
    "voice_context_append.json": VoiceContextAppend,
    "voice_session_context_append.json": VoiceSessionContextAppend,
    "voice_session_event.delegation.json": VoiceSessionEvent,
    "voice_session_event.error.json": VoiceSessionEvent,
    "voice_session_event.ready.json": VoiceSessionEvent,
    "voice_session_event.transcript.json": VoiceSessionEvent,
    "voice_session_event.closed.json": VoiceSessionEvent,
    "voice_session_event.processing.json": VoiceSessionEvent,
    "voice_session_event.response_started.json": VoiceSessionEvent,
    "voice_session_event.response_complete.json": VoiceSessionEvent,
    "voice_session_event.ignored.json": VoiceSessionEvent,
}


def main() -> int:
    fixture_names = {path.name for path in FIXTURES_DIR.glob("*.json")}
    expected_names = set(FIXTURE_MODELS)
    missing = sorted(expected_names - fixture_names)
    unmapped = sorted(fixture_names - expected_names)
    if missing or unmapped:
        for name in missing:
            print(f"missing fixture: {name}")
        for name in unmapped:
            print(f"fixture has no model mapping: {name}")
        return 1

    errors: list[str] = []
    for name, model in sorted(FIXTURE_MODELS.items()):
        try:
            payload = json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))
            model.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            errors.append(f"{name}: {type(error).__name__}")

    for model in SCHEMA_MODELS:
        schema = model.model_json_schema(mode="serialization")
        schema["$id"] = f"https://cube-assistant.invalid/contracts/v1/{model.__name__}.schema.json"
        expected = json.dumps(schema, indent=2, sort_keys=True) + "\n"
        schema_path = SCHEMAS_DIR / f"{model.__name__}.schema.json"
        try:
            actual = schema_path.read_text(encoding="utf-8")
        except OSError:
            errors.append(f"{schema_path.name}: missing generated schema")
            continue
        if actual != expected:
            errors.append(f"{schema_path.name}: stale generated schema")

    if errors:
        for error in errors:
            print(error)
        return 1

    print(
        f"Validated {len(FIXTURE_MODELS)} JSON fixtures and "
        f"{len(SCHEMA_MODELS)} generated JSON Schemas."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
