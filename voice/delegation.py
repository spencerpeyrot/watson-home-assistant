"""Join finalized transcript fragments with client-delegation events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from uuid import UUID, uuid4

from contracts.models import RecentTurn, TaskRequest

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommittedDelegation:
    delegation_id: str
    offset_ms: int
    transcript: str
    committed_at: datetime


class DelegationTranscriptBuffer:
    """Commit one user turn only after audio ends and transcript goes quiet."""

    def __init__(self, *, transcript_quiet_seconds: float = 1.0) -> None:
        if transcript_quiet_seconds <= 0:
            raise ValueError("transcript quiet interval must be positive")
        self._quiet_seconds = transcript_quiet_seconds
        self._fragments: list[str] = []
        self._last_delta_at: float | None = None
        self._audio_finished = False
        self._delegation: tuple[str, int] | None = None
        self._seen_delegations: set[str] = set()
        self._transcript_cutoff_ms: int | None = None

    def record_transcript_delta(self, text: str, *, received_at: float, start_ms: int | None = None) -> bool:
        if received_at < 0:
            raise ValueError("transcript event time must be monotonic and nonnegative")
        if not text:
            return False
        if self._transcript_cutoff_ms is not None and start_ms is not None and start_ms >= self._transcript_cutoff_ms:
            _LOGGER.info("voice transcript rejected chars=%d start_ms=%s cutoff_ms=%s", len(text), start_ms, self._transcript_cutoff_ms)
            return False
        self._fragments.append(text)
        self._last_delta_at = received_at
        _LOGGER.info("voice transcript accepted chars=%d start_ms=%s cutoff_ms=%s fragments=%d", len(text), start_ms, self._transcript_cutoff_ms, len(self._fragments))
        return True

    def record_delegation(
        self, *, delegation_id: str, target: str, offset_ms: int
    ) -> None:
        if target != "client":
            return
        if not delegation_id.strip() or offset_ms < 0:
            raise ValueError("client delegation identity or offset is invalid")
        if delegation_id in self._seen_delegations or self._delegation is not None:
            raise ValueError("duplicate or overlapping client delegation")
        self._delegation = (delegation_id, offset_ms)
        self._transcript_cutoff_ms = offset_ms

    def mark_audio_finished(self) -> None:
        self._audio_finished = True

    @property
    def has_delegation(self) -> bool:
        """Whether a client delegation is waiting for this turn's transcript."""

        return self._delegation is not None

    @property
    def transcript(self) -> str:
        """Current exact fragments, kept in memory only."""
        return "".join(self._fragments).strip()

    def ready(self, *, now: float) -> CommittedDelegation | None:
        """Return committed exact transcript after sender completion and quiet gap."""

        if not self._audio_finished or self._delegation is None or self._last_delta_at is None:
            return None
        if now < self._last_delta_at:
            raise ValueError("commit time precedes last transcript fragment")
        if now - self._last_delta_at < self._quiet_seconds:
            return None
        transcript = "".join(self._fragments).strip()
        if not transcript:
            return None
        delegation_id, offset_ms = self._delegation
        result = CommittedDelegation(
            delegation_id=delegation_id,
            offset_ms=offset_ms,
            transcript=transcript,
            committed_at=datetime.now(timezone.utc),
        )
        self._seen_delegations.add(delegation_id)
        self._fragments.clear()
        self._last_delta_at = None
        self._audio_finished = False
        self._delegation = None
        self._transcript_cutoff_ms = None
        return result

    def reset_turn(self) -> None:
        """Drop incomplete transcript state; retain seen IDs for this session."""

        self._fragments.clear()
        self._last_delta_at = None
        self._audio_finished = False
        self._delegation = None
        self._transcript_cutoff_ms = None


def build_task_request(
    committed: CommittedDelegation,
    *,
    device_id: str,
    conversation_id: UUID,
    session_id: UUID,
    context_revision: int,
    locale: str,
    timezone_name: str,
    recent_turns: tuple[RecentTurn, ...] = (),
    turn_id: UUID | None = None,
    trace_id: UUID | None = None,
) -> TaskRequest:
    """Build a correlated request from committed transcript and server context."""

    new_turn_id = turn_id or uuid4()
    current_turn = RecentTurn(
        turn_id=new_turn_id,
        role="user",
        text=committed.transcript,
        committed_at=committed.committed_at,
    )
    return TaskRequest(
        device_id=device_id,
        conversation_id=conversation_id,
        session_id=session_id,
        turn_id=new_turn_id,
        delegation_id=committed.delegation_id,
        trace_id=trace_id or uuid4(),
        request_text=committed.transcript,
        context_revision=context_revision,
        recent_turns=(*recent_turns, current_turn),
        locale=locale,
        timezone=timezone_name,
        idempotency_key=f"{device_id}/{session_id}/{committed.delegation_id}",
    )
