"""Per-task callback for usage confirmed before a task completes or is cancelled."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar, Token

from contracts.models import UsageMeasurement


UsageReporter = Callable[[UsageMeasurement], None]
_USAGE_REPORTER: ContextVar[UsageReporter | None] = ContextVar(
    "cube_task_usage_reporter", default=None
)


def set_usage_reporter(reporter: UsageReporter | None) -> Token:
    return _USAGE_REPORTER.set(reporter)


def reset_usage_reporter(token: Token) -> None:
    _USAGE_REPORTER.reset(token)


def report_usage(measurement: UsageMeasurement) -> None:
    reporter = _USAGE_REPORTER.get()
    if reporter is not None:
        reporter(measurement)
