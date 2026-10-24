"""Validate captured evaluation results and compute reproducible metrics."""

from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvaluationCase(EvaluationModel):
    case_id: str = Field(min_length=1)
    request_text: str = Field(min_length=1, max_length=8_000)
    expected_route: Literal["simple", "complex"]
    requires_sources: bool = False
    requires_report: bool = False
    tags: tuple[str, ...] = ()


class EvaluationObservation(EvaluationModel):
    case_id: str = Field(min_length=1)
    actual_route: Literal["simple", "complex"]
    succeeded: bool
    latency_ms: Decimal = Field(ge=0)
    measured_cost_usd: Decimal | None = Field(default=None, ge=0)
    correction_required: bool = False
    false_wake_count: int = Field(default=0, ge=0)
    sources_provided: bool = False
    report_saved: bool = False


class EvaluationEnvironment(EvaluationModel):
    target: Literal["mock", "development", "pi", "cloud"]
    hardware: str = Field(min_length=1, max_length=200)
    network: str = Field(min_length=1, max_length=120)
    model_ids: tuple[str, ...] = ()


class EvaluationRun(EvaluationModel):
    run_id: str = Field(min_length=1, max_length=120)
    recorded_at: AwareDatetime
    environment: EvaluationEnvironment
    observations: tuple[EvaluationObservation, ...]
    idle_false_wake_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def unique_observations(self) -> EvaluationRun:
        case_ids = [item.case_id for item in self.observations]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("each evaluation run may contain one observation per case")
        return self


class EvaluationSummary(EvaluationModel):
    expected_cases: int = Field(ge=0)
    observed_cases: int = Field(ge=0)
    missing_case_ids: tuple[str, ...] = ()
    route_accuracy: Decimal | None = Field(default=None, ge=0, le=1)
    task_success_rate: Decimal | None = Field(default=None, ge=0, le=1)
    source_required_cases: int = Field(default=0, ge=0)
    source_coverage_rate: Decimal | None = Field(default=None, ge=0, le=1)
    report_required_cases: int = Field(default=0, ge=0)
    report_completion_rate: Decimal | None = Field(default=None, ge=0, le=1)
    correction_rate: Decimal | None = Field(default=None, ge=0, le=1)
    false_wake_count: int = Field(default=0, ge=0)
    p50_latency_ms: Decimal | None = Field(default=None, ge=0)
    p95_latency_ms: Decimal | None = Field(default=None, ge=0)
    total_measured_cost_usd: Decimal | None = Field(default=None, ge=0)
    cost_per_successful_task_usd: Decimal | None = Field(default=None, ge=0)


class EvaluationReport(EvaluationModel):
    run_id: str
    recorded_at: AwareDatetime
    environment: EvaluationEnvironment
    summary: EvaluationSummary


def _rate(numerator: int, denominator: int) -> Decimal | None:
    if denominator == 0:
        return None
    return (Decimal(numerator) / Decimal(denominator)).quantize(Decimal("0.001"))


def _nearest_rank(values: list[Decimal], percentile: Decimal) -> Decimal | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(float(percentile) * len(ordered)) - 1)
    return ordered[index]


def build_report(
    cases: tuple[EvaluationCase, ...], run: EvaluationRun
) -> EvaluationReport:
    """Calculate task, route, latency, correction, false-wake, and cost metrics."""

    case_by_id = {case.case_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("evaluation case IDs must be unique")
    unknown = sorted(
        {item.case_id for item in run.observations} - set(case_by_id)
    )
    if unknown:
        raise ValueError(f"run contains unknown case IDs: {', '.join(unknown)}")
    observed = run.observations
    count = len(observed)
    successful = [item for item in observed if item.succeeded]
    costs_complete = all(item.measured_cost_usd is not None for item in observed)
    total_cost = (
        sum((item.measured_cost_usd for item in observed), Decimal("0"))
        if costs_complete
        else None
    )
    cost_per_success = (
        total_cost / Decimal(len(successful))
        if successful and total_cost is not None
        else None
    )
    route_correct = sum(
        item.actual_route == case_by_id[item.case_id].expected_route
        for item in observed
    )
    observed_by_id = {item.case_id: item for item in observed}
    source_cases = [case for case in cases if case.requires_sources]
    report_cases = [case for case in cases if case.requires_report]
    source_covered = sum(
        observed_by_id[case.case_id].sources_provided
        for case in source_cases
        if case.case_id in observed_by_id
    )
    reports_saved = sum(
        observed_by_id[case.case_id].report_saved
        for case in report_cases
        if case.case_id in observed_by_id
    )
    summary = EvaluationSummary(
        expected_cases=len(cases),
        observed_cases=count,
        missing_case_ids=tuple(sorted(set(case_by_id) - {item.case_id for item in observed})),
        route_accuracy=_rate(route_correct, count),
        task_success_rate=_rate(len(successful), count),
        source_required_cases=len(source_cases),
        source_coverage_rate=_rate(source_covered, len(source_cases)),
        report_required_cases=len(report_cases),
        report_completion_rate=_rate(reports_saved, len(report_cases)),
        correction_rate=_rate(sum(item.correction_required for item in observed), count),
        false_wake_count=run.idle_false_wake_count
        + sum(item.false_wake_count for item in observed),
        p50_latency_ms=_nearest_rank(
            [item.latency_ms for item in observed], Decimal("0.50")
        ),
        p95_latency_ms=_nearest_rank(
            [item.latency_ms for item in observed], Decimal("0.95")
        ),
        total_measured_cost_usd=total_cost,
        cost_per_successful_task_usd=cost_per_success,
    )
    return EvaluationReport(
        run_id=run.run_id,
        recorded_at=run.recorded_at,
        environment=run.environment,
        summary=summary,
    )


def _load_cases(path: Path) -> tuple[EvaluationCase, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return tuple(EvaluationCase.model_validate(item) for item in payload)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"evaluation cases are invalid ({type(error).__name__})") from None


def _load_run(path: Path) -> EvaluationRun:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return EvaluationRun.model_validate(payload)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"evaluation run is invalid ({type(error).__name__})") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = build_report(_load_cases(args.cases), _load_run(args.run))
    except ValueError as error:
        parser.error(str(error))
    serialized = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
