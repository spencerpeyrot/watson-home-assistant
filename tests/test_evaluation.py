from __future__ import annotations

import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from evaluation.report import (
    EvaluationEnvironment,
    EvaluationObservation,
    EvaluationRun,
    _load_cases,
    build_report,
)
from service.job_graph import choose_route


CASES_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "cases.json"


class EvaluationReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = _load_cases(CASES_PATH)
        self.environment = EvaluationEnvironment(
            target="mock",
            hardware="fake provider; no audio hardware",
            network="loopback",
            model_ids=("fake-simple", "fake-complex"),
        )

    def test_seed_cases_match_router_and_report_reproducible_metrics(self) -> None:
        self.assertEqual(len(self.cases), 32)
        self.assertEqual(sum(case.expected_route == "simple" for case in self.cases), 16)
        self.assertEqual(sum(case.expected_route == "complex" for case in self.cases), 16)
        self.assertTrue(
            all(case.expected_route == choose_route(case.request_text) for case in self.cases)
        )
        run = EvaluationRun(
            run_id="synthetic-run-1",
            recorded_at=datetime(2026, 10, 7, tzinfo=timezone.utc),
            environment=self.environment,
            observations=(
                EvaluationObservation(
                    case_id="simple-arithmetic",
                    actual_route="simple",
                    succeeded=True,
                    latency_ms=100,
                    measured_cost_usd=Decimal("0.01"),
                ),
                EvaluationObservation(
                    case_id="simple-explanation",
                    actual_route="simple",
                    succeeded=False,
                    latency_ms=200,
                    measured_cost_usd=Decimal("0.03"),
                    correction_required=True,
                    false_wake_count=2,
                ),
                EvaluationObservation(
                    case_id="current-factual-search",
                    actual_route="complex",
                    succeeded=True,
                    latency_ms=300,
                    measured_cost_usd=Decimal("0"),
                ),
            ),
        )

        summary = build_report(self.cases, run).summary

        self.assertEqual(summary.expected_cases, 32)
        self.assertEqual(summary.observed_cases, 3)
        self.assertEqual(summary.route_accuracy, Decimal("1.000"))
        self.assertEqual(summary.task_success_rate, Decimal("0.667"))
        self.assertEqual(summary.source_required_cases, 9)
        self.assertEqual(summary.source_coverage_rate, Decimal("0.000"))
        self.assertEqual(summary.report_required_cases, 12)
        self.assertEqual(summary.report_completion_rate, Decimal("0.000"))
        self.assertEqual(summary.correction_rate, Decimal("0.333"))
        self.assertEqual(summary.false_wake_count, 2)
        self.assertEqual(summary.p50_latency_ms, Decimal("200"))
        self.assertEqual(summary.p95_latency_ms, Decimal("300"))
        self.assertEqual(summary.total_measured_cost_usd, Decimal("0.04"))
        self.assertEqual(summary.cost_per_successful_task_usd, Decimal("0.02"))
        self.assertEqual(len(summary.missing_case_ids), 29)
        self.assertEqual(
            set(summary.missing_case_ids),
            {case.case_id for case in self.cases}
            - {item.case_id for item in run.observations},
        )

    def test_missing_cost_stays_unknown_and_unknown_case_is_rejected(self) -> None:
        observation = EvaluationObservation(
            case_id="simple-arithmetic",
            actual_route="simple",
            succeeded=True,
            latency_ms=50,
        )
        run = EvaluationRun(
            run_id="synthetic-run-2",
            recorded_at=datetime(2026, 10, 7, tzinfo=timezone.utc),
            environment=self.environment,
            observations=(observation,),
        )
        summary = build_report(self.cases, run).summary
        self.assertIsNone(summary.total_measured_cost_usd)
        self.assertIsNone(summary.cost_per_successful_task_usd)
        self.assertEqual(summary.source_coverage_rate, Decimal("0.000"))
        self.assertEqual(summary.report_completion_rate, Decimal("0.000"))

        unknown_run = run.model_copy(
            update={
                "observations": (
                    observation.model_copy(update={"case_id": "unknown-case"}),
                )
            }
        )
        with self.assertRaisesRegex(ValueError, "unknown case IDs"):
            build_report(self.cases, unknown_run)

    def test_failed_attempt_cost_is_included_and_unknown_failure_cost_stays_unknown(self) -> None:
        success = EvaluationObservation(
            case_id="simple-arithmetic",
            actual_route="simple",
            succeeded=True,
            latency_ms=50,
            measured_cost_usd=Decimal("0.01"),
        )
        failure = EvaluationObservation(
            case_id="simple-explanation",
            actual_route="simple",
            succeeded=False,
            latency_ms=100,
            measured_cost_usd=Decimal("0.03"),
        )
        run = EvaluationRun(
            run_id="failed-attempt-cost",
            recorded_at=datetime(2026, 10, 7, tzinfo=timezone.utc),
            environment=self.environment,
            observations=(success, failure),
        )
        summary = build_report(self.cases, run).summary
        self.assertEqual(summary.cost_per_successful_task_usd, Decimal("0.04"))

        incomplete_run = run.model_copy(
            update={
                "observations": (
                    success,
                    failure.model_copy(update={"measured_cost_usd": None}),
                )
            }
        )
        incomplete_summary = build_report(self.cases, incomplete_run).summary
        self.assertIsNone(incomplete_summary.total_measured_cost_usd)
        self.assertIsNone(incomplete_summary.cost_per_successful_task_usd)

    def test_idle_false_wakes_count_without_a_task_observation(self) -> None:
        run = EvaluationRun(
            run_id="idle-false-wakes",
            recorded_at=datetime(2026, 10, 7, tzinfo=timezone.utc),
            environment=self.environment,
            observations=(),
            idle_false_wake_count=3,
        )

        summary = build_report(self.cases, run).summary

        self.assertEqual(summary.observed_cases, 0)
        self.assertEqual(summary.false_wake_count, 3)

    def test_required_deliverable_rates_count_missing_cases_as_uncovered(self) -> None:
        run = EvaluationRun(
            run_id="synthetic-deliverables",
            recorded_at=datetime(2026, 10, 7, tzinfo=timezone.utc),
            environment=self.environment,
            observations=(
                EvaluationObservation(
                    case_id="current-factual-search",
                    actual_route="complex",
                    succeeded=True,
                    latency_ms=200,
                    sources_provided=True,
                ),
                EvaluationObservation(
                    case_id="comparison-report",
                    actual_route="complex",
                    succeeded=True,
                    latency_ms=400,
                    sources_provided=False,
                    report_saved=True,
                ),
            ),
        )

        summary = build_report(self.cases, run).summary

        self.assertEqual(summary.source_required_cases, 9)
        self.assertEqual(summary.source_coverage_rate, Decimal("0.111"))
        self.assertEqual(summary.report_required_cases, 12)
        self.assertEqual(summary.report_completion_rate, Decimal("0.083"))
        self.assertEqual(len(summary.missing_case_ids), 30)
        self.assertEqual(
            set(summary.missing_case_ids),
            {case.case_id for case in self.cases}
            - {item.case_id for item in run.observations},
        )

        cases_without_deliverables = tuple(
            case.model_copy(update={"requires_sources": False, "requires_report": False})
            for case in self.cases
        )
        no_requirement_summary = build_report(cases_without_deliverables, run).summary
        self.assertIsNone(no_requirement_summary.source_coverage_rate)
        self.assertIsNone(no_requirement_summary.report_completion_rate)

    def test_duplicate_observation_ids_are_rejected(self) -> None:
        observation = EvaluationObservation(
            case_id="simple-arithmetic",
            actual_route="simple",
            succeeded=True,
            latency_ms=50,
            measured_cost_usd=Decimal("0"),
        )
        with self.assertRaisesRegex(ValueError, "one observation per case"):
            EvaluationRun(
                run_id="duplicate-run",
                recorded_at=datetime(2026, 10, 7, tzinfo=timezone.utc),
                environment=self.environment,
                observations=(observation, observation),
            )


if __name__ == "__main__":
    unittest.main()
