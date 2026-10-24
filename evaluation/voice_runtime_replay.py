"""Replay private cases through production routing; device actions are simulated.

uv run --env-file .env python -m evaluation.voice_runtime_replay CASES OUTPUT
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from service.voice_commands import (
    FallbackDecisionsClient, JevDecisionsClient, OpenAIDecisionsClient,
    VoiceCommandFlow,
)


class SimulatedTools:
    """No device transport or ledger writes exist in this replay."""

    def __init__(self):
        self.requested = []

    async def dispatch(self, **request):
        self.requested.append(request['tool_name'])
        return SimpleNamespace(status='failed', verified=False)


async def replay(cases, decisions):
    semaphore = asyncio.Semaphore(3)

    async def trial(case):
        async with semaphore:
            tools = SimulatedTools()
            flow = VoiceCommandFlow(
                decisions=decisions,
                principal=SimpleNamespace(role='device', owner_id=uuid4(), device_id='simulation'),
                session=SimpleNamespace(session_id=uuid4(), conversation_id=uuid4()),
                tool_registry=tools, ledger=None,
            )
            started = time.perf_counter()
            mode = case.get('mode', 'route')
            if mode == 'sequence':
                previous_user = previous_reply = None
                steps = []
                for index, step in enumerate(case['turns']):
                    decision = await flow.handle(
                        step['current_transcript'], uuid4(), previous_user, previous_reply
                    )
                    steps.append(dict(
                        id=f"{case['id']}:{index + 1}", mode='route',
                        expected=step.get('expected_route'), prediction=decision.route,
                    ))
                    if decision.route != 'conversation' or not step.get('cube_reply'):
                        continue
                    expectation = await flow.classify_reply_expectation(
                        user_turn=step['current_transcript'], reply=step['cube_reply']
                    )
                    if step.get('expected_expectation') is not None:
                        steps.append(dict(
                            id=f"{case['id']}:{index + 1}", mode='reply_expectation',
                            expected=step['expected_expectation'], prediction=expectation,
                        ))
                    previous_user = step['current_transcript']
                    previous_reply = step['cube_reply']
            elif mode == 'reply_expectation':
                prediction = await flow.classify_reply_expectation(
                    user_turn=case['user_turn'], reply=case['cube_reply']
                )
            else:
                if case.get('previous_reply_expectation') in {
                    'none', 'open_answer', 'confirmation', 'closing'
                }:
                    flow._last_reply_expectation = case['previous_reply_expectation']
                elif case.get('previous_reply'):
                    flow._last_reply_expectation = await flow.classify_reply_expectation(
                        user_turn=case.get('previous_user') or '',
                        reply=case['previous_reply'],
                    )
                decision = await flow.handle(
                    case['current_transcript'], uuid4(),
                    case.get('previous_user'), case.get('previous_reply'),
                )
                prediction = decision.route
            result = dict(id=case['id'], mode=mode,
                          simulated_tools=tools.requested,
                          seconds=time.perf_counter() - started)
            if mode == 'sequence':
                result['steps'] = steps
            else:
                result.update(expected=case.get('expected'), prediction=prediction)
            return result

    return await asyncio.gather(*(trial(case) for case in cases))


def summarize(results):
    decisions = [
        step
        for result in results
        for step in (result['steps'] if result.get('mode') == 'sequence' else [result])
    ]
    scored = [r for r in decisions if r['expected'] is not None]
    route_scored = [r for r in scored if r.get('mode', 'route') == 'route']
    expectation_scored = [r for r in scored if r.get('mode') == 'reply_expectation']
    accepted = {'conversation', 'internal'}
    return dict(cases=len(results), scored=len(scored),
                behaviorally_correct=sum(r['prediction'] == r['expected'] for r in scored),
                route_scored=len(route_scored),
                route_labels_correct=sum(r['prediction'] == r['expected'] for r in route_scored),
                false_accepts=sum(r['expected'] == 'ignore' and r['prediction'] in accepted for r in route_scored),
                missed_requests=sum(r['expected'] in accepted and r['prediction'] in {'ignore', 'close'} for r in route_scored),
                false_closures=sum(r['expected'] == 'ignore' and r['prediction'] == 'close' for r in route_scored),
                missed_closures=sum(r['expected'] == 'close' and r['prediction'] != 'close' for r in route_scored),
                expectation_scored=len(expectation_scored),
                expectation_labels_correct=sum(r['prediction'] == r['expected'] for r in expectation_scored),
                median_seconds=round(statistics.median(r['seconds'] for r in results), 3),
                device_actions_executed=0)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    if not cases:
        parser.error('cases must not be empty')
    decisions = FallbackDecisionsClient(
        JevDecisionsClient(os.environ['TYPESAFE_API_KEY']),
        OpenAIDecisionsClient(os.environ['OPENAI_API_KEY']),
    )
    results = await replay(cases, decisions)
    summary = summarize(results)
    # Output includes only case IDs and decisions; private source text stays in cases.
    args.output.write_text(json.dumps(dict(summary=summary, results=results), indent=2))
    args.output.chmod(0o600)
    print(json.dumps(summary))


if __name__ == '__main__':
    asyncio.run(main())
