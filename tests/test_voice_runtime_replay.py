import json
import unittest

from evaluation.voice_runtime_replay import replay, summarize


class RuntimeReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_sequence_replay_keeps_context_after_ignored_turn(self):
        class Decisions:
            def __init__(self):
                self.inputs = []

            async def decide(self, *, input_text, questions):
                state = json.loads(input_text)
                self.inputs.append((state, questions[0]['name']))
                if questions[0]['name'] == 'expectation':
                    choice = 'open_answer' if 'mind' in state['cube_reply'] else 'none'
                    return {'expectation': (choice, 0.95)}
                current = state['current_transcript']
                return {'route': (('ignore' if current == 'Yes' else 'conversation'), 0.95)}

        decisions = Decisions()
        results = await replay([{
            'id': 'social', 'mode': 'sequence', 'turns': [
                {'current_transcript': 'Hey Watson, how are you?',
                 'expected_route': 'conversation', 'cube_reply': 'Doing well. What’s on your mind?',
                 'expected_expectation': 'open_answer'},
                {'current_transcript': 'Just wanted to say hi.',
                 'expected_route': 'conversation', 'cube_reply': 'Good to hear from you.',
                 'expected_expectation': 'none'},
                {'current_transcript': 'Yes', 'expected_route': 'ignore'},
                {'current_transcript': 'What is the weather?', 'expected_route': 'conversation'},
            ],
        }], decisions)

        self.assertEqual([step['prediction'] for step in results[0]['steps']], [
            'conversation', 'open_answer', 'conversation', 'none', 'ignore', 'conversation'
        ])
        last_route_input = next(
            state for state, name in reversed(decisions.inputs)
            if name == 'route' and state['current_transcript'] == 'What is the weather?'
        )
        self.assertEqual(last_route_input['previous_user'], 'Just wanted to say hi.')
        self.assertEqual(last_route_input['previous_cube_reply'], 'Good to hear from you.')
        self.assertEqual(last_route_input['previous_reply_expectation'], 'none')
        summary = summarize(results)
        self.assertEqual(summary['route_scored'], 4)
        self.assertEqual(summary['route_labels_correct'], 4)
        self.assertEqual(summary['expectation_labels_correct'], 2)

    async def test_local_controls_use_simulated_tools_and_ignored_speech_does_not(self):
        class Decisions:
            async def decide(self, *, input_text, questions):
                if questions[0]['name'] == 'route':
                    return {'route': ('ignore' if 'television' in input_text else 'internal', 0.99)}
                return {'action': ('read_volume', 0.99)}

        results = await replay([
            dict(id='control', current_transcript='What volume is Cube?', expected='internal'),
            dict(id='noise', current_transcript='television dialogue', expected='ignore'),
        ], Decisions())

        self.assertEqual(results[0]['simulated_tools'], ['get_speaker_volume'])
        self.assertEqual(results[1]['simulated_tools'], [])
        self.assertEqual(summarize(results)['behaviorally_correct'], 2)
        self.assertEqual(summarize(results)['false_accepts'], 0)

    async def test_reply_expectations_run_through_production_decision_flow(self):
        class Decisions:
            async def decide(self, *, input_text, questions):
                self.input_text = input_text
                return {'expectation': ('confirmation', 0.95)}

        results = await replay([
            dict(id='asked', mode='reply_expectation', user_turn='Need more?',
                 cube_reply='Would you like tomorrow too?',
                 expected='confirmation'),
        ], Decisions())

        self.assertEqual(results[0]['prediction'], 'confirmation')
        self.assertEqual(summarize(results)['expectation_labels_correct'], 1)
        self.assertEqual(results[0]['simulated_tools'], [])

    def test_ignore_and_close_have_different_session_behavior(self):
        metrics = summarize([
            dict(expected='ignore', prediction='close', seconds=0.1),
            dict(expected='close', prediction='ignore', seconds=0.1),
        ])
        self.assertEqual(metrics['route_labels_correct'], 0)
        self.assertEqual(metrics['behaviorally_correct'], 0)
        self.assertEqual(metrics['false_closures'], 1)
        self.assertEqual(metrics['missed_closures'], 1)
