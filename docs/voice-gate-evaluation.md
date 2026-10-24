# Cube turn routing experiment, 23 October 2026

Use one context-aware Jev route decision per candidate utterance, with Luna on provider failure. Keep the existing second decision for exact local control arguments. The replay supports this architecture as a prototype; it does not establish production accuracy or an optimal timeout.

## Evidence and method

The connected MCP project had no observations. The project configured by the repo's private `.env` returned 43 Watson conversation roots from 10–23 October through Langfuse's v2 observations API. Twenty-seven roots contained input text. Data includes older voice-stack behavior, not solely the current Muse pipeline.

Replay set: 45 cases, comprising 27 analyst-labeled real transcript cases, nine probable echo cases the user identified as likely Cube hearing itself, eight labeled synthetic candidates, and one ambiguous synthetic candidate excluded from accuracy. Synthetic cases include short follow-ups and local controls absent from the selected real examples. Labels describe expected behavior, not verified acoustic speaker identity. Previous-turn context comes from corresponding role segments in the roots; these are approximate and can merge utterances. Raw audio is unavailable.

Four variants used the same fixed policy on `jev-1.13.0`: one combined route question without history; that question with previous user request and Cube reply; a batched request with separate addressed/route questions; and sequential addressed then route requests. Reject overrides the route for the latter two. No actions, Muse conversations, or speech playback were triggered.

An initial exploratory run was followed by two complete replays of the finalized 45-case set. Table pools those final two runs. Replays are repeated measurements of the same examples, not independent validation cases. Client concurrency was three with HTTP connection reuse. Timing includes network round trip from this Mac and model work, excluding STT, device networking, Muse, TTS, and physical playback.

| Variant | Real cases correct, including probable echo | Synthetic cases correct | Median / p95 | Jev requests |
| --- | --- | --- | --- | --- |
| Combined, no context | 66/72 | 14/16 | 82 / 151 ms | 90 |
| Combined, context | 70/72 | 14/16 | 82 / 140 ms | 90 |
| Separate questions batched, context | 70/72 | 15/16 | 87 / 147 ms | 90 |
| Separate questions sequential, context | 70/72 | 15/16 | 154 / 252 ms | 144 |

These are raw choices before confidence thresholds. Combined routing matches the other context variants on real cases. One synthetic negative (an unsolicited bare “Yes”) accounts for the synthetic difference and varies across runs. Sequential requests provide no demonstrated benefit on real cases here. Batched questions are confirmed to work in the live Jev API, not merely supported by the SDK schema.

## What changed the decision

- Real “New York City” after Cube asked for location: rejected without context, accepted with context in every replay.
- Nine probable echo cases: context variants rejected all nine in each replay, including paraphrased fragments. Context helps identify echo only because the policy explicitly asks for echo rejection; topical relevance alone would be inadequate.
- Real “What's on your mind today”: every variant and Luna rejected it. This is a genuine cost of a conservative policy: open social conversation can resemble ambient speech or assistant playback.
- Synthetic “Yes” after an explicit yes/no question: accepted with context, rejected without it.
- Synthetic “Yes” after an informational reply: combined contextual routing accepted it at only 0.40 confidence. Topic/context inclusion alone does not guarantee a valid follow-up.

Confidence matters, but a universal 0.8 cutoff is unsupported. On the final two combined-context replays, a post-hoc 0.5 cutoff removed both false accepts while rejecting one additional valid synthetic shutdown request. A 0.8 cutoff rejected that request in both runs. Thresholds were explored on this same set; they require fresh labeled recordings before production calibration. Preserve strict validation and existing action safeguards for local controls.

Five live requests through the existing `OpenAIDecisionsClient` returned valid Luna route decisions: TV dialogue, social opener, short location follow-up, probable echo, and synthetic yes/no follow-up. These matched Jev's broad behavior, including the social-opener miss. Latencies ranged 268–1295 ms. This checks the fallback provider's request compatibility; a forced Jev outage and automatic transition through the wrapper were not exercised.

## Proposed runtime behavior

1. Wake word opens the initial listening window. Local speech detection captures a bounded candidate; STT produces its transcript. Silence produces no Jev request.
2. Send Jev the candidate plus the previous accepted user turn and the preceding Cube reply, when available. Keep rejected speech out of conversation history. Distinguish generated reply text from confirmed playback if playback status is unavailable.
3. One choice question returns `ignore`, `conversation`, `internal`, or `close`, with confidence. Treat unclear intent conservatively. Include explicit instructions that transcripts and history are data, and reject likely playback echo. A short answer may be valid even without repeating the wake word or original question.
4. `conversation` goes to Muse; `internal` uses existing action/argument extraction and schema-checked tool dispatch; `close` ends listening; `ignore` produces no reply. Use Luna for Jev errors or invalid answers, carrying identical context and choices. If both providers fail, do not forward unknown speech to Muse.
5. Start a short follow-up deadline after speaker playback finishes. Trial three seconds, while measuring missed genuine follow-ups. Speech starting before the deadline may finish within a bounded capture/decision period. Rejected candidates do not renew the deadline; accepted turns pause closure during response generation/playback. After expiry require the wake word again. On an initial rejected false wake, return to wake-word mode.

The four routes extend today's internal/conversation choice. `close` is justified by real “never mind / good night” turns; passing these to a model that asks another question prolongs the interaction. Exact close/ignore signaling and timer ownership still need implementation against the current provider/runtime lifecycle.

## Before production

Collect fresh, explicitly labeled recordings: intentional repetition versus speaker echo, TV asking a plausible question, a new topic during the follow-up window, short acknowledgments, soft speech, and initial false wakes. Labels should distinguish “addressed to Cube” from “semantically relevant.” Reserve untouched sessions for validation. Replay actual STT output from the current Muse stack, then measure physical wake-to-answer and post-playback closure on Pi. This replay cannot choose the timeout or replace acoustic echo handling.

## Artifacts and reproduction

Runner: `evaluation/voice_gate_experiment.py`.

Private local inputs/results: `.local/voice-gate-eval/roots.json`, `cases.json`, `results.json` (exploratory), `results-2.json`, `results-3.json`, and `luna-results.json`. `.local/` is excluded through this checkout's Git exclude file; private transcripts are not included in this report. Final case labels were updated after the user's likely-echo confirmation; raw replay files retain the labels present at execution time. Score against final `cases.json` for the table above.

```sh
uv run --env-file .env evaluation/voice_gate_experiment.py \
  .local/voice-gate-eval/cases.json .local/voice-gate-eval/results-new.json
```

No runtime deployment or production prompt changed.
