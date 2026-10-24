# Plan: natural, bounded conversations with Watson

Status: conversational routing, invitation-based listening, and one bounded recovery prompt are active on the Pi canary. Offline replay and lifecycle checks pass; controlled audio acceptance remains pending. Interruption behavior stays gated on echo testing.

## Desired experience

Watson should recognize social conversation, remember the exchange, allow time to answer its questions, and quietly return to wake-word mode when the person is done. Silence should not produce another spoken message.

| Exchange | Expected behavior |
| --- | --- |
| Person: “Hey Watson, how are you?” Watson: “Doing well. What’s on your mind?” | Treat the greeting as intentional conversation. Expect an open-ended answer and allow a longer reply window. |
| Person: “Just wanted to say hi.” Watson: “Hey, good to hear from you.” | Recognize small talk. Allow a short continuation window without forcing another question. |
| Watson: “Which city?” Person: “New York.” | Accept the brief answer using conversation context. |
| Watson: “Want tomorrow’s forecast too?” Person: “Yes.” | Accept the confirmation because Watson invited it. |
| Watson gives an appointment time; background audio transcribes as “Yes.” | Ignore uncertain speech and keep the original listening deadline. Do not generate a reply. |
| Person: “Thanks, that’s all.” | End naturally. Any goodbye should be brief, with no new invitation. |
| Person stays quiet after “What’s on your mind?” | Close silently when the longer window expires. No repeated “Are you there?” |
| Person: “And Friday?” after a calendar answer | Carry the calendar context into the next accepted turn. |

## Roles and conversation state

- Muse writes spoken replies and carries conversation context.
- Jev classifies intentional input and the kind of answer Watson expects. Luna remains the fallback for unavailable or invalid Jev results.
- The Pi manages capture, playback, pauses, and listening deadlines.

Use explicit reply expectations: `none`, `open_answer`, `confirmation`, or `closing`. These describe what the actual reply invites, including an invitation without a question mark, such as “Tell me what you need.” A rhetorical or quoted question does not automatically invite a reply.

Keep incoming routing separate: `ignore`, `conversation`, `internal`, or `close`. Greetings and intentional social comments belong to `conversation`; they need not contain a task or question. Short answers require the last accepted exchange plus the current reply expectation. This is conversational context, not speaker identification.

## Implementation and remaining work

Muse's spoken-response instructions now welcome greetings and small talk without adding a question after every answer. Jev routes intentional social conversation and classifies completed Muse replies. That second decision overlaps TTS where possible and has a 1.5-second bound; low confidence uses the configured idle window. The authenticated response-complete event carries the optional expectation to the Pi.

After playback drains, the Pi uses 3 seconds after an ordinary answer, at least 8 seconds after a confirmation, at least 10 seconds after an open answer, or at most 0.5 seconds after closing. These are trial defaults. Ignored speech restores the previous deadline instead of closing the session or extending it. A 40-second cap bounds a continuous VAD candidate. Recognized turns reach Jev with the previous reply expectation. Empty transcripts use a local recovery rule: after an invitation and at least 300 ms of VAD speech, ask once for a repeat; otherwise stay quiet. Per-turn audio duration, VAD speech time, RMS, peak, and clipping enter the trace; raw PCM does not.

Twelve synthetic single-turn cases and one four-turn sequence were added to the private local replay set. Latest replay matched all 54 labeled routes and all eight reply-expectation decisions. Offline lifecycle tests cover deadline restoration, dynamic windows, ignored turns, and bounded speech. This does not verify audio quality or room behavior.

### 1. Establish conversation examples and a baseline

Extend the existing private replay dataset with greetings, small talk, invitations without punctuation, rhetorical questions, confirmations, contextual follow-ups, endings, and probable playback echo. Include multi-turn sequences, not only independently supplied context pairs. Reserve a separate evaluation set before changing prompts.

Measure unwanted spoken replies, rejected intentional turns, premature session endings, routing latency, fallback use, and model requests per accepted exchange. Keep real transcripts, analyst labels, and synthetic examples identifiable. Small replay scores do not establish room performance.

### 2. Improve Muse’s spoken response instructions

Update `CUBE_VOICE_RESPONSE_INSTRUCTION` in `service/muse_provider.py` to recognize greetings and social exchanges, give concise first answers, and ask one useful question when more information is needed. Avoid repetitive “How can I help?” endings and questions added only to prolong conversation. Respect explicit endings. Keep voice instructions consistent across turns without requesting spoken metadata.

Replay “Hey Watson, how are you?” through Muse and inspect the resulting conversation, including a social follow-up, an actual task, and silence. Natural wording may vary; evaluate behavior rather than exact strings.

### 3. Classify the reply’s expectation

Start a bounded Jev classification as soon as Muse has produced its reply text. Supply the last accepted user turn and actual reply, and return one of the four expectation values. Use the existing Decisions clients and Luna fallback. This adds one decision request per spoken reply and must be measured for cost and latency.

Overlap classification with remaining playback when possible. Do not delay the first spoken audio. After playback drains, wait no more than 1.5 seconds for Jev or Luna; missing or low-confidence classification uses the configured base window. A late result must not reopen an expired conversation.

Keep the Jev reply check separate from Muse text generation. The current integration requests text only; revisit a combined metadata path only if the pinned SDK supports it and replay proves it preserves the agreed Jev decision.

Pass expectation to incoming Jev routing with the previous exchange. Do not pre-filter short acknowledgments locally; Jev should decide whether each captured turn answers Cube or starts a new request.

### 4. Carry expectation to the Pi and apply bounded listening

Extend the authenticated response lifecycle event with optional, validated expectation metadata. Cover `contracts/models.py`, schema/fixtures, gateway forwarding, device event parsing, and both Muse and internal-control reply paths. Missing metadata preserves the current three-second behavior. The strict v1 event contract rejects unknown fields and event types, so stage matching API and Pi releases together before enabling Muse V2. Ignore stale metadata using the active response/session identity.

Extend the existing session controller rather than adding a second conversation controller. Start the listening deadline only after speaker playback drains. Initial simulation settings: a short continuation window around 2–3 seconds, a longer answer window around 6–8 seconds, and closure after a closing reply. Treat these as tuning candidates, not established production values.

Speech beginning before the deadline gets a bounded opportunity to finish and be classified. Background noise must not repeatedly restart the window or hold it open forever. Preserve separate limits for candidate speech duration, processing time, and overall session duration. A resumed wake word remains an explicit way to get Watson’s attention after closure.

Change ignored-input handling in `service/muse_provider.py`: currently both `ignore` and `close` emit `session.closed`. `ignore` should silently resume listening only for the original deadline’s remaining time; it must not update accepted conversation context or extend that deadline. An explicit `close` ends the conversation. Handle expired deadlines and outstanding processing deterministically.

Update replay scoring: ignore and close will no longer be behaviorally interchangeable. Detect premature closure separately from unwanted replies and missed requests.

### 5. Simulate the complete conversation lifecycle

Extend existing runtime and provider simulations to cover reply expectation propagation, playback-drain timing, silence, pauses, speech starting just before expiration, ignored noise, provider timeouts, malformed decisions, late metadata, reconnects, and old event consumers. Verify that speech onset reserves time without indefinitely extending the session.

Run text replay for routing and controlled audio through VAD/STT/routing for transcription and timing failures. Use available recordings only where authorized; synthetic audio and mixed noise are stress cases, not substitutes for the Cube microphone. Keep device actions simulated.

Advance when new greeting and question-answer sequences work, held-out legitimate follow-ups show no new misses, silence closes without speech, noise never extends the original deadline, and failure paths remain bounded. Compare latency and cost with the baseline before enabling the new policy. Report counts and failures, not only aggregate accuracy.

### 6. Add interruption support after an acoustic trial

The current Muse stack suppresses speech detection while processing and playing audio. Natural interruption needs playback-reference echo cancellation or another validated way to distinguish the person from Watson’s speaker. Measure available Pi audio support before selecting an implementation.

First validate an explicit wake-word interruption. On a confirmed interruption, cancel the active response, flush queued playback, prevent late chunks from the cancelled response from playing, and preserve enough new input to understand the correction. Then evaluate ordinary speech interruptions. Use “Wait, I meant Friday” as a full-path acceptance case.

Keep existing playback suppression until physical testing shows interruptions do not cause Watson to hear itself. Computer simulation alone cannot approve that change.

## Release and documentation

Ship steps 1–5 as the first increment; keep interruption support as the next increment requiring hardware. Use existing voice-stack rollback and coordinate API/Pi contract changes. Preserve a simple fallback to the fixed three-second policy if expectation metadata is unavailable.

Update `docs/v2-muse-elevenlabs-plan.md`, `docs/voice-simulation.md`, and the current system map when implementation changes land. Keep this proposal distinct from the diagram of the running source.

Remaining work: add a held-out replay set; run controlled audio through VAD/STT; then try the new windows on the Pi. Malformed Jev choices are rejected, and the Luna fallback path has an offline test using a simulated Jev timeout. Physical testing remains required for room echo, microphone quality, and whether pauses feel comfortable. Add interruption support only after playback echo can be measured and controlled.
