# Plan: make Watson's conversations easy to follow

Status: first release active on the Pi canary; physical conversation checks remain. Builds on `conversational-voice-plan.md`.

## Outcome

A person should know when Watson has heard them, when an answer is still coming, and when it is their turn. Spoken language carries those handoffs. The light supports them; a chime is not required to use the product.

Example target:

> Person: “Hey Watson.”
> Watson: “Hey, what's on your mind?”
> Person: “What's the weather tomorrow?”
> Watson, if a lookup takes time: “Let me check tomorrow's forecast.”
> Watson: “Rain in the morning, clearing by lunch. Around 65 degrees.”
> Person: “And Friday?”
> Watson answers using the same context.

The weather wording is illustrative, not live information. Watson need not ask another question after every complete answer. When a question helps, make it specific: “What time are you heading out?” is more useful than repeatedly asking “Anything else?”

## What exists and what remains

- Muse already receives instructions for concise spoken answers, greetings, and useful follow-up questions.
- Jev already routes input using conversation context, with Luna fallback. A separate bounded decision classifies whether Watson's actual reply invites an answer.
- The Pi now applies trial listening windows after playback drains: 3 seconds normally, at least 8 for confirmation, at least 10 for an open answer, and at most 0.5 after closing.
- Cached greetings and selected progress acknowledgments already exist. Their timing and consistency need physical acceptance.
- Capture is half duplex: speech during response processing or playback is suppressed. Ordinary interruptions are disabled.
- Empty transcription now gets one spoken repeat only when the last reply invited an answer and the endpoint included at least 300 ms of VAD-accepted speech. The turn trace records audio length, VAD speech time, RMS, peak, and clipping. This reduces noise-triggered retries; physical Pi checks still need to validate the 300 ms rule.
- Transcript replay success does not establish a usable spoken conversation.

## Step 1: agree on observable conversation behavior

Use a small acceptance script as the product specification:

| Situation | Required behavior |
| --- | --- |
| “Hey Watson” alone | One prompt spoken promptly, then listen. |
| “Hey Watson, what's the weather?” | Preserve the question and skip the greeting. |
| “Hey Watson, how are you?” | Natural social reply, with an appropriate invitation to continue. |
| Slow request | One truthful acknowledgment when useful, followed by the answer. |
| Fast request | Speak the answer directly; avoid an extra acknowledgment. |
| Watson asks “Which city?” | Accept “Boston” as the answer using context. |
| Person says “And Friday?” | Continue the previous subject. |
| Person pauses midway through a sentence | Wait rather than reply to an unfinished thought. |
| Person stays quiet | Finish the listening window silently. |
| Person says “Thanks, that's all” | Brief closing, then leave the conversation. |
| Noise after a reply | No spoken response and no repeated deadline extension. |
| Genuine speech cannot be transcribed | Recover when evidence supports a real attempt, rather than silently losing the turn. |
| Person interrupts | In the interruption release, stop speaking and capture the correction. |

Before changes, run the same script on the Pi and record each failure stage. Count completed exchanges, repeated requests, premature endings, unwanted responses, and moments where the person cannot tell whether to speak.

Ready when: baseline exists for the actual room and microphone, with a transcript-only comparison kept separate.

## Step 2: make listening and playback share one reliable handoff

Extend the existing runtime and session controller. Model the existing phases explicitly: waiting for wake, listening, processing, speaking, listening for a follow-up, and closing. Do not create another competing controller.

The next listening window begins after the last sound has actually left the speaker, not after Muse finishes generating text or ElevenLabs finishes sending bytes. Open the microphone at that same handoff. A spoken invitation must lead directly into a working listening window.

Track session, turn, and response identity so a late acknowledgment, audio chunk, or model result cannot reopen an old turn. Cancel queued acknowledgments if the answer is already available. Preserve the current echo suppression until interruption work is validated.

Add the minimum missing evidence to each submitted audio turn: byte count, duration, RMS signal level, peak, clipping, and capture/forwarding boundaries. Pair these with transcription timing, result length, ignore reason, playback drain, microphone state, and close reason. Session-wide audio levels cannot explain a particular failed turn. Retain no raw audio by default.

Touchpoints: `device/voice_runtime.py`, `device/session_control.py`, `service/muse_provider.py`, `service/live_gateway.py`, `voice/observability.py`; update existing event contracts and parsers together only if additional lifecycle fields are needed.

Ready when: each acceptance exchange has a complete timeline, and stale events cannot play audio or change listening state.

## Step 3: make wake and waiting feel like a conversation

Finish the existing cached greeting flow. Wake alone gets a short invitation. Continued speech after the wake phrase takes priority, with enough retained audio to preserve the beginning of the request. Test this distinction acoustically, including a brief pause after “Hey Watson.”

Use the existing Jev routing request to choose an allowed acknowledgment ID, with Luna fallback. The runtime decides whether it is still useful to play. Avoid adding another model call just to decide when to speak.

Acknowledgments describe the actual next step: “Let me check the forecast,” “Let me think that through,” or a brief clarification. They never claim an action succeeded. Play at most one routine acknowledgment per request. If waiting becomes unusually long, issue a bounded status or failure message tied to actual progress; do not loop filler phrases.

Remove the routine chime from the default conversation path after spoken wake handling passes acceptance. Keep the light as a secondary cue.

Ready when: wake-only, combined wake-plus-question, quick answers, and slow answers each produce one coherent exchange without overlapping or redundant speech.

## Step 4: fix follow-up capture and failed recognition

Use the new per-turn measurements to reproduce the empty-transcript failure. Confirm whether the submitted buffer contained the intended duration and signal before changing providers or mic gain.

Evaluate starting transcription while the person speaks, using the existing ElevenLabs integration. Partial transcripts can provide earlier evidence of words and reduce work after the person stops. Compare this against the current upload-after-endpoint path using identical controlled audio, especially short answers such as “yes” and “Friday.” A streaming rewrite ships only if measurements justify it.

Separate two outcomes:

- Noise or silence: ignore without speaking or restarting the original deadline.
- Evidence of a real spoken attempt, followed by empty or failed recognition: allow one bounded recovery, such as reprocessing the retained turn where appropriate, or “I missed that. Could you say it again?” Then open a fresh answer window.

Use multiple signals to qualify recovery, such as sustained speech evidence, partial words, the preceding invitation, and measured audio. Duration or volume alone cannot prove intent. Prevent repeated retries on background noise. Keep the last accepted conversation context through recovery.

Ready when: controlled short and contextual follow-ups reach routing; recognition failures recover without creating unwanted replies to noise. Physical trials still decide whether the mic captured the person clearly.

## Step 5: give answers natural endings and comfortable pauses

Have Muse lead with a short useful answer, then expand when asked. Tune spoken output by listening to actual audio, including pacing and how sentences end. Do not rely on punctuation alone to signal a handoff.

Ask a specific question when the next answer depends on the person. Use a broad invitation occasionally when appropriate, including social openings. Complete answers may end naturally with no question. Clear goodbyes end without reopening the exchange.

Keep Jev's reply-expectation check grounded in what Watson actually said, with Luna fallback. Run it alongside speech generation where possible. The Pi owns immediate playback and microphone timing; a delayed model result must not hold the microphone closed or reopen an expired session.

Trial windows now use 3 seconds after a complete answer, at least 8 after confirmation, at least 10 after an open question, and at most 0.5 after closing. These still need room testing. Speech that begins before the deadline reserves bounded time to finish. Silence closes quietly; noise cannot keep the session alive indefinitely.

Ready when: people can answer an invitation at a comfortable pace, contextual follow-ups work, and silent endings remain quiet. Compare these invitation-based windows against the prior 3/5/7-second policy.

## Step 6: stop cutting people off while they formulate a thought

Evaluate streaming transcription plus local speech activity for end-of-turn decisions. Distinguish a pause inside “Can you… check Friday's calendar?” from the end of “What's on Friday?” Keep a bounded fallback when partial text is unavailable or uncertain.

Use local timing and available transcript evidence first. Benchmark a model-based endpoint decision only if those miss real pauses; do not add a network decision to every silence by default. Measure both premature replies and the delay after a completed sentence.

Ready when: ordinary hesitation no longer triggers an early answer, while finished short requests receive prompt replies.

## Step 7: let people interrupt safely

Measure speaker echo with the actual Pi mic and speaker. Evaluate echo cancellation, which uses the speaker's outgoing audio to remove its contribution from the microphone signal. Select an implementation only after checking hardware support, CPU cost, and room behavior.

First support a confirmed wake-word interruption during playback. Then evaluate ordinary corrections such as “Wait, I meant Friday.” On interruption: stop playback, flush queued audio, cancel response generation, reject late chunks from that response, retain the start of the correction, and continue in the same conversation.

This needs cancellation support across Pi playback, gateway events, Muse subscription/generation where supported, and ElevenLabs speech. If remote generation cannot be cancelled, suppress its remaining output locally and track that limitation.

Ready when: intentional interruptions work during the beginning, middle, and end of a reply, while Watson's own speaker never causes a new conversation in the acceptance trials. General interruption remains gated if echo trials fail.

## Step 8: release against the whole experience

For each increment, run deterministic lifecycle checks, held-out transcript replay, controlled audio scenarios, then the Pi acceptance script. Include long answers, network delays, provider failures, late events, silence, nearby speech, television audio, distance, and reconnects. Synthetic audio does not replace room trials.

Measure wake-to-first-spoken-prompt, person-finished-to-first-response, acknowledgment-to-answer gap, playback-ended-to-mic-open gap, follow-up completion rate, repeated requests, premature cutoffs, unwanted responses, interruption success, and model cost per completed exchange. Report counts and failure cases, not just a single accuracy score.

Start with no lost intended follow-ups across the fixed acceptance script, no silence/noise responses in its negative cases, and no stale playback. Set broader numerical latency and reliability targets from the measured baseline. A small passing script is a release gate, not evidence of production-wide accuracy.

Ship coordinated API/Pi canary releases with the existing rollback. Update the current system map only when behavior actually changes. Update `conversational-voice-plan.md`, `voice-simulation.md`, and the trace guide alongside implementation so implemented behavior and pending acoustic work remain distinguishable.

## Build order

1. Baseline and acceptance script.
2. Correct playback/listening handoff and per-turn evidence.
3. Consistent spoken wake and useful waiting acknowledgments.
4. Reliable follow-up recognition and bounded recovery.
5. Shorter answers, useful endings, and measured listening windows.
6. Better pause handling.
7. Echo control and interruption.
8. Whole-experience acceptance and gradual rollout throughout.

Steps 2–5 form the first usable conversational release. They improve the normal exchange while retaining half duplex. Step 7 removes the remaining requirement to wait for Watson to finish speaking. Prompt changes and longer timers alone cannot provide that ability.
