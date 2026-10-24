# Improving voice behavior without Cube

We can replay transcripts through the exact production route handler and simulate device lifecycle events. These check different things: model replay checks routing choices; offline runtime tests check event order, playback suppression, timeouts, and provider fallback. Neither establishes acoustic performance in a room.

## Production routing replay

The earlier architecture experiment compares several experimental prompts. This runner calls the production route or reply-expectation classifier, with Jev requests, confidence thresholds, Luna fallback, and local-action extraction. Device tools are replaced with an inert simulator. No Muse chat, speech synthesis, device transport, or ledger writes occur. Calls to Jev and, if needed, Luna are real and billable.

From repository root:

```sh
uv run --locked --env-file .env python -m evaluation.voice_runtime_replay \
  .local/voice-gate-eval/cases.json \
  .local/voice-gate-eval/runtime-results.json
```

Private cases must remain outside tracked files. Results contain case IDs, route choices, elapsed time, and simulated tool names, without transcript bodies. Most cases supply previous context independently; one four-turn case checks that accepted exchanges and reply expectations carry forward after ignored noise. Replay uses transcripts, not audio.

23 October 2026 conversational iteration: Jev classifies whether Muse's answer invites an open answer, confirmation, no answer, or closing. After playback drains, the Pi uses 3 seconds normally, at least 8 after confirmation, at least 10 after an open invitation, or at most 0.5 after closing. Luna remains fallback. Ambiguous expectations use the configured base idle window. Ignored speech preserves the existing deadline; uninterrupted VAD candidates remain capped at 40 seconds. After an invitation, an empty transcription gets one repeat prompt only if Pi VAD measured at least 300 ms of speech. Other empty turns remain silent. Per-turn PCM duration, VAD speech time, RMS, peak, and clipping are traced; no raw audio is retained.

The latest production replay scored 62/62 labeled decisions: 54/54 routes and 8/8 reply expectations, with zero false accepts, missed requests, false closures, or missed closures. One of 58 cases had no score label. Median case time was 167 ms; no device actions ran. These cases combine historical Langfuse context and synthetic scenarios. This replay checks transcript decisions, not audio quality or Muse's generated conversation style.

These small, partly synthetic sets are useful for finding regressions, not estimating real-world accuracy. Ignore and close have different outcomes: ignore preserves the session deadline; close ends the session. Latency includes production HTTP connection setup and reply expectation classification, so it is not directly comparable to the earlier experiment's reused-connection route-only timing. The four-turn case runs a transcript sequence; it does not simulate continuous audio. Empty-transcript retry is covered by provider tests, including the no-invitation and one-retry limits, but VAD threshold behavior still needs Pi audio trials.

## Offline lifecycle checks

```sh
bash scripts/test_muse_v2.sh
```

Runtime simulation revealed an ordering error introduced in the previous change: endpoint detection ran before speech classification. `WebRtcSpeechActivity` expects endpoint detection to consume its cached classification, so reversed order could classify a frame twice or consume a previous frame's decision. The runtime now classifies input first, consumes its endpoint immediately afterward, and sets the response-wait guard before checking the active wake word. A regression assertion covers this order while retaining half-duplex suppression coverage.

Next useful simulations: late metadata and reconnect edge cases, plus controlled audio through VAD/STT. Jev unavailability, malformed choices, near-deadline speech, and Luna fallback have offline coverage. Keep tuning cases separate from a held-out set once one is reserved.

The coordinated API/device canary is now staged on the Pi. Physical Cube still required to check microphone quality, speaker echo, room noise, short answers, and whether the listening window feels natural. No acoustic acceptance has run yet.
