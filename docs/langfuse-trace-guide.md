# Reading Watson traces

Open a **Watson conversation** from a new voice session. Earlier traces retain
old names and separate component roots; this rollout cannot reconstruct missing
historical text. A Langfuse session groups the same voice session UUID; one trace
contains its wake-to-close conversation, including multiple questions.

```text
Watson conversation                         recognized user speech → Watson speech
├── Live voice connection                   microphone/model connection
│   ├── Connect to voice model              provider startup duration
│   ├── Forward result to voice model       facts/context sent to Live
│   └── Diagnostics                         timing and transcript fragments
│       ├── Turn speech into text             ElevenLabs recognition
│       ├── Choose how Watson handles request Jev route and acknowledgement ID
│       ├── Wait for Muse to answer            Muse response duration
│       └── Turn reply into speech             ElevenLabs TTS
├── Understand request and deliver answer   speech/request coordination
│   ├── Get current time                    question/place → verified clock result
│   ├── Submit background request           request → accepted job ID
│   ├── Deliver local/background result     facts supplied for the spoken answer
│   └── Diagnostics                         transcript boundaries and local VAD
├── Background request                      one job attempt
│   └── Execute request
│       └── Choose route and generate answer
│           ├── Model response              messages → generated text/tool calls
│           └── Run tool                    arguments → result
└── Diagnostics                             device audio queue health
```

UI labels carry a `cube.` prefix to restrict exports to our instrumentation.
Every observation includes its original `stage` code and an `explanation`.

| Stage | What to inspect |
| --- | --- |
| Watson conversation | Input shows recognized user speech, output shows recognized Watson speech. These are transcripts, not audio or proof that the entire reply was heard. |
| Connect to voice model | Long duration means connection/provider startup is slow. |
| Get current time | Check place input and returned answer. Direct clock requests need no worker job. |
| Submit background request | Compare actual request with intended question; output supplies job ID. |
| Model response | Compare prompt messages with generated answer/tool calls; inspect token usage and latency. |
| Run tool | Check tool name, arguments, returned data and error status. |
| Deliver result | Confirms context was sent, not that the model spoke it correctly. |
| First audible reply on speaker | First nonsilent PCM delivered to ALSA; physical audibility still needs user confirmation. |
| Turn speech into text | ElevenLabs recognition duration and sanitized failure type. |
| Choose how Watson handles request | Selected route and fixed acknowledgement ID. |
| Jev intent decision | Route, acknowledgement, confidence, and whether low confidence caused the turn to be ignored. |
| Audio submitted for transcription | That turn's PCM duration, VAD speech time, RMS, peak, and clipping; no audio is retained. |
| Ask for a repeat after empty transcription | One retry only when the last reply invited an answer and VAD measured at least 300 ms of speech. |
| Local speech detection | The Pi's voice detector started or ended a speech-like sound; it does not prove the words were clear. |
| Microphone muted during response | Shows when half-duplex mode stops sending mic audio while Watson is thinking or speaking. If this overlaps your follow-up, Cube never sent it for transcription. |
| Answer received / ready for speaker | Separates Muse finishing its answer from ElevenLabs finishing the spoken audio. |
| Wait for Muse to answer | Duration separates slow Muse responses from speech generation. |
| Turn reply into speech | ElevenLabs TTS duration and sanitized failure type. |
| Voice turn failed | Marks failed stage and safe exception type. |
| Wait for answer before closing | `pending=true` holds idle closure; `false` resumes the inactivity window. |
| Hey Watson detected | Wake score, threshold and recent history peak/count. Numeric history/background summaries remain in Pi journal. |
| Microphone levels captured / sent | Frame count, cumulative RMS, peak and clipping. Forwarded levels are measured after provider append succeeds. Device omits duplicate pre-roll; gateway includes it. These do not identify speech source or measure intelligibility. |
| Listen for first question | Ten seconds after the voice session becomes ready, before any accepted speech or audible answer. |
| Follow-up listening window | Three seconds normally, at least eight after confirmation, at least ten after an open invitation; starts after speaker drain. |
| Diagnostics | Transcript fragments, boundary acceptance, VAD endpoints, queue depth/drop counters. Expand when troubleshooting. |

## Text capture

`CUBE_LANGFUSE_CAPTURE_CONTENT=true` explicitly enables transcripts, requests,
answers, model messages and tool arguments/results. User requested this for
troubleshooting. Text may include personal conversation and saved context passed
to a model; it is exported to the configured Langfuse project. No audio, full
provider envelopes, environment dumps, or object representations are exported.
Credential field names, known environment credentials and common key/Bearer
patterns are redacted. Redaction does not remove all possible private information.

Payloads are bounded: strings 4,000 characters, collections 20 items, nesting 5
levels and a 16,000-character serialized preview cap. Conversation transcript
storage keeps the latest 20 role segments; fragments within a same-role segment
are joined. Live has no reliable completed-turn IDs, so segments are approximate
and can include multiple same-speaker utterances. This does not alter voice state.

Set capture flag false and restart voice/API/worker to retain metadata tracing
without text. Existing exported content remains subject to project retention.
Raw audio stays in bounded runtime RAM and is never exported by this adapter.

## Grouping implementation

A device-owned real root uses the session UUID as trace ID and a deterministic
root span ID. A dedicated OpenTelemetry provider uses a context-scoped ID
generator only while creating that root. API/bridge/job lifecycle attach a remote
parent context to the same real root; they do not set SDK `trace_context`, which
marks components as separate application roots in SDK 4.17. Job execution inherits
its lifecycle observation, with graph/model/tools nested below it. Low-level
measurements nest below a Diagnostics span.

The API turn now inherits the session trace when local audio marks a turn end.
Without that context, recognition and Jev/Muse steps can appear outside the
conversation tree. Export failure stays fail-safe and cannot block voice; startup,
observation-creation, and session-flush failures are logged with safe error types.
Abrupt process shutdown can still leave open observations. New physical requests
are needed to inspect the deployed tree and content.

References: [Langfuse distributed tracing](https://langfuse.com/docs/observability/features/trace-ids-and-distributed-tracing),
[Python SDK](https://python.reference.langfuse.com/langfuse).
