# Watson Home Assistant

Watson Home Assistant is an experimental voice assistant built around a Raspberry Pi. The Pi handles wake-word detection, microphone capture, and speaker playback. Python services handle conversation routing, speech providers, home controls, timers, and longer background jobs.

The repository includes the custom `hey_watson` openWakeWord ONNX model at `assets/wakeword/hey_watson.onnx`. The two pinned openWakeWord feature models are downloaded and checksum-verified during setup.

## What it includes

- Wake-word listening and conversational follow-up windows
- Speech transcription, routing, and spoken replies
- Allowlisted Home Assistant actions, timers, and reminders
- Durable research jobs, PostgreSQL storage, and a small status dashboard
- Langfuse observability and replay/evaluation tools

The system is a working prototype for a single device and household. It is not yet a plug-and-play image; cloud providers, database, and hardware need configuration.

## Start here

- [Current system map](docs/current-system-map.md)
- [Hardware runbook](docs/hardware-runbook.md)
- [Conversation experience plan](docs/conversation-experience-plan.md)
- [Voice simulation guide](docs/voice-simulation.md)

Use Python 3.12 or 3.13 with [uv](https://docs.astral.sh/uv/). Install the server and wake-word dependencies, then prepare the pinned ONNX feature models:

```sh
uv sync --group server --group wakeword
uv run --group wakeword python scripts/prepare_openwakeword_assets.py /var/lib/cube-assistant-voice/openwakeword
```

Set `CUBE_WAKEWORD_MODEL` to `assets/wakeword/hey_watson.onnx` and `CUBE_WAKEWORD_ASSET_DIR` to the feature-model directory. Hardware-specific configuration lives in `pyproject.toml` and `deploy/voice-client.env.example`.

Keep credentials in local environment files. Never commit them. The checked-in environment examples use placeholders. Cached phrase audio is generated separately and omitted; the service falls back to live TTS when those files are absent.

## License

License choice is pending. The project is not yet licensed for reuse.
