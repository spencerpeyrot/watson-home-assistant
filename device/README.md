# Raspberry Pi voice device

The `device` package runs the local microphone, wake-word, session, and speaker path. It contains ALSA capture/playback, the bundled Watson ONNX wake-word adapter, VAD, audio buffering, gateway transport, timers, and local Home Assistant actions.

The device process runs with `python -m device.voice_client`. Configure it with `deploy/voice-client.env.example` and the systemd service template. The wake-word classifier ships in `assets/wakeword/hey_watson.onnx`; pinned openWakeWord feature models are prepared with `scripts/prepare_openwakeword_assets.py`.

The Pi uses 24 kHz mono PCM through ALSA. Device names, HAT drivers, mixer levels, and wake threshold require local hardware calibration. See the [hardware runbook](../docs/hardware-runbook.md) and [wake-word guide](../docs/openwakeword-watson-model.md).
