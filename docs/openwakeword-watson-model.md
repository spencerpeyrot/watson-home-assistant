# Watson wake word

The repository ships a custom openWakeWord ONNX classifier for “Hey Watson” at `assets/wakeword/hey_watson.onnx`. It was trained from generated speech examples. Training audio and private evaluation logs are not included.

## Prepare feature models

Install the optional wake-word dependencies and download the two pinned feature models into the device state directory:

```sh
uv sync --group wakeword
uv run --group wakeword python scripts/prepare_openwakeword_assets.py /var/lib/cube-assistant-voice/openwakeword
```

The script checks each downloaded file against its SHA-256 before saving it. The application does not download models at startup.

## Configure the device

Set these values in the voice-client environment file:

```ini
CUBE_WAKEWORD_MODEL=assets/wakeword/hey_watson.onnx
CUBE_WAKEWORD_ASSET_DIR=/var/lib/cube-assistant-voice/openwakeword
CUBE_WAKEWORD_THRESHOLD=0.5
CUBE_WAKEWORD_CONFIRMATION_WINDOWS=3
```

The model path is relative to the service working directory. The included systemd unit uses the release directory as its working directory. Tune the threshold and confirmation windows with the intended microphone, speaker volume, and room; defaults are not calibrated for every space.

See the [current system map](current-system-map.md) for the voice path and the [hardware runbook](hardware-runbook.md) for device setup.
