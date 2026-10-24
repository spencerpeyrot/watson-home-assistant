# Hardware runbook

## Supported starting point

- Raspberry Pi 5
- ReSpeaker 2-Mics Pi HAT V2
- ALSA capture and playback through the HAT
- WS2812 status ring is optional

The included voice service template expects a dedicated `cube` account with access to the `audio` and `gpio` groups. Review the ALSA selectors in `deploy/voice-client.env.example` for the devices exposed by the operating system.

## Bring-up

1. Install Python 3.12 or 3.13 and `uv`.
2. Install the server and device dependency groups from `pyproject.toml`.
3. Install the Pi's audio drivers and confirm the HAT appears in ALSA.
4. Set capture and playback device names in the voice-client environment file.
5. Prepare the checksummed openWakeWord feature models as described in the [Watson wake-word guide](openwakeword-watson-model.md).
6. Configure service credentials and gateway URL. Keep secrets in root-owned environment files, not in this repository.
7. Review the systemd templates under `deploy/` before enabling services.
8. Verify microphone capture, speaker playback, wake detection, and quiet-room behavior at intended volume.

## Calibration

Wake threshold, VAD timing, mixer gain, speaker volume, and microphone placement depend on the room and hardware. Start with the values in the example environment file. Check nearby and distant wake attempts, ordinary room noise, and speaker echo. Do not treat one room's threshold as universal.
