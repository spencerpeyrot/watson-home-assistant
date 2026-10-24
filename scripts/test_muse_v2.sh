#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

uv_args=(
  uv run --locked --no-default-groups
  --group server --group device --group wakeword --group muse
  python -m unittest discover -s tests
)

patterns=(
  'test_muse*.py'
  'test_voice_commands.py'
  'test_voice_runtime_replay.py'
  'test_elevenlabs_stt.py'
  'test_elevenlabs_tts.py'
  'test_live_gateway.py'
  'test_device_voice_runtime.py'
  'test_voice_service.py'
  'test_device_voice_relay.py'
  'test_session_control.py'
  'test_voice_job_orchestrator.py'
  'test_server.py'
)

for pattern in "${patterns[@]}"; do
  "${uv_args[@]}" -p "$pattern" -v
done
