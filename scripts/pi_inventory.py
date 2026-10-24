"""Read-only Pi hardware and ALSA inventory; safe to run over SSH."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path


def show_file(label: str, path: str) -> None:
    print(f"\n--- {label} ---")
    try:
        value = Path(path).read_text(encoding="utf-8", errors="replace")
        print(value.replace("\x00", "").strip() or "(empty)")
    except OSError as error:
        print(f"unavailable: {error.__class__.__name__}")


def show_command(label: str, args: list[str]) -> None:
    print(f"\n--- {label} ---")
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"unavailable: {error.__class__.__name__}")
        return
    output = (result.stdout + result.stderr).strip()
    print(output or f"(no output; exit {result.returncode})")


def main() -> None:
    print(f"Python: {platform.python_version()}")
    show_file("Pi model", "/proc/device-tree/model")
    show_file("OS release", "/etc/os-release")
    show_command("Kernel", ["uname", "-a"])
    for command in (
        ["arecord", "-l"],
        ["aplay", "-l"],
        ["arecord", "-L"],
        ["aplay", "-L"],
        ["amixer", "controls"],
        ["amixer", "scontrols"],
        ["amixer", "contents"],
        ["pinctrl", "get"],
        ["vcgencmd", "measure_temp"],
        ["vcgencmd", "get_throttled"],
    ):
        show_command(" ".join(command), list(command))


if __name__ == "__main__":
    main()
