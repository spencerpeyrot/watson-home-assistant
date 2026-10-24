"""Atomically point a deployment checkout at one immutable release directory."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from uuid import uuid4


_RELEASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def switch_release(root: str | Path, release_id: str) -> str | None:
    """Atomically update ``root/current``; return prior release ID, if any.

    ``root`` and ``root/releases`` must already exist. The helper never creates
    release content, removes old releases, runs migrations, or restarts services.
    """

    if not _RELEASE_ID.fullmatch(release_id) or release_id in {".", ".."}:
        raise ValueError("release ID must be a simple directory name")
    deployment_root = Path(root).resolve(strict=True)
    releases_root = deployment_root / "releases"
    if releases_root.is_symlink() or not releases_root.is_dir():
        raise ValueError("releases directory is missing or invalid")
    releases_root = releases_root.resolve(strict=True)
    release_path = deployment_root / "releases" / release_id
    if release_path.is_symlink() or not release_path.is_dir():
        raise ValueError("target release directory is missing or invalid")
    resolved_release = release_path.resolve(strict=True)
    if resolved_release.parent != releases_root:
        raise ValueError("target release escaped the releases directory")

    current = deployment_root / "current"
    previous_release: str | None = None
    if current.is_symlink():
        try:
            resolved_current = current.resolve(strict=True)
        except OSError:
            raise ValueError("current release symlink is broken") from None
        if resolved_current.parent != releases_root or not resolved_current.is_dir():
            raise ValueError("current release symlink points outside releases")
        previous_release = resolved_current.name
        if resolved_current == resolved_release:
            return previous_release
    elif current.exists():
        raise ValueError("current path exists and is not a symlink")

    temporary = deployment_root / f".current-{uuid4().hex}.tmp"
    try:
        os.symlink(Path("releases") / release_id, temporary, target_is_directory=True)
        os.replace(temporary, current)
    finally:
        temporary.unlink(missing_ok=True)
    return previous_release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_id", help="existing immutable directory under releases/")
    parser.add_argument(
        "--root",
        default="/opt/cube-assistant",
        help="deployment root containing releases/ and current (default: /opt/cube-assistant)",
    )
    arguments = parser.parse_args()
    try:
        previous = switch_release(arguments.root, arguments.release_id)
    except (OSError, ValueError) as error:
        print(f"Release switch failed ({type(error).__name__}): {error}", file=sys.stderr)
        return 2
    print(f"active release: {arguments.release_id}")
    print(f"previous release: {previous or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
