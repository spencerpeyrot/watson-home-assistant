"""Fetch pinned openWakeWord ONNX feature assets for offline inference."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen


ASSETS = {
    "melspectrogram.onnx": (
        "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
        "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    ),
    "embedding_model.onnx": (
        "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
        "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
    ),
}


def prepare_assets(target: Path) -> tuple[Path, ...]:
    """Download only required ONNX feature assets and verify pinned digests."""

    if target.is_symlink():
        raise ValueError("asset directory must not be a symlink")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    saved: list[Path] = []
    for name, (url, expected_digest) in ASSETS.items():
        destination = target / name
        if destination.is_symlink():
            raise ValueError(f"asset path must not be a symlink: {name}")
        if destination.exists():
            if _sha256(destination) != expected_digest:
                raise ValueError(f"existing asset checksum mismatch: {name}")
            saved.append(destination)
            continue

        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=target)
        temporary_path = Path(temporary_name)
        try:
            digest = hashlib.sha256()
            with os.fdopen(descriptor, "wb") as output, urlopen(url, timeout=30) as response:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != expected_digest:
                raise ValueError(f"downloaded asset checksum mismatch: {name}")
            os.chmod(temporary_path, 0o644)
            os.replace(temporary_path, destination)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        saved.append(destination)
    return tuple(saved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        type=Path,
        help="directory for the ONNX feature assets",
    )
    args = parser.parse_args()
    for path in prepare_assets(args.target.expanduser()):
        print(path)


if __name__ == "__main__":
    main()
