"""Owner-scoped artifact storage with immutable, atomic local writes."""

from __future__ import annotations

import asyncio
import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID


class ArtifactError(RuntimeError):
    """Base class for sanitized artifact storage errors."""


class ArtifactNotFound(ArtifactError):
    """Artifact does not exist for the requested owner."""


class ArtifactAlreadyExists(ArtifactError):
    """Artifact IDs are immutable and cannot be overwritten."""


@dataclass(frozen=True)
class ArtifactMetadata:
    byte_size: int
    sha256: str


class ArtifactStore(Protocol):
    async def put(
        self, owner_id: UUID, artifact_id: UUID, content: bytes
    ) -> ArtifactMetadata: ...

    async def get(self, owner_id: UUID, artifact_id: UUID) -> bytes: ...

    async def delete(self, owner_id: UUID, artifact_id: UUID) -> bool: ...


class LocalArtifactStore:
    """Store opaque blobs below one root, partitioned by owner UUID."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as error:
            raise ArtifactError(f"artifact root could not be created ({type(error).__name__})") from None
        if not self.root.is_dir():
            raise ArtifactError("artifact root is not a directory")

    async def put(
        self, owner_id: UUID, artifact_id: UUID, content: bytes
    ) -> ArtifactMetadata:
        raw = bytes(content)
        return await asyncio.to_thread(self._put, owner_id, artifact_id, raw)

    async def get(self, owner_id: UUID, artifact_id: UUID) -> bytes:
        return await asyncio.to_thread(self._get, owner_id, artifact_id)

    async def delete(self, owner_id: UUID, artifact_id: UUID) -> bool:
        return await asyncio.to_thread(self._delete, owner_id, artifact_id)

    def _owner_directory(self, owner_id: UUID, *, create: bool) -> Path:
        directory = self.root / owner_id.hex
        if create:
            try:
                directory.mkdir(mode=0o700, exist_ok=True)
            except OSError as error:
                raise ArtifactError(
                    f"artifact owner directory unavailable ({type(error).__name__})"
                ) from None
        if directory.is_symlink():
            raise ArtifactError("artifact owner directory is unavailable")
        if not directory.exists():
            raise ArtifactNotFound("artifact not found")
        if not directory.is_dir():
            raise ArtifactError("artifact owner directory is unavailable")
        if directory.resolve().parent != self.root:
            raise ArtifactError("artifact owner directory escaped storage root")
        return directory

    def _target(self, owner_id: UUID, artifact_id: UUID, *, create: bool) -> Path:
        return self._owner_directory(owner_id, create=create) / f"{artifact_id.hex}.blob"

    def _put(self, owner_id: UUID, artifact_id: UUID, content: bytes) -> ArtifactMetadata:
        target = self._target(owner_id, artifact_id, create=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".upload-", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, target, follow_symlinks=False)
            except FileExistsError:
                raise ArtifactAlreadyExists("artifact ID already exists") from None
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        return ArtifactMetadata(byte_size=len(content), sha256=hashlib.sha256(content).hexdigest())

    def _get(self, owner_id: UUID, artifact_id: UUID) -> bytes:
        target = self._target(owner_id, artifact_id, create=False)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(target, flags)
        except FileNotFoundError:
            raise ArtifactNotFound("artifact not found") from None
        except OSError as error:
            raise ArtifactError(f"artifact read failed ({type(error).__name__})") from None
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ArtifactError("artifact is not a regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as source:
                return source.read()
        finally:
            os.close(descriptor)

    def _delete(self, owner_id: UUID, artifact_id: UUID) -> bool:
        try:
            target = self._target(owner_id, artifact_id, create=False)
            target.unlink()
            return True
        except FileNotFoundError:
            return False
        except ArtifactNotFound:
            return False
