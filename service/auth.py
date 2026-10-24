"""Constant-time bearer authentication for one-owner, device-scoped deployments."""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass, field
from typing import Literal, Mapping
from uuid import UUID

from fastapi import HTTPException, Request, WebSocket


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    """Identity resolved by credentials; never accepted from a request body."""

    role: Literal["device", "owner", "operator", "worker"]
    owner_id: UUID | None
    device_id: str | None = None


@dataclass(frozen=True)
class BearerCredential:
    """Provisioned bearer secret and its fixed server-side principal."""

    token: str = field(repr=False)
    principal: AuthenticatedPrincipal


@dataclass(frozen=True)
class _CredentialDigest:
    digest: bytes
    principal: AuthenticatedPrincipal


class AuthenticationRejected(Exception):
    """Credential did not match; message contains no credential details."""


class CredentialAuthenticator:
    """Authenticate HTTP and WebSocket requests against hashed bearer tokens."""

    MIN_TOKEN_CHARACTERS = 32
    MAX_TOKEN_CHARACTERS = 4_096

    def __init__(self, credentials: tuple[BearerCredential, ...] | list[BearerCredential]):
        if not credentials:
            raise ValueError("at least one bearer credential is required")
        digests: list[_CredentialDigest] = []
        seen: set[bytes] = set()
        for credential in credentials:
            token = credential.token
            if not self.MIN_TOKEN_CHARACTERS <= len(token) <= self.MAX_TOKEN_CHARACTERS:
                raise ValueError("bearer tokens must contain 32–4096 characters")
            if any(character.isspace() for character in token):
                raise ValueError("bearer tokens must not contain whitespace")
            principal = credential.principal
            if principal.role in {"device", "worker"}:
                if principal.owner_id is None or not principal.device_id or not principal.device_id.strip():
                    raise ValueError("device and worker credentials require owner and device mappings")
            elif principal.role == "owner":
                if principal.owner_id is None or principal.device_id is not None:
                    raise ValueError("owner credentials require owner mapping only")
            elif principal.role == "operator":
                if principal.device_id is not None:
                    raise ValueError("operator credentials cannot map to a device")
            else:
                raise ValueError("credential role is invalid")

            digest = hashlib.sha256(token.encode("utf-8")).digest()
            if digest in seen:
                raise ValueError("bearer credentials must be distinct")
            seen.add(digest)
            digests.append(_CredentialDigest(digest=digest, principal=principal))
        self._credentials = tuple(digests)

    async def authenticate_http(self, request: Request) -> AuthenticatedPrincipal:
        """FastAPI dependency for authenticated REST routes."""

        principal = self._resolve(request.headers.get("authorization", ""))
        if principal is None:
            raise HTTPException(
                status_code=401,
                detail="Authentication required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return principal

    async def authenticate_websocket(self, websocket: WebSocket) -> AuthenticatedPrincipal:
        """Authenticator for the durable device WebSocket route."""

        principal = self._resolve(websocket.headers.get("authorization", ""))
        if principal is None:
            raise AuthenticationRejected("WebSocket authentication rejected")
        return principal

    def _resolve(self, authorization: str) -> AuthenticatedPrincipal | None:
        if len(authorization) > self.MAX_TOKEN_CHARACTERS + 7:
            return None
        scheme, separator, token = authorization.partition(" ")
        if (
            not separator
            or scheme.casefold() != "bearer"
            or not token
            or any(character.isspace() for character in token)
            or len(token) > self.MAX_TOKEN_CHARACTERS
        ):
            return None
        candidate = hashlib.sha256(token.encode("utf-8")).digest()
        matched: AuthenticatedPrincipal | None = None
        for credential in self._credentials:
            if hmac.compare_digest(candidate, credential.digest):
                matched = credential.principal
        return matched


def authenticator_from_environment(
    environ: Mapping[str, str] | None = None,
) -> CredentialAuthenticator:
    """Build fixed owner/device credentials from process env, never from files."""

    values = os.environ if environ is None else environ
    required = (
        "CUBE_OWNER_ID",
        "CUBE_DEVICE_ID",
        "CUBE_OWNER_TOKEN",
        "CUBE_DEVICE_TOKEN",
    )
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise RuntimeError("Missing required authentication settings: " + ", ".join(missing))
    try:
        owner_id = UUID(values["CUBE_OWNER_ID"])
    except ValueError:
        raise RuntimeError("CUBE_OWNER_ID must be a UUID") from None
    device_id = values["CUBE_DEVICE_ID"]
    if not device_id.strip():
        raise RuntimeError("CUBE_DEVICE_ID must not be blank")

    credentials = [
        BearerCredential(
            token=values["CUBE_OWNER_TOKEN"],
            principal=AuthenticatedPrincipal(role="owner", owner_id=owner_id),
        ),
        BearerCredential(
            token=values["CUBE_DEVICE_TOKEN"],
            principal=AuthenticatedPrincipal(
                role="device", owner_id=owner_id, device_id=device_id
            ),
        ),
    ]
    operator_token = values.get("CUBE_OPERATOR_TOKEN", "")
    if operator_token:
        credentials.append(
            BearerCredential(
                token=operator_token,
                principal=AuthenticatedPrincipal(role="operator", owner_id=None),
            )
        )
    worker_token = values.get("CUBE_WORKER_TOKEN", "")
    if worker_token:
        credentials.append(
            BearerCredential(
                token=worker_token,
                principal=AuthenticatedPrincipal(
                    role="worker", owner_id=owner_id, device_id=device_id
                ),
            )
        )
    return CredentialAuthenticator(credentials)
