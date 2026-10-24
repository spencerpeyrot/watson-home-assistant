"""Extract only allowlisted transport details from ElevenLabs failures."""

from __future__ import annotations

import re
from typing import Any


def safe_provider_value(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    return value[:128] if re.fullmatch(r"[A-Za-z0-9._:/+-]{1,128}", value) else None


def transport_diagnostics(error: BaseException) -> dict[str, Any]:
    fields: dict[str, Any] = {"cause_type": type(error).__name__}
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", getattr(error, "status_code", None))
    if isinstance(status, int):
        fields["http_status"] = status
        fields["error_category"] = (
            "rate_limit" if status == 429 else "authentication" if status in {401, 403}
            else "provider_unavailable" if status >= 500 else "provider_rejected"
        )
        fields["retryable"] = status == 429 or status >= 500
    received = getattr(error, "rcvd", None)
    close_code = getattr(received, "code", None)
    if isinstance(close_code, int):
        fields["websocket_close_code"] = close_code
    headers = getattr(response, "headers", None)
    if headers is not None:
        for key in ("request-id", "x-request-id", "xi-request-id"):
            value = headers.get(key)
            request_id = safe_provider_value(value)
            if request_id is not None:
                fields["provider_request_id"] = request_id
                break
    return fields
