"""Read-only OpenAI key/model-catalog check. Makes no generation request."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EXPECTED_MODELS = ("gpt-live-1", "gpt-6-luna", "gpt-6.1-sol", "gpt-6-sol")


def load_key() -> str | None:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    env_file = Path(__file__).resolve().parents[1] / ".env"
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        name, separator, value = line.partition("=")
        if separator and name.strip() == "OPENAI_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value or None
    return None


def main() -> int:
    key = load_key()
    if not key:
        print("OPENAI_API_KEY missing; key value was not read from environment or project .env.")
        return 2

    request = Request(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {key}"},
        method="GET",
    )
    del key
    try:
        with urlopen(request, timeout=20) as response:
            status = response.status
            catalog = json.load(response)
    except HTTPError as error:
        print(f"OpenAI returned HTTP {error.code}; response body suppressed.")
        return 1
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        print(f"OpenAI catalog request failed: {error.__class__.__name__}.")
        return 1

    visible = {item.get("id") for item in catalog.get("data", [])}
    print(f"OpenAI API key accepted; GET /v1/models returned HTTP {status}.")
    for model in EXPECTED_MODELS:
        print(f"{model}: {'visible' if model in visible else 'not listed'}")
    print("Catalog visibility does not prove Live-session or Responses inference access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
