"""Cited web lookup through the OpenAI Responses API web-search tool."""

from __future__ import annotations

import asyncio
import ipaddress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from openai import AsyncOpenAI
from pydantic import ValidationError

from contracts.models import SourceRef, UsageMeasurement


class SearchUnavailable(RuntimeError):
    """Sanitized search timeout, provider error, or unusable answer."""


@dataclass(frozen=True)
class SearchAnswer:
    text: str
    sources: tuple[SourceRef, ...]
    usage: UsageMeasurement | None


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class OpenAIWebSearch:
    """Force a Responses web-search call and preserve cited source metadata."""

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str = "gpt-6-luna",
        timeout_seconds: float = 45.0,
        max_sources: int = 20,
        max_output_tokens: int | None = None,
    ) -> None:
        if not model.strip() or timeout_seconds <= 0 or max_sources <= 0:
            raise ValueError("search configuration values must be positive and nonblank")
        if max_output_tokens is not None and not 1 <= max_output_tokens <= 128_000:
            raise ValueError("search output-token ceiling must be between 1 and 128000")
        self._client = client
        self._model = model
        self._timeout = timeout_seconds
        self._max_sources = max_sources
        self._max_output_tokens = max_output_tokens

    async def search(self, query: str) -> SearchAnswer:
        if not query.strip() or len(query) > 4000:
            raise ValueError("search query must contain 1–4000 characters")
        prompt = (
            "Search the web for this request. Answer with concise, current facts. "
            "Use citations for factual claims and do not invent sources.\n\n"
            f"Request: {query.strip()}"
        )
        return await self._request_answer(prompt)

    async def fetch(self, url: str, query: str) -> SearchAnswer:
        """Open one public URL and require matching open-page evidence."""
        target = url.strip()
        question = query.strip()
        if not target or len(target) > 2048:
            raise ValueError("fetch URL must contain 1–2048 characters")
        if not question or len(question) > 4000:
            raise ValueError("fetch query must contain 1–4000 characters")
        parsed = urlsplit(target)
        try:
            hostname = parsed.hostname
            port = parsed.port
        except ValueError:
            raise ValueError("fetch URL must be a valid public HTTP(S) URL") from None
        normalized_hostname = hostname.lower().rstrip(".") if hostname else ""
        if (
            parsed.scheme not in {"http", "https"}
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or normalized_hostname == "localhost"
            or normalized_hostname.endswith(
                (".localhost", ".local", ".internal", ".test", ".invalid", ".example")
            )
        ):
            raise ValueError("fetch URL must be a valid public HTTP(S) URL")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError("fetch URL must be a valid public HTTP(S) URL")
        if address is None and "." not in normalized_hostname:
            raise ValueError("fetch URL must be a valid public HTTP(S) URL")
        prompt = (
            "Open this exact web page using the web-search open-page action. Do not "
            "search for or substitute a different page. Answer only from that "
            "page, cite its URL, and say if it cannot be opened or does not contain "
            "the requested information.\n\n"
            f"Page URL: {target}\n"
            f"Request: {question}"
        )
        return await self._request_answer(prompt, required_open_page=(target, port))

    async def _request_answer(
        self,
        prompt: str,
        *,
        required_open_page: tuple[str, int | None] | None = None,
    ) -> SearchAnswer:
        try:
            async with asyncio.timeout(self._timeout):
                request = {
                    "model": self._model,
                    "input": prompt,
                    "tools": [
                        {"type": "web_search", "search_context_size": "medium"}
                    ],
                    "tool_choice": "required",
                    "include": ["web_search_call.action.sources"],
                }
                if self._max_output_tokens is not None:
                    request["max_output_tokens"] = self._max_output_tokens
                response = await self._client.responses.create(
                    **request
                )
        except TimeoutError:
            raise SearchUnavailable("web search timed out") from None
        except Exception as error:
            raise SearchUnavailable(
                f"web search failed ({type(error).__name__})"
            ) from None

        output = _field(response, "output", ())
        if required_open_page is not None and not self._opened_page(
            output, *required_open_page
        ):
            raise SearchUnavailable("web fetch did not open the requested URL")
        text = _field(response, "output_text", "")
        if not isinstance(text, str) or not text.strip():
            text = self._extract_message_text(output)
        if not text.strip():
            raise SearchUnavailable("web search returned no answer text")
        sources = self._extract_sources(output)
        if not sources:
            raise SearchUnavailable("web search returned no usable citations")
        return SearchAnswer(
            text=text.strip(),
            sources=sources,
            usage=self._usage(_field(response, "usage")),
        )

    @classmethod
    def _opened_page(
        cls, output: Any, requested: str, requested_port: int | None
    ) -> bool:
        if not isinstance(output, (list, tuple)):
            return False
        for item in output:
            if _field(item, "type") != "web_search_call":
                continue
            action = _field(item, "action", {})
            if (
                _field(action, "type") == "open_page"
                and cls._same_page(
                    requested,
                    _field(action, "url", ""),
                    requested_port,
                )
            ):
                return True
        return False

    @staticmethod
    def _same_page(requested: str, cited: str, requested_port: int | None) -> bool:
        if not isinstance(requested, str) or not isinstance(cited, str):
            return False
        try:
            requested_url = urlsplit(requested)
            cited_url = urlsplit(cited)
        except ValueError:
            return False
        try:
            cited_port = cited_url.port
        except ValueError:
            return False
        requested_host = requested_url.hostname
        cited_host = cited_url.hostname
        if not requested_host or not cited_host:
            return False
        normalized_requested_port = requested_port
        if normalized_requested_port == (80 if requested_url.scheme == "http" else 443):
            normalized_requested_port = None
        if cited_port == (80 if cited_url.scheme == "http" else 443):
            cited_port = None
        requested_path = requested_url.path or "/"
        cited_path = cited_url.path or "/"
        return (
            cited_url.scheme in {"http", "https"}
            and requested_host.lower() == cited_host.lower()
            and normalized_requested_port == cited_port
            and requested_path.rstrip("/") == cited_path.rstrip("/")
            and requested_url.query == cited_url.query
        )

    @staticmethod
    def _extract_message_text(output: Any) -> str:
        fragments: list[str] = []
        if not isinstance(output, (list, tuple)):
            return ""
        for item in output:
            if _field(item, "type") != "message":
                continue
            content = _field(item, "content", ())
            if not isinstance(content, (list, tuple)):
                continue
            for part in content:
                if _field(part, "type") in {"output_text", "text"}:
                    value = _field(part, "text", "")
                    if isinstance(value, str):
                        fragments.append(value)
        return "".join(fragments)

    def _extract_sources(self, output: Any) -> tuple[SourceRef, ...]:
        found: dict[str, SourceRef] = {}
        if not isinstance(output, (list, tuple)):
            return ()
        retrieved_at = datetime.now(timezone.utc)

        def add_source(url: Any, title: Any = "") -> bool:
            if not isinstance(url, str) or not isinstance(title, str):
                return False
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                return False
            normalized_url = url.split("#", 1)[0]
            try:
                source = SourceRef(
                    title=title.strip() or parsed.hostname,
                    url=normalized_url,
                    retrieved_at=retrieved_at,
                )
            except ValidationError:
                return False
            key = str(source.url)
            previous = found.get(key)
            if previous is None or (
                previous.title == parsed.hostname and title.strip()
            ):
                found[key] = source
            return len(found) >= self._max_sources

        for item in output:
            if _field(item, "type") == "web_search_call":
                action = _field(item, "action", {})
                if _field(action, "type") == "open_page":
                    if add_source(_field(action, "url")):
                        return tuple(found.values())
                action_sources = _field(action, "sources", ())
                if isinstance(action_sources, (list, tuple)):
                    for source in action_sources:
                        if _field(source, "type", "url") != "url":
                            continue
                        if add_source(_field(source, "url")):
                            return tuple(found.values())
                continue
            if _field(item, "type") != "message":
                continue
            content = _field(item, "content", ())
            if not isinstance(content, (list, tuple)):
                continue
            for part in content:
                annotations = _field(part, "annotations", ())
                if not isinstance(annotations, (list, tuple)):
                    continue
                for annotation in annotations:
                    if _field(annotation, "type") != "url_citation":
                        continue
                    citation = _field(annotation, "url_citation", {})
                    if add_source(
                        _field(citation, "url", ""),
                        _field(citation, "title", ""),
                    ):
                        return tuple(found.values())
        return tuple(found.values())

    @staticmethod
    def _usage(value: Any) -> UsageMeasurement | None:
        if value is None:
            return None
        input_tokens = _field(value, "input_tokens")
        output_tokens = _field(value, "output_tokens")
        if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
            return None
        return UsageMeasurement(input_tokens=input_tokens, output_tokens=output_tokens)
