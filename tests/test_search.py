from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any

from service.search import OpenAIWebSearch, SearchUnavailable


class FakeResponses:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.responses = self

    async def create(self, **values: Any) -> Any:
        self.calls.append(values)
        if self.error:
            raise self.error
        return self.response


def response_with_citation() -> SimpleNamespace:
    citation = SimpleNamespace(
        type="url_citation",
        url_citation=SimpleNamespace(
            title="Example source",
            url="https://example.com/facts#section",
        ),
    )
    output_text = SimpleNamespace(
        type="output_text",
        text="Current answer [1].",
        annotations=[citation],
    )
    message = SimpleNamespace(type="message", content=[output_text])
    return SimpleNamespace(
        output_text="Current answer [1].",
        output=[SimpleNamespace(type="web_search_call"), message],
        usage=SimpleNamespace(input_tokens=21, output_tokens=8),
    )


class SearchAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_forces_responses_web_search_and_returns_cited_contract(self) -> None:
        fake = FakeResponses(response_with_citation())
        adapter = OpenAIWebSearch(fake, model="gpt-6-luna")  # type: ignore[arg-type]

        answer = await adapter.search("What changed this week?")

        self.assertEqual(answer.text, "Current answer [1].")
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(str(answer.sources[0].url), "https://example.com/facts")
        self.assertEqual(answer.sources[0].title, "Example source")
        self.assertEqual(answer.usage.input_tokens, 21)
        call = fake.calls[0]
        self.assertEqual(call["tools"][0]["type"], "web_search")
        self.assertEqual(call["tool_choice"], "required")
        self.assertEqual(call["include"], ["web_search_call.action.sources"])
        self.assertNotIn("max_output_tokens", call)
        self.assertIn("What changed this week?", call["input"])

    async def test_optional_output_ceiling_is_sent_to_responses_api(self) -> None:
        fake = FakeResponses(response_with_citation())
        await OpenAIWebSearch(fake, max_output_tokens=384).search("Need one fact")
        self.assertEqual(fake.calls[0]["max_output_tokens"], 384)

        with self.assertRaises(ValueError):
            OpenAIWebSearch(fake, max_output_tokens=0)  # type: ignore[arg-type]

    async def test_search_action_sources_work_without_inline_titles(self) -> None:
        response = SimpleNamespace(
            output_text="Answer from search.",
            output=[
                SimpleNamespace(
                    type="web_search_call",
                    action=SimpleNamespace(
                        type="search",
                        sources=[
                            SimpleNamespace(type="url", url="https://pi.example/docs"),
                            {"type": "url", "url": "https://raspberrypi.com/news"},
                        ],
                    ),
                ),
            ],
            usage=None,
        )
        answer = await OpenAIWebSearch(
            FakeResponses(response)  # type: ignore[arg-type]
        ).search("Find sources")

        self.assertEqual(len(answer.sources), 2)
        self.assertEqual(answer.sources[0].title, "pi.example")
        self.assertEqual(str(answer.sources[0].url), "https://pi.example/docs")
        self.assertEqual(answer.sources[1].title, "raspberrypi.com")

    async def test_search_sources_and_inline_citations_are_deduplicated(self) -> None:
        response = response_with_citation()
        response.output.insert(
            0,
            SimpleNamespace(
                type="web_search_call",
                action=SimpleNamespace(
                    type="search",
                    sources=[
                        SimpleNamespace(
                            type="url", url="https://example.com/facts#other"
                        )
                    ],
                ),
            ),
        )

        answer = await OpenAIWebSearch(
            FakeResponses(response)  # type: ignore[arg-type]
        ).search("Find sources")

        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0].title, "Example source")

    async def test_fetch_requires_citation_of_requested_page(self) -> None:
        response = response_with_citation()
        response.output[0].action = SimpleNamespace(
            type="open_page",
            url="https://example.com/manual/#setup",
        )
        response.output[1].content[0].annotations[0].url_citation.url = (
            "https://example.com/manual/#setup"
        )
        fake = FakeResponses(response)  # type: ignore[arg-type]
        answer = await OpenAIWebSearch(fake).fetch(
            "https://example.com/manual", "What is the setup process?"
        )

        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(str(answer.sources[0].url), "https://example.com/manual/")
        self.assertIn("Open this exact web page", fake.calls[0]["input"])
        self.assertIn("Page URL: https://example.com/manual", fake.calls[0]["input"])

    async def test_fetch_extracts_open_page_action_as_source(self) -> None:
        response = SimpleNamespace(
            output_text="Page answer.",
            output=[
                SimpleNamespace(
                    type="web_search_call",
                    action=SimpleNamespace(
                        type="open_page",
                    url="https://docs.example.com/guide/",
                    ),
                )
            ],
            usage=None,
        )
        answer = await OpenAIWebSearch(
            FakeResponses(response)  # type: ignore[arg-type]
        ).fetch("https://docs.example.com/guide", "Summarize the guide")

        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0].title, "docs.example.com")
        self.assertEqual(str(answer.sources[0].url), "https://docs.example.com/guide/")

    async def test_fetch_rejects_wrong_source_and_non_public_targets(self) -> None:
        response = response_with_citation()
        response.output[0].action = SimpleNamespace(
            type="open_page", url="https://example.com/wrong"
        )
        fake = FakeResponses(response)  # type: ignore[arg-type]
        adapter = OpenAIWebSearch(fake)
        with self.assertRaisesRegex(
            SearchUnavailable, "did not open the requested URL"
        ):
            await adapter.fetch("https://requested.example.net/page", "Find this detail")
        self.assertEqual(len(fake.calls), 1)

        for url in (
            "file:///etc/passwd",
            "http://127.0.0.1/admin",
            "http://router.local/",
            "http://localhost./",
            "http://printer/",
            "http://example.test/",
            "https://user:pass@example.com/",
        ):
            with self.assertRaisesRegex(ValueError, "public HTTP"):
                await adapter.fetch(url, "Read this")
        self.assertEqual(len(fake.calls), 1)

    async def test_missing_citations_and_provider_details_are_sanitized(self) -> None:
        no_sources = SimpleNamespace(output_text="An uncited answer.", output=[])
        adapter = OpenAIWebSearch(FakeResponses(no_sources))  # type: ignore[arg-type]
        with self.assertRaisesRegex(SearchUnavailable, "no usable citations"):
            await adapter.search("Need a source")

        failed = OpenAIWebSearch(
            FakeResponses(error=RuntimeError("private response body"))  # type: ignore[arg-type]
        )
        with self.assertRaises(SearchUnavailable) as context:
            await failed.search("Query")
        self.assertNotIn("private response body", str(context.exception))

    async def test_timeout_and_bad_queries_fail_without_fallback(self) -> None:
        async def delayed(**_values: Any) -> Any:
            await asyncio.sleep(0.1)

        fake = SimpleNamespace(responses=SimpleNamespace(create=delayed))
        adapter = OpenAIWebSearch(fake, timeout_seconds=0.001)  # type: ignore[arg-type]
        with self.assertRaisesRegex(SearchUnavailable, "timed out"):
            await adapter.search("slow query")
        with self.assertRaises(ValueError):
            await adapter.search("  ")


if __name__ == "__main__":
    unittest.main()
