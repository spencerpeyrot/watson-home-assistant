from __future__ import annotations

import unittest
from uuid import UUID

from langgraph.checkpoint.memory import InMemorySaver

from service.job_graph import TaskGraph
from service.worker import WorkerSettings, build_task_graph


class FakeStore:
    async def list_memories(self, *, owner_id):
        return ()


class FakeSearchClient:
    pass


class FakeChatModel:
    def __init__(self, model: str):
        self.model_name = model
        self.model = model


def worker_environment(**overrides: str) -> dict[str, str]:
    values = {
        "DATABASE_URL": "postgresql://test.invalid/cube",
        "OPENAI_API_KEY": "test-openai-key-never-logged",
        "CUBE_ARTIFACT_DIR": "/tmp/cube-test-artifacts",
    }
    values.update(overrides)
    return values


class WorkerRuntimeTests(unittest.TestCase):
    def test_worker_settings_validate_and_hide_secrets(self):
        settings = WorkerSettings.from_environment(
            worker_environment(
                CUBE_WORKER_ID="00000000-0000-4000-8000-000000000901",
                OPENAI_LUNA_MODEL="gpt-6-luna",
                OPENAI_SOL_MODEL="gpt-6.1-sol",
            )
        )
        self.assertEqual(settings.worker_id, UUID("00000000-0000-4000-8000-000000000901"))
        self.assertEqual(settings.artifact_root.as_posix(), "/tmp/cube-test-artifacts")
        self.assertIsNone(settings.luna_max_output_tokens)
        self.assertIsNone(settings.sol_max_output_tokens)
        self.assertNotIn("test-openai-key-never-logged", repr(settings))
        self.assertNotIn("test.invalid", repr(settings))
        with self.assertRaisesRegex(RuntimeError, "CUBE_ARTIFACT_DIR"):
            WorkerSettings.from_environment(worker_environment(CUBE_ARTIFACT_DIR=" "))
        limited = WorkerSettings.from_environment(
            worker_environment(
                OPENAI_LUNA_MAX_OUTPUT_TOKENS="1024",
                OPENAI_SOL_MAX_OUTPUT_TOKENS="4096",
            )
        )
        self.assertEqual(limited.luna_max_output_tokens, 1024)
        self.assertEqual(limited.sol_max_output_tokens, 4096)
        for value in ("zero", "0", "128001"):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_SOL_MAX_OUTPUT_TOKENS"):
                WorkerSettings.from_environment(
                    worker_environment(OPENAI_SOL_MAX_OUTPUT_TOKENS=value)
                )

    def test_local_light_bridge_settings_require_full_scope_and_hide_worker_token(self):
        with self.assertRaisesRegex(RuntimeError, "must be configured together"):
            WorkerSettings.from_environment(
                worker_environment(CUBE_LOCAL_LIGHT_ENTITIES="light.desk")
            )
        settings = WorkerSettings.from_environment(
            worker_environment(
                CUBE_INTERNAL_API_URL="http://127.0.0.1:8000",
                CUBE_WORKER_TOKEN="w" * 48,
                CUBE_LOCAL_LIGHT_ENTITIES="light.desk,light.office",
            )
        )
        self.assertEqual(settings.local_light_entities, frozenset({"light.desk", "light.office"}))
        self.assertNotIn("w" * 48, repr(settings))

    def test_worker_composition_builds_luna_sol_search_graph_without_network(self):
        calls = []

        def model_factory(
            *, model, api_key, use_responses_api, output_version, reasoning, **options
        ):
            calls.append((model, api_key, use_responses_api, output_version, reasoning, options))
            return FakeChatModel(model)

        graph, search_client = build_task_graph(
            store=FakeStore(),  # type: ignore[arg-type]
            checkpointer=InMemorySaver(),
            api_key="test-key",
            search_client=FakeSearchClient(),  # type: ignore[arg-type]
            chat_model_factory=model_factory,
        )

        self.assertIsInstance(graph, TaskGraph)
        self.assertIsInstance(search_client, FakeSearchClient)
        self.assertEqual(
            [
                (model, use_responses_api, output_version, reasoning, options)
                for model, _key, use_responses_api, output_version, reasoning, options in calls
            ],
            [
                ("gpt-6-luna", True, "responses/v1", {"effort": "low"}, {}),
                ("gpt-6.1-sol", True, "responses/v1", {"effort": "low"}, {}),
            ],
        )
        self.assertTrue(all(key == "test-key" for _model, key, *_options in calls))

    def test_worker_composition_applies_optional_model_output_limits(self):
        calls = []

        def model_factory(*, model, api_key, **options):
            calls.append((model, api_key, options))
            return FakeChatModel(model)

        build_task_graph(
            store=FakeStore(),  # type: ignore[arg-type]
            checkpointer=InMemorySaver(),
            api_key="test-key",
            search_client=FakeSearchClient(),  # type: ignore[arg-type]
            chat_model_factory=model_factory,
            luna_max_output_tokens=1024,
            sol_max_output_tokens=4096,
        )

        self.assertEqual(calls[0][2]["max_tokens"], 1024)
        self.assertEqual(calls[1][2]["max_tokens"], 4096)
        self.assertTrue(all(call[2]["use_responses_api"] for call in calls))


if __name__ == "__main__":
    unittest.main()
