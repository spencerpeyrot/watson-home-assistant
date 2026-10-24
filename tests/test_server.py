from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from service.artifacts import LocalArtifactStore
from service.postgres import PostgresStore
from service.server import DEFAULT_LIVE_INSTRUCTIONS, ServerSettings, create_app


OWNER_ID = "00000000-0000-4000-8000-000000000801"
OWNER_TOKEN = "o" * 48
DEVICE_TOKEN = "d" * 48
OPERATOR_TOKEN = "x" * 48
MUSE_TOKEN = "mgst_" + ("A" * 42) + "A"


def configuration(**overrides: str) -> dict[str, str]:
    values = {
        "DATABASE_URL": "postgresql://test.invalid/cube",
        "OPENAI_API_KEY": "test-openai-key-never-logged",
        "CUBE_OWNER_ID": OWNER_ID,
        "CUBE_DEVICE_ID": "cube-01",
        "CUBE_OWNER_TOKEN": OWNER_TOKEN,
        "CUBE_DEVICE_TOKEN": DEVICE_TOKEN,
        "CUBE_OPERATOR_TOKEN": OPERATOR_TOKEN,
    }
    values.update(overrides)
    return values


class FakeStore:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.open_count = 0
        self.close_count = 0
        self.migrations = 0
        self.healthchecks = 0

    async def open(self):
        self.open_count += 1

    async def close(self):
        self.close_count += 1

    async def apply_migrations(self):
        self.migrations += 1
        return ("0001_initial",)

    async def healthcheck(self):
        self.healthchecks += 1
        return True


class ServerSettingsTests(unittest.TestCase):
    def test_settings_default_to_no_automatic_migration_or_session_cap(self):
        settings = ServerSettings.from_environment(configuration())

        self.assertEqual(settings.live_model, "gpt-live-1")
        self.assertEqual(settings.live_voice, "marin")
        self.assertEqual(settings.live_instructions, DEFAULT_LIVE_INSTRUCTIONS)
        self.assertEqual(settings.voice_stack, "v1")
        self.assertFalse(settings.auto_migrate)
        self.assertIsNone(settings.max_session_seconds)
        self.assertNotIn("test-openai-key-never-logged", repr(settings))
        self.assertNotIn("test.invalid", repr(settings))

    def test_settings_parse_explicit_caps_and_reject_bad_values(self):
        settings = ServerSettings.from_environment(
            configuration(
                CUBE_AUTO_MIGRATE="true",
                CUBE_IDLE_CLOSE_SECONDS="15",
                CUBE_MAX_SESSION_SECONDS="900",
            )
        )
        self.assertTrue(settings.auto_migrate)
        self.assertEqual(settings.idle_close_seconds, 15)
        self.assertEqual(settings.max_session_seconds, 900)
        with self.assertRaisesRegex(RuntimeError, "CUBE_AUTO_MIGRATE"):
            ServerSettings.from_environment(configuration(CUBE_AUTO_MIGRATE="yes"))
        with self.assertRaisesRegex(RuntimeError, "CUBE_MAX_SESSION_SECONDS"):
            ServerSettings.from_environment(configuration(CUBE_MAX_SESSION_SECONDS="0"))

    def test_artifact_directory_is_shared_and_local_store_is_composed(self):
        with tempfile.TemporaryDirectory() as temporary:
            values = configuration(CUBE_ARTIFACT_DIR=temporary)
            settings = ServerSettings.from_environment(values)
            self.assertEqual(settings.artifact_root, Path(temporary))

            app = create_app(
                values,
                store_factory=lambda url: FakeStore(url),  # type: ignore[arg-type]
            )

            self.assertIsInstance(app.state.artifact_store, LocalArtifactStore)
            self.assertEqual(app.state.artifact_store.root, Path(temporary).resolve())

    def test_postgres_server_composes_timer_worker_without_connecting(self):
        values = configuration()
        store = PostgresStore(values["DATABASE_URL"])

        async def no_network_connector(*_args, **_kwargs):
            raise AssertionError("app construction must not connect to OpenAI")

        app = create_app(
            values,
            store_factory=lambda _database_url: store,
            live_connector=no_network_connector,
        )

        self.assertIs(app.state.store, store)
        self.assertIsNotNone(app.state.timer_worker)

    def test_settings_errors_name_missing_setting_without_echoing_secrets(self):
        values = configuration()
        del values["OPENAI_API_KEY"]
        with self.assertRaises(RuntimeError) as caught:
            ServerSettings.from_environment(values)
        self.assertIn("OPENAI_API_KEY", str(caught.exception))
        self.assertNotIn(OWNER_TOKEN, str(caught.exception))

    def test_muse_stack_requires_secrets_and_keeps_them_out_of_repr(self):
        with self.assertRaisesRegex(RuntimeError, "MUSE_SDK_TOKEN"):
            ServerSettings.from_environment(configuration(CUBE_VOICE_STACK="muse"))
        muse_values = configuration(
            CUBE_VOICE_STACK="muse",
            MUSE_SDK_TOKEN=MUSE_TOKEN,
            TYPESAFE_API_KEY="typesafe-test-secret-key",
            ELEVENLABS_API_KEY="eleven-test-secret-key",
            ELEVENLABS_VOICE_ID="voice-abc123",
            OPENAI_API_KEY="openai-fallback-test-key",
            OPENAI_LIVE_MODEL="",
            OPENAI_LIVE_VOICE="",
            CUBE_LIVE_INSTRUCTIONS="",
        )
        settings = ServerSettings.from_environment(
            muse_values
        )
        self.assertEqual(settings.voice_stack, "muse")
        self.assertEqual(settings.openai_api_key, "openai-fallback-test-key")
        self.assertEqual(settings.typesafe_api_key, "typesafe-test-secret-key")
        self.assertEqual(settings.elevenlabs_model, "eleven_flash_v2_5")
        self.assertNotIn(MUSE_TOKEN, repr(settings))
        self.assertNotIn("eleven-test-secret-key", repr(settings))
        self.assertNotIn("typesafe-test-secret-key", repr(settings))

    def test_muse_stack_composes_without_openai_voice_connection(self):
        values = configuration(
            CUBE_VOICE_STACK="muse",
            MUSE_SDK_TOKEN=MUSE_TOKEN,
            TYPESAFE_API_KEY="typesafe-test-secret-key",
            ELEVENLABS_API_KEY="eleven-test-secret-key",
            ELEVENLABS_VOICE_ID="voice-abc123",
            OPENAI_API_KEY="openai-fallback-test-key",
        )
        built = []

        class FakeMuseHost:
            def __init__(self, token):
                self.token = token

            async def run(self, _stop):
                return None

            async def wait_for_link(self, _timeout=20):
                raise AssertionError("app construction must not connect to Muse")

        def host_factory(token):
            value = FakeMuseHost(token)
            built.append(value)
            return value

        store = PostgresStore(values["DATABASE_URL"])
        app = create_app(
            values,
            store_factory=lambda _url: store,
            live_connector=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Muse stack must not build an OpenAI voice connection")
            ),
            muse_host_factory=host_factory,  # type: ignore[arg-type]
        )

        self.assertEqual(app.state.voice_stack, "muse")
        self.assertIs(app.state.muse_host, built[0])
        self.assertEqual(built[0].token, MUSE_TOKEN)
        self.assertIsNone(app.state.timer_worker)
        self.assertEqual(app.state.live_session_factory.__name__, "create_provider")

    def test_app_factory_wires_auth_live_provider_and_store_lifecycle(self):
        values = configuration(CUBE_AUTO_MIGRATE="true")
        store = FakeStore(values["DATABASE_URL"])

        async def no_network_connector(*_args, **_kwargs):
            raise AssertionError("app creation/start must not connect to OpenAI")

        app = create_app(
            values,
            store_factory=lambda _database_url: store,  # type: ignore[arg-type]
            live_connector=no_network_connector,
        )
        self.assertIs(app.state.store, store)
        self.assertEqual(store.open_count, 0)

        with TestClient(app) as client:
            response = client.get(
                "/v1/health", headers={"Authorization": f"Bearer {OPERATOR_TOKEN}"}
            )
            self.assertEqual(response.status_code, 200)

        self.assertEqual(store.open_count, 1)
        self.assertEqual(store.migrations, 1)
        self.assertEqual(store.close_count, 1)
        self.assertEqual(store.healthchecks, 1)


if __name__ == "__main__":
    unittest.main()
