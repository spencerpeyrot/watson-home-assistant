from __future__ import annotations

import configparser
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"


def read_unit(name: str) -> configparser.ConfigParser:
    # systemd permits repeated directives such as EnvironmentFile. Tests that
    # need every occurrence use read_directives below.
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    with (DEPLOY / name).open(encoding="utf-8") as source:
        parser.read_file(source)
    return parser


def read_directives(name: str, section: str, key: str) -> list[str]:
    current_section = ""
    values: list[str] = []
    for raw_line in (DEPLOY / name).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            continue
        directive, separator, value = line.partition("=")
        if current_section == section and separator and directive == key:
            values.append(value)
    return values


def read_environment_example(name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (DEPLOY / name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ValueError(f"invalid environment example: {name}")
        values[key] = value
    return values


class DeploymentTemplateTests(unittest.TestCase):
    def test_backend_units_use_current_release_and_separate_processes(self) -> None:
        api = read_unit("cube-assistant-api.service")["Service"]
        worker = read_unit("cube-assistant-worker.service")["Service"]
        migration = read_unit("cube-assistant-migrate.service")["Service"]

        self.assertEqual(api["User"], "cube-assistant")
        self.assertEqual(api["Group"], "cube-assistant")
        self.assertIn("python -m service.api", api["ExecStart"])
        self.assertIn("--host 127.0.0.1", api["ExecStart"])
        self.assertIn("current/.venv/bin/", api["ExecStart"])
        self.assertEqual(worker["ExecStart"], "/opt/cube-assistant/current/.venv/bin/python -m service.worker")
        self.assertEqual(migration["Type"], "oneshot")
        self.assertEqual(migration["ExecStart"], "/opt/cube-assistant/current/.venv/bin/python -m service.migrate")
        self.assertNotIn("[Install]", read_unit("cube-assistant-migrate.service"))

    def test_secret_environment_examples_are_blank_and_outside_repo(self) -> None:
        api_unit = read_unit("cube-assistant-api.service")["Service"]
        worker_unit = read_unit("cube-assistant-worker.service")["Service"]
        migrate_unit = read_unit("cube-assistant-migrate.service")["Service"]
        api = read_environment_example("api.env.example")
        worker = read_environment_example("worker.env.example")
        migrate = read_environment_example("migrate.env.example")

        self.assertIn(
            "/etc/cube-assistant/api.env",
            read_directives("cube-assistant-api.service", "Service", "EnvironmentFile"),
        )
        self.assertIn(
            "/etc/cube-assistant/worker.env",
            read_directives("cube-assistant-worker.service", "Service", "EnvironmentFile"),
        )
        self.assertIn(
            "/etc/cube-assistant/migrate.env",
            read_directives("cube-assistant-migrate.service", "Service", "EnvironmentFile"),
        )
        for env_values, secret_names in (
            (api, ("DATABASE_URL", "OPENAI_API_KEY", "TYPESAFE_API_KEY", "CUBE_OWNER_TOKEN", "CUBE_DEVICE_TOKEN")),
            (worker, ("DATABASE_URL", "OPENAI_API_KEY", "CUBE_WORKER_TOKEN")),
            (migrate, ("DATABASE_URL",)),
        ):
            for name in secret_names:
                self.assertEqual(env_values[name], "")

    def test_api_worker_hardening_and_timer_release_path(self) -> None:
        for name in ("cube-assistant-api.service", "cube-assistant-worker.service"):
            service = read_unit(name)["Service"]
            for directive in (
                "NoNewPrivileges",
                "PrivateTmp",
                "ProtectSystem",
                "ProtectHome",
                "RestrictAddressFamilies",
            ):
                self.assertIn(directive, service)
            self.assertEqual(service["ProtectSystem"], "strict")
            self.assertEqual(service["NoNewPrivileges"], "true")

        timer = read_unit("cube-assistant-timer.service")["Service"]
        self.assertEqual(timer["WorkingDirectory"], "/opt/cube-assistant/current")
        self.assertIn("/opt/cube-assistant/current/.venv/bin/python", timer["ExecStart"])


if __name__ == "__main__":
    unittest.main()
