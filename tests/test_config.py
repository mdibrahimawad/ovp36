import json
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from ovp36_benchmark.config import load_config, redact_config, resolve_endpoint
from ovp36_benchmark.schemas import EndpointConfig, RunConfig


CONFIG = '''
experiment_label = "stage-1-synthetic"
evaluation = "exploratory"
timeout_seconds = 30.0
concurrency = 1
repetitions = 2

[endpoint]
alias = "local-test"
base_url_env = "TEST_BASE_URL"
model_env = "TEST_MODEL"
api_key_env = "TEST_API_KEY"

[generation]
temperature = 0.0
max_tokens = 4000
seed = 7
top_p = 1.0
'''


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test.toml"
        self.path.write_text(CONFIG, encoding="utf-8")

    def test_toml_load_without_environment_resolution(self):
        config = load_config(self.path)
        self.assertEqual(config.endpoint.base_url_env, "TEST_BASE_URL")
        self.assertIsNone(config.endpoint.base_url)
        self.assertEqual(config.generation.max_tokens, 4000)
        self.assertEqual(config.generation.seed, 7)
        self.assertEqual(config.repetitions, 2)

    def test_environment_resolution_and_secret_redaction(self):
        config = load_config(self.path)
        before = config.model_dump_json()
        env = {"TEST_BASE_URL": "http://127.0.0.1:8080/v1", "TEST_MODEL": "synthetic-model",
               "TEST_API_KEY": "synthetic-secret-do-not-save"}
        resolved = resolve_endpoint(config.endpoint, env)
        self.assertEqual(resolved.base_url, env["TEST_BASE_URL"])
        self.assertEqual(resolved.model, env["TEST_MODEL"])
        self.assertEqual(resolved.api_key.get_secret_value(), env["TEST_API_KEY"])
        for output in (repr(resolved), resolved.model_dump_json(), json.dumps(redact_config(config))):
            self.assertNotIn(env["TEST_API_KEY"], output)
            self.assertNotIn(env["TEST_BASE_URL"], output)
        self.assertEqual(config.model_dump_json(), before)
        self.assertEqual(resolved.model_dump(), {"alias": "local-test"})

    def test_missing_empty_and_invalid_environment_values(self):
        endpoint = load_config(self.path).endpoint
        for env in ({}, {"TEST_BASE_URL": " "},
                    {"TEST_BASE_URL": "http://127.0.0.1/v1", "TEST_MODEL": ""}):
            with self.assertRaisesRegex(ValueError, "environment variable"):
                resolve_endpoint(endpoint, env)
        env = {"TEST_BASE_URL": "http://secret:token@127.0.0.1/v1",
               "TEST_MODEL": "synthetic", "TEST_API_KEY": "another-secret"}
        with self.assertRaises(ValueError) as caught:
            resolve_endpoint(endpoint, env)
        self.assertNotIn("secret", str(caught.exception))
        self.assertNotIn("token", str(caught.exception))

    def test_literal_endpoint_no_key(self):
        endpoint = EndpointConfig(alias="test", base_url="http://127.0.0.1:8080/v1", model="synthetic")
        resolved = resolve_endpoint(endpoint, {})
        self.assertIsNone(resolved.api_key)
        self.assertEqual(resolved.model, "synthetic")

    def test_ambiguous_endpoint_sources_rejected(self):
        base = {"alias": "test", "base_url": "http://127.0.0.1/v1", "model": "synthetic"}
        for patch in ({"base_url_env": "URL"}, {"model_env": "MODEL"},
                      {"base_url": None}, {"model": None}, {"api_key": "do-not-save"},
                      {"api_key_env": "not a variable"}):
            with self.subTest(patch=patch), self.assertRaises(ValidationError):
                EndpointConfig(**(base | patch))

    def test_url_embedded_credentials_and_invalid_urls_rejected(self):
        for url in ("ftp://127.0.0.1", "http://", "http://name:secret@127.0.0.1", "http://@127.0.0.1",
                    "http://127.0.0.1?api_key=secret", "http://127.0.0.1#secret",
                    "http://127.0.0.1:invalid", "http://127.0.0.1:0", "http://host with spaces"):
            with self.subTest(url=url), self.assertRaises(ValidationError):
                EndpointConfig(alias="test", base_url=url, model="synthetic")

    def test_positive_and_strict_run_values(self):
        base = load_config(self.path).model_dump(mode="json")
        for field in ("concurrency", "repetitions", "timeout_seconds"):
            for value in (0, -1, "2", True, None):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    RunConfig.model_validate_json(json.dumps(base | {field: value}))
        for marker in ("official", True):
            with self.assertRaises(ValidationError):
                RunConfig.model_validate_json(json.dumps(base | {"evaluation": marker}))

    def test_generation_constraints(self):
        base = load_config(self.path).model_dump(mode="json")
        for field, values in {"max_tokens": [0, -1, "4000", True],
                              "temperature": [-1, "0", True, float("inf")],
                              "top_p": [0, 1.1, "1"], "seed": ["7", True]}.items():
            for value in values:
                data = base | {"generation": base["generation"] | {field: value}}
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    RunConfig.model_validate_json(json.dumps(data))

    def test_unknown_config_fields_and_wrong_toml_types(self):
        for content in (CONFIG.replace("concurrency = 1", 'concurrency = "1"'),
                        CONFIG + '\nunknown = "value"\n',
                        CONFIG.replace("[endpoint]", '[endpoint]\napi_key = "do-not-save"')):
            self.path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_config(self.path)


if __name__ == "__main__":
    unittest.main()
