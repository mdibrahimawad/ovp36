import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from ovp36_benchmark.config import (
    load_config, persistent_config_projection, redact_config, resolve_endpoint,
)
from ovp36_benchmark.schemas import EndpointConfig, GenerationConfig, RunConfig, ServerMetadata


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


class PersistentConfigTests(unittest.TestCase):
    def config(self, **updates):
        return RunConfig(**({
            "endpoint": EndpointConfig(alias="local-test", base_url_env="TEST_BASE_URL",
                                       model_env="TEST_MODEL", api_key_env="TEST_API_KEY"),
            "generation": GenerationConfig(max_tokens=137, token_limit_field="max_completion_tokens"),
            "experiment_label": "stage-3a-synthetic", "evaluation": "exploratory",
        } | updates))

    def test_safe_aliases(self):
        for alias in ("local_granite", "local-test", "gb10_research_a", "spark.dev.a", "A1"):
            self.assertEqual(EndpointConfig(alias=alias, base_url_env="URL", model="m").alias, alias)

    def test_unsafe_aliases_rejected(self):
        # Reserved documentation addresses only; never real infrastructure.
        for alias in ("http://192.0.2.1:8000", "https://example.invalid", "192.0.2.1:8000",
                      "192.0.2.1", "/Users/example/server", "../server", "a..b", "user@host",
                      "a b", "a\n", "a\t", "a/b", "a\\b", "a" * 65, ""):
            with self.subTest(alias=alias), self.assertRaises(ValidationError):
                EndpointConfig(alias=alias, base_url_env="URL", model="m")

    def test_safe_metadata_versions_and_checkpoint_names(self):
        server = ServerMetadata(runtime="test-runtime", runtime_version="1.4.0.dev8128+build",
                                model_checkpoint="example-org/synthetic-model", model_revision="rev-a1",
                                model_artifact_hash="a" * 64, quantization="Q4_K_M",
                                chat_template_hash="b" * 64, context_limit=8192)
        config = self.config(server=server, evaluation="canonical")
        result = persistent_config_projection(config, resolved_model="example-org/synthetic-model")
        self.assertEqual(result["server"], server.model_dump())

    def test_obviously_unsafe_metadata_rejected_in_all_string_fields(self):
        invalid = ("https://example.invalid/model", "192.0.2.1:8000", "192.0.2.1",
                   "/tmp/model", "../model", "C:\\models\\model", "~/model", "a/b/c",
                   "a\nb", "a\rb", "a\x00b", "a\tb", "Bearer synthetic-placeholder",
                   "api_key=synthetic-placeholder", "user:password@host", "sk-synthetic-placeholder",
                   "hf_synthetic_placeholder", "a" * 201)
        for field in ServerMetadata.model_fields:
            if field == "context_limit":
                continue
            for value in invalid:
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    ServerMetadata(**{field: value})

    def test_canonical_requires_useful_identity_exploratory_allows_unknown(self):
        for server in (ServerMetadata(), ServerMetadata(runtime="test"),
                       ServerMetadata(runtime_version="1.0"), ServerMetadata(quantization="q4", context_limit=8192)):
            self.config(server=server)
            with self.assertRaisesRegex(ValidationError, "canonical evaluation requires"):
                self.config(evaluation="canonical", server=server)
        for values in ({"model_checkpoint": "model"}, {"model_revision": "revision"},
                       {"model_artifact_hash": "a" * 64}, {"chat_template_hash": "b" * 64},
                       {"runtime": "test", "runtime_version": "1.0"}):
            self.config(evaluation="canonical", server=ServerMetadata(**values))

    def test_projection_exact_allowlist_and_no_environment_leaks(self):
        config = self.config()
        env = {"TEST_BASE_URL": "http://192.0.2.1:8000/v1", "TEST_MODEL": "synthetic-model",
               "TEST_API_KEY": "synthetic-secret-do-not-save"}
        resolved = resolve_endpoint(config.endpoint, env)
        projected = persistent_config_projection(config, resolved_model=resolved.model)
        self.assertEqual(projected, {
            "endpoint_alias": "local-test", "requested_model": "synthetic-model",
            "generation": {"temperature": 0.0, "max_tokens": 137,
                           "token_limit_field": "max_completion_tokens", "top_p": 1.0},
            "timeout_seconds": 30.0, "repetitions": 1, "concurrency": 1,
            "experiment_label": "stage-3a-synthetic", "evaluation": "exploratory", "server": {},
        })
        text = json.dumps(projected)
        for private in (env["TEST_BASE_URL"], env["TEST_API_KEY"], *env.keys(),
                        "Authorization", "Bearer", "base_url", "api_key", "headers",
                        hashlib.sha256(env["TEST_BASE_URL"].encode()).hexdigest()):
            self.assertNotIn(private, text)
        self.assertEqual(projected, persistent_config_projection(config, resolved_model=resolved.model))

    def test_literal_url_change_has_no_effect_and_old_redaction_unchanged(self):
        projections = []
        for url in ("http://192.0.2.1:8000/v1", "http://192.0.2.2:9000/v1"):
            config = self.config(endpoint=EndpointConfig(alias="local-test", base_url=url, model="m"))
            projection = persistent_config_projection(config, resolved_model="m")
            self.assertNotIn(url, json.dumps(projection))
            self.assertEqual(redact_config(config)["endpoint"]["base_url"], url)
            projections.append(projection)
        self.assertEqual(*projections)

    def test_projection_does_not_dump_then_redact(self):
        from unittest.mock import patch
        config = self.config()
        with patch.object(RunConfig, "model_dump", side_effect=AssertionError("unsafe dump")):
            persistent_config_projection(config, resolved_model="m")

    def test_model_label_and_seed_persistence_policy(self):
        for model in ("https://example.invalid", "/tmp/model", "Bearer synthetic-placeholder"):
            with self.assertRaises(ValidationError):
                persistent_config_projection(self.config(), resolved_model=model)
        for label in ("/tmp/experiment", "a\nb", "api_key=synthetic"):
            with self.assertRaises(ValidationError):
                persistent_config_projection(self.config(experiment_label=label), resolved_model="m")
        config = self.config(generation=GenerationConfig(max_tokens=4, seed=0))
        self.assertEqual(persistent_config_projection(config, resolved_model="m")["generation"]["seed"], 0)
        endpoint = EndpointConfig(alias="test", base_url_env="URL", model="literal")
        with self.assertRaisesRegex(ValueError, "differs"):
            persistent_config_projection(self.config(endpoint=endpoint), resolved_model="other")


if __name__ == "__main__":
    unittest.main()
