import hashlib
import inspect
import json
import unittest

from pydantic import ValidationError

from ovp36_benchmark.config import resolve_endpoint
from ovp36_benchmark.identity import (
    ExecutionKey, IDENTITY_VERSION, canonical_json_bytes, fingerprint_messages,
    fingerprint_request, fingerprint_safe_config, make_run_id,
)
from ovp36_benchmark.requests import prepare_generated_request
from ovp36_benchmark.schemas import EndpointConfig, GenerationConfig, RunConfig, ServerMetadata, Source, Task


def config(url="http://192.0.2.1:8000/v1", **updates):
    return RunConfig(**({
        "endpoint": EndpointConfig(alias="local-test", base_url=url, model="synthetic-model"),
        "generation": GenerationConfig(max_tokens=137), "experiment_label": "synthetic",
        "evaluation": "canonical", "server": ServerMetadata(model_revision="revision-a"),
    } | updates))


def run_components(**updates):
    cfg = config()
    return {
        "dataset_hash": "a" * 64, "ordered_execution_plan_hash": "b" * 64,
        "safe_config_fingerprint": fingerprint_safe_config(cfg, resolved_model="synthetic-model"),
        "requested_model": "synthetic-model", "endpoint_alias": cfg.endpoint.alias,
        "server": cfg.server, "contract_hashes": {"qa": "c" * 64},
        "harness_code_fingerprint": "d" * 64, "dependency_lock_fingerprint": "e" * 64,
    } | updates


class IdentityTests(unittest.TestCase):
    def test_canonical_key_order_and_versioned_domain(self):
        first, second = [{"b": 2, "a": 1}], [{"a": 1, "b": 2}]
        self.assertEqual(canonical_json_bytes(first), b'[{"a":1,"b":2}]')
        self.assertEqual(fingerprint_messages(first), fingerprint_messages(second))
        expected = hashlib.sha256(f"{IDENTITY_VERSION}:messages\0".encode() + canonical_json_bytes(first)).hexdigest()
        self.assertEqual(fingerprint_messages(first), expected)
        self.assertNotEqual(fingerprint_messages(first), hashlib.sha256(canonical_json_bytes(first)).hexdigest())

    def test_message_semantics_change_fingerprint(self):
        pairs = [([{"a": 1}, {"b": 2}], [{"b": 2}, {"a": 1}]),
                 ([{}], [{"content": None}]), ([{"content": "x"}], [{"content": " x"}]),
                 ([{"x": 1}], [{"x": "1"}]), ([{"x": 1}], [{"x": 1.0}]),
                 ([{"arguments": '{"x":1}'}], [{"arguments": '{ "x": 1 }'}]),
                 ([{"arguments": '{"x":1}'}], [{"arguments": {"x": 1}}])]
        for left, right in pairs:
            self.assertNotEqual(fingerprint_messages(left), fingerprint_messages(right))

    def test_nonstandard_json_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), (1, 2), {1: "x"}, b"x", object()):
            with self.assertRaises(ValueError):
                canonical_json_bytes({"nested": [value]})
        cycle = []
        cycle.append(cycle)
        with self.assertRaisesRegex(ValueError, "cyclic or too deeply nested"):
            canonical_json_bytes(cycle)

    def test_canonical_unicode_and_whitespace_roundtrip(self):
        value = {"x": " café\t\n☎ ", "y": [True, None, 1.5, "1.5"]}
        self.assertEqual(json.loads(canonical_json_bytes(value)), value)

    def test_request_fingerprint_only_dispatch_semantics(self):
        data = dict(messages=[{"role": "user", "content": " x "}], case_id="x", task=Task.QA,
                    source=Source.CURATED, model="model-a", generation=GenerationConfig(max_tokens=137))
        request = prepare_generated_request(**data)
        original = fingerprint_request(request)
        expected = hashlib.sha256(f"{IDENTITY_VERSION}:request\0".encode() +
                                  canonical_json_bytes(request.to_request_body())).hexdigest()
        self.assertEqual(original, expected)
        self.assertEqual(request.request_fingerprint, original)
        for change in ({"model": "model-b"}, {"generation": GenerationConfig(max_tokens=138)},
                       {"generation": GenerationConfig(max_tokens=137, temperature=0.5)},
                       {"generation": GenerationConfig(max_tokens=137, top_p=0.5)},
                       {"generation": GenerationConfig(max_tokens=137, seed=0)},
                       {"generation": GenerationConfig(max_tokens=137, token_limit_field="max_completion_tokens")},
                       {"messages": [{"role": "user", "content": "x"}]}):
            self.assertNotEqual(original, fingerprint_request(prepare_generated_request(**(data | change))))
        for change in ({"case_id": "other"}, {"task": Task.EXTRACTION}, {"source": Source.ADVERSARIAL}):
            self.assertEqual(original, fingerprint_request(prepare_generated_request(**(data | change))))

    def test_config_fingerprint_url_independent_but_settings_sensitive(self):
        original = fingerprint_safe_config(config(), resolved_model="synthetic-model")
        self.assertEqual(original, fingerprint_safe_config(config(url="http://192.0.2.2:9000/v1"),
                                                          resolved_model="synthetic-model"))
        for changes in ({"timeout_seconds": 12.0}, {"repetitions": 2}, {"concurrency": 2},
                        {"experiment_label": "other"}, {"evaluation": "exploratory"}):
            self.assertNotEqual(original, fingerprint_safe_config(config(**changes), resolved_model="synthetic-model"))

    def test_run_identity_deterministic_and_all_components_sensitive(self):
        original = make_run_id(**run_components())
        self.assertEqual(original, make_run_id(**run_components()))
        changes = {
            "dataset_hash": "f" * 64, "ordered_execution_plan_hash": "f" * 64,
            "safe_config_fingerprint": "f" * 64, "requested_model": "model-b",
            "endpoint_alias": "research-b", "server": ServerMetadata(model_revision="revision-b"),
            "contract_hashes": {"qa": "f" * 64}, "harness_code_fingerprint": "f" * 64,
            "dependency_lock_fingerprint": "f" * 64,
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                self.assertNotEqual(original, make_run_id(**run_components(**{field: value})))
        self.assertEqual(make_run_id(**run_components(contract_hashes={"qa": "a" * 64, "voicemail": "b" * 64})),
                         make_run_id(**run_components(contract_hashes={"voicemail": "b" * 64, "qa": "a" * 64})))

    def test_identity_functions_no_timestamp_url_or_credentials_parameters(self):
        for function in (make_run_id, fingerprint_request):
            params = inspect.signature(function).parameters
            for field in ("timestamp", "base_url", "url_hash", "api_key", "headers"):
                self.assertNotIn(field, params)
        for field in ("timestamp", "base_url", "url_hash", "api_key", "headers"):
            with self.assertRaises(TypeError):
                make_run_id(**(run_components() | {field: "disallowed"}))

    def test_run_id_unchanged_when_only_runtime_url_changes(self):
        ids = [make_run_id(**run_components(safe_config_fingerprint=fingerprint_safe_config(
            config(url=url), resolved_model="synthetic-model")))
            for url in ("http://192.0.2.1:8000/v1", "http://192.0.2.2:9000/v1")]
        self.assertEqual(*ids)

    def test_environment_reference_changes_and_key_rotation_do_not_change_identity(self):
        fingerprints = []
        for suffix in ("A", "B"):
            endpoint = EndpointConfig(alias="local-test", base_url_env=f"URL_{suffix}",
                                      model_env=f"MODEL_{suffix}", api_key_env=f"KEY_{suffix}")
            cfg = config(endpoint=endpoint)
            resolved = resolve_endpoint(endpoint, {
                f"URL_{suffix}": "http://192.0.2.1:8000/v1",
                f"MODEL_{suffix}": "synthetic-model", f"KEY_{suffix}": f"synthetic-placeholder-{suffix}",
            })
            fingerprints.append(fingerprint_safe_config(cfg, resolved_model=resolved.model))
        self.assertEqual(*fingerprints)

    def test_identity_components_reject_arbitrary_text(self):
        for field in ("dataset_hash", "harness_code_fingerprint", "dependency_lock_fingerprint"):
            with self.assertRaises(ValidationError):
                make_run_id(**run_components(**{field: "not-a-digest"}))
        for changes in ({"requested_model": "/tmp/model"}, {"endpoint_alias": "a/b"},
                        {"contract_hashes": {"../qa": "a" * 64}}):
            with self.assertRaises(ValidationError):
                make_run_id(**run_components(**changes))

    def test_execution_key_zero_based_and_strict(self):
        data = dict(run_id=make_run_id(**run_components()), case_id="case-1",
                    request_fingerprint="a" * 64, repetition_index=0)
        self.assertEqual(ExecutionKey(**data).repetition_index, 0)
        self.assertNotEqual(ExecutionKey(**data), ExecutionKey(**(data | {"repetition_index": 1})))
        for value in (-1, True, "0"):
            with self.assertRaises(ValidationError):
                ExecutionKey(**(data | {"repetition_index": value}))


if __name__ == "__main__":
    unittest.main()
