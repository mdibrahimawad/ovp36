import hashlib
import inspect
import json
import unittest

from pydantic import ValidationError

from ovp36_benchmark.config import resolve_endpoint
from ovp36_benchmark.identity import (
    ExecutionKey, IDENTITY_VERSION, canonical_json_bytes, fingerprint_messages,
    fingerprint_request, fingerprint_safe_config, make_run_id, fingerprint_execution_plan,
    fingerprint_replay_dataset,
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
    def test_replay_dataset_golden_order_and_field_sensitivity(self):
        entry = dict(case_id="ovp34-0000000000000001", contract="variable_extraction",
                     source_observation_id="0000000000000001", message_fingerprint="a" * 64)
        digest = fingerprint_replay_dataset([entry])
        self.assertEqual(digest, "468aeb55285ebb6a02948b588fe5f9ddbc1ee72bff17affc85d9492764104d9c")
        self.assertEqual(fingerprint_replay_dataset([]),
                         "43648bb35a692fb935aaf6449cecf6b8856beaed5f736f0b9db49a6980fbb4d8")
        self.assertEqual(digest, fingerprint_replay_dataset([dict(reversed(list(entry.items())))]))
        for change in ({"case_id": "other-safe-case"}, {"source_observation_id": "0000000000000002"},
                       {"message_fingerprint": "b" * 64}, {"contract": "runtime_context_summary"},
                       {"contract": "voicemail_detection"}, {"contract": "qa_evaluation"},
                       {"contract": "qa_conversation_summary"}):
            other = entry | change
            self.assertNotEqual(digest, fingerprint_replay_dataset([other]))
            self.assertNotEqual(fingerprint_replay_dataset([entry, other]),
                                fingerprint_replay_dataset([other, entry]))

    def test_replay_dataset_rejects_invalid_projections_safely(self):
        entry = dict(case_id="ovp34-0000000000000001", contract="variable_extraction",
                     source_observation_id="0000000000000001", message_fingerprint="a" * 64)
        invalid = [None, (), {}, "PRIVATE_MARKER", [None], [list(entry)],
                   [entry | {"messages": "PRIVATE_MARKER"}]]
        invalid.extend([{k: v for k, v in entry.items() if k != field}] for field in entry)
        for field, values in {
            "case_id": ["../PRIVATE_MARKER", "", True],
            "contract": ["qa_node_summary", "PRIVATE_MARKER", None, True],
            "source_observation_id": ["A" * 16, "a" * 15, "a" * 17, "a" * 16 + "\n", 1],
            "message_fingerprint": ["A" * 64, "bad", None],
        }.items():
            invalid.extend([[entry | {field: value}] for value in values])
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                fingerprint_replay_dataset(value)
            self.assertEqual(str(caught.exception), "invalid_replay_dataset_projection")
            self.assertIsNone(caught.exception.__cause__)
            self.assertIsNone(caught.exception.__context__)

    def test_stage_3a_identity_golden_digests_unchanged(self):
        messages = [{"role": "user", "content": " x "}]
        request = prepare_generated_request(messages, case_id="x", task=Task.QA,
            source=Source.CURATED, model="model-a", generation=GenerationConfig(max_tokens=137))
        self.assertEqual(fingerprint_messages(messages),
                         "a884c388d6130007c7d5baee2aa475074b96accbc1894bdd87a309d56715cebb")
        self.assertEqual(fingerprint_request(request),
                         "3f65fb74cd450622a67275b3e94edaadc49a2a30d4a29c2ac317b45c6c825f5e")
        self.assertEqual(fingerprint_safe_config(config(), resolved_model="synthetic-model"),
                         "cbbb1ce399bc1e227bf34b61214c2cc9a3ddee1988bc7af66ef56de2f73dfcb4")
        self.assertEqual(make_run_id(**run_components()),
                         "36d1809a1a8d344d32476494bcddddde5f280e448b6c76c97ea762cb8f651870")

    def test_execution_plan_digest_and_order(self):
        entry = dict(case_id="case-a", request_fingerprint="a" * 64, repetition_index=0)
        digest = fingerprint_execution_plan([entry])
        self.assertEqual(digest, "1250d7f71ac76011a0fd7b7d7cfced7e8b6e9d227c8b635bd025eda9e71fbcf3")
        self.assertEqual(digest, fingerprint_execution_plan([dict(reversed(list(entry.items())))]))
        for change in ({"case_id": "case-b"}, {"request_fingerprint": "b" * 64}, {"repetition_index": 1}):
            other = entry | change
            self.assertNotEqual(digest, fingerprint_execution_plan([other]))
            self.assertNotEqual(fingerprint_execution_plan([entry, other]),
                                fingerprint_execution_plan([other, entry]))
        # Duplicates are valid projections; only the runner rejects duplicate executions.
        self.assertNotEqual(digest, fingerprint_execution_plan([entry, entry]))
        self.assertNotEqual(digest, fingerprint_execution_plan([]))

    def test_execution_plan_strict_projection_and_safe_errors(self):
        entry = dict(case_id="case-a", request_fingerprint="a" * 64, repetition_index=0)
        malformed = [None, "private-marker", (), {}, [None], [list(entry)],
                     [entry | {"messages": "private-marker"}],
                     [{key: value for key, value in entry.items() if key != "case_id"}]]
        for field, values in {"case_id": ["../private-marker", "", 1],
                              "request_fingerprint": ["private-marker", "A" * 64, 12],
                              "repetition_index": [True, "0", -1, 0.0, None]}.items():
            malformed.extend([[entry | {field: value}] for value in values])
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                fingerprint_execution_plan(value)
            self.assertEqual(str(caught.exception), "invalid_execution_plan_projection")
            self.assertIsNone(caught.exception.__cause__)
            self.assertIsNone(caught.exception.__context__)

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
