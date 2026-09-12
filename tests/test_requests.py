import ast
from dataclasses import FrozenInstanceError
import inspect
import json
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from ovp36_benchmark import requests
from ovp36_benchmark.requests import (
    CapturedReplayRequest, MessageSnapshot, PreparedRequest,
    prepare_captured_request, prepare_generated_request,
)
from ovp36_benchmark.schemas import ContextMessage, GenerationConfig, Source, Task


def messages():
    return [
        {"role": "system", "content": "  Synthetic private instruction.\n\n"},
        {"role": "user", "content": " Hello?\t "},
        {"role": "assistant", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {
                "name": "lookup", "arguments": ' { "x" : 1, "y": "\\n" } '}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": None,
         "unknown": {"nested": [1, "1", True, None, {"text": " café ☎ "}]}},
    ]


def capture(**updates):
    snapshot = MessageSnapshot(messages())
    return CapturedReplayRequest(**({
        "case_id": "synthetic-replay", "task": Task.VOICEMAIL,
        "messages": snapshot, "captured_message_fingerprint": snapshot.fingerprint,
        "source_ref": "synthetic-export", "observation_id": "span-1",
        "historical_generation": {"max_tokens": 999, "temperature": 1.0},
        "historical_output": "Synthetic private historical output",
    } | updates))


class SnapshotTests(unittest.TestCase):
    def test_all_message_fields_preserved(self):
        original = messages()
        snapshot = MessageSnapshot(original)
        restored = snapshot.to_messages()
        self.assertEqual(restored, original)
        self.assertNotIn("content", restored[2])
        self.assertIsNone(restored[3]["content"])
        self.assertEqual(list(map(lambda m: m["role"], restored)),
                         ["system", "user", "assistant", "tool"])
        self.assertIs(type(restored[3]["unknown"]["nested"][0]), int)
        self.assertIs(type(restored[3]["unknown"]["nested"][1]), str)
        self.assertIsInstance(restored[2]["tool_calls"][0]["function"]["arguments"], str)

    def test_mutating_input_or_dispatch_copy_cannot_change_snapshot(self):
        original = messages()
        snapshot = MessageSnapshot(original)
        before = snapshot.fingerprint
        original[2]["tool_calls"][0]["function"]["arguments"] = "changed"
        original.reverse()
        outbound = snapshot.to_messages()
        outbound[3]["unknown"]["nested"].clear()
        self.assertEqual(snapshot.to_messages(), messages())
        self.assertEqual(snapshot.fingerprint, before)
        with self.assertRaises(FrozenInstanceError):
            snapshot._json = "[]"

    def test_invalid_outbound_json_rejected(self):
        invalid = [None, {}, (), [None], ["message"], [{1: "nonstring key"}],
                   [{"x": (1,)}], [{"x": {1}}], [{"x": b"bytes"}]]
        invalid += [[{"nested": [n]}] for n in (float("nan"), float("inf"), -float("inf"))]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                MessageSnapshot(value)

    def test_context_message_contract_remains_strict(self):
        for message in ({"role": "assistant"}, {"role": "user", "content": "x", "unknown": 1}):
            with self.assertRaises(ValidationError):
                ContextMessage(**message)

    def test_snapshot_cannot_be_constructed_from_unvalidated_json_field(self):
        data = {"case_id": "x", "task": "qa", "source": "curated", "model": "m",
                "generation": {"max_tokens": 3}, "messages": {"_json": "[NaN]"}}
        with self.assertRaises(ValidationError):
            PreparedRequest.model_validate_json(json.dumps(data))


class PreparationTests(unittest.TestCase):
    def test_captured_path_preserves_snapshot_and_fingerprint(self):
        captured = capture()
        generation = GenerationConfig(max_tokens=123, temperature=0.25, seed=4)
        prepared = prepare_captured_request(captured, model="synthetic-model", generation=generation)
        self.assertIs(prepared.messages, captured.messages)
        self.assertEqual(prepared.message_fingerprint, captured.captured_message_fingerprint)
        self.assertEqual(prepared.to_request_body()["messages"], messages())
        self.assertEqual(prepared.to_request_body()["max_tokens"], 123)
        self.assertEqual(prepared.to_request_body()["temperature"], 0.25)
        self.assertIs(prepared.captured, captured)
        self.assertEqual(captured.historical_generation["max_tokens"], 999)

    def test_captured_requires_replay_source(self):
        for source in (Source.CURATED, Source.ADVERSARIAL):
            with self.assertRaises(ValidationError):
                capture(source=source)

    def test_captured_fingerprint_mismatch_rejected(self):
        with self.assertRaisesRegex(ValidationError, "fingerprint mismatch"):
            capture(captured_message_fingerprint="0" * 64)

    def test_generated_rejects_replay(self):
        for source in (Source.OVP34_REPLAY, "ovp34_replay"):
            with self.assertRaisesRegex(ValueError, "prepare_captured_request"):
                prepare_generated_request(messages(), case_id="x", task=Task.VOICEMAIL,
                                          source=source, model="m", generation=GenerationConfig(max_tokens=3))

    def test_generated_accepts_both_sources_and_all_tasks(self):
        for source in (Source.CURATED, Source.ADVERSARIAL):
            for task in Task:
                request = prepare_generated_request(messages(), case_id="x", task=task,
                                                    source=source, model="m", generation=GenerationConfig(max_tokens=3))
                self.assertEqual(request.to_request_body()["messages"], messages())
                self.assertIsNone(request.captured)

    def test_prepared_replay_cannot_omit_or_change_capture(self):
        captured = capture()
        data = dict(case_id=captured.case_id, task=captured.task, source=Source.OVP34_REPLAY,
                    model="m", generation=GenerationConfig(max_tokens=5), messages=captured.messages)
        with self.assertRaises(ValidationError):
            PreparedRequest(**data)
        for change in ({"case_id": "other"}, {"task": Task.QA},
                       {"messages": MessageSnapshot([])}, {"source": Source.CURATED}):
            with self.assertRaises(ValidationError):
                PreparedRequest(**(data | {"captured": captured} | change))

    def test_no_rendering_or_adapter_dependencies(self):
        tree = ast.parse(inspect.getsource(requests))
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        self.assertFalse(any("adapter" in name or "contracts" in name for name in imports))
        with patch("ovp36_benchmark.contracts.render_contract", side_effect=AssertionError("rendered")):
            prepare_captured_request(capture(), model="m", generation=GenerationConfig(max_tokens=5))

    def test_private_payload_not_in_repr_or_ordinary_serialization(self):
        captured = capture(qualification="Synthetic private evidence qualification")
        prepared = prepare_captured_request(captured, model="m", generation=GenerationConfig(max_tokens=5))
        for rendered in (repr(captured.messages), repr(captured), repr(prepared),
                         captured.model_dump_json(), prepared.model_dump_json()):
            for private in ("Synthetic private", "lookup", "arguments", "historical_output"):
                self.assertNotIn(private, rendered)

    def test_summary_qualification_is_explicit_provenance_only(self):
        caveat = ("current tracing reconstructs the summary request after generation; "
                  "historical wire equality is not independently proven")
        summary = capture(task=Task.CONTEXT_SUMMARY, qualification=caveat)
        prepared = prepare_captured_request(summary, model="m", generation=GenerationConfig(max_tokens=5))
        self.assertEqual(prepared.captured.qualification, caveat)
        self.assertNotIn("qualification", prepared.to_request_body())
        self.assertIsNone(capture().qualification)

    def test_historical_output_and_settings_do_not_change_candidate_identity(self):
        first = capture()
        second = capture(historical_output={"different": "baseline"},
                         historical_generation={"temperature": 0.8, "seed": 9})
        prepared = [prepare_captured_request(item, model="m", generation=GenerationConfig(max_tokens=7))
                    for item in (first, second)]
        self.assertEqual(prepared[0].request_fingerprint, prepared[1].request_fingerprint)
        self.assertEqual(prepared[0].to_request_body(), prepared[1].to_request_body())

    def test_generation_fields_and_request_allowlist(self):
        for field in ("max_tokens", "max_completion_tokens"):
            for seed in (None, 0, 7):
                generation = GenerationConfig(max_tokens=137, token_limit_field=field, seed=seed)
                prepared = prepare_captured_request(capture(), model="m", generation=generation)
                body = prepared.to_request_body()
                expected_keys = {"messages", "model", "temperature", field, "top_p", "stream"}
                if seed is not None:
                    expected_keys.add("seed")
                    self.assertEqual(body["seed"], seed)
                self.assertEqual(set(body), expected_keys)
                self.assertEqual(body[field], 137)
                self.assertIs(body["stream"], False)
                self.assertEqual(prepared.generation.token_limit_field, field)
        self.assertEqual(GenerationConfig(max_tokens=7).token_limit_field, "max_tokens")
        with self.assertRaises(ValidationError):
            GenerationConfig(max_tokens=7, token_limit_field="max_output_tokens")
        with self.assertRaises(ValidationError):
            PreparedRequest(case_id="x", task=Task.QA, source=Source.CURATED, model="m",
                            generation=GenerationConfig(max_tokens=7), messages=MessageSnapshot([]), stream=True)


if __name__ == "__main__":
    unittest.main()
