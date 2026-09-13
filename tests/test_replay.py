"""Entirely synthetic telemetry; no private files, checksum bypasses, or model calls."""

import ast
from collections import Counter
import copy
import csv
import hashlib
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from ovp36_benchmark import replay
from ovp36_benchmark.identity import canonical_json_bytes, fingerprint_execution_plan
from ovp36_benchmark.replay import (
    CapturedReplayCase, ReplayContract, ReplayDataset, ReplayError,
    load_ovp34_replay, prepare_replay_plan,
)
from ovp36_benchmark.runner import PlannedExecution
from ovp36_benchmark.schemas import EndpointConfig, GenerationConfig, RunConfig, Source, Task


DISTRIBUTION = (
    ("variable_extraction", "llm-variable-extraction", "variable_extraction", Task.EXTRACTION, 40),
    ("context_summarization", "llm-context-summarization", "runtime_context_summary", Task.CONTEXT_SUMMARY, 56),
    ("voicemail_detection", "llm-voicemail-detect", "voicemail_detection", Task.VOICEMAIL, 22),
    ("post_call_qa", "qa-node-Synthetic", "qa_evaluation", Task.QA, 72),
    ("post_call_qa_summary", "conversation-summary-before-Synthetic", "qa_conversation_summary", Task.QA_CONVERSATION_SUMMARY, 52),
)


def messages():
    return [
        {"role": "system", "content": "  SYNTHETIC_PRIVATE_REQUEST\n\n"},
        {"role": "user", "content": " café ☎ مرحبا\t "},
        {"role": "assistant", "tool_calls": [{"id": "synthetic-call", "type": "function",
            "function": {"name": "synthetic_lookup", "arguments": ' { "number" : 1, "text": "\\n" } '}}]},
        {"role": "assistant", "content": None},
        {"role": "tool", "tool_call_id": "synthetic-call", "content": "SYNTHETIC_TOOL_RESULT",
         "unknown": {"nested": [1, 1.0, "1", True, None]}},
    ]


def fixture():
    rows, selection = [], []
    for job, name, _, _, count in DISTRIBUTION:
        for _ in range(count):
            observation_id = f"{len(rows) + 1:016x}"
            trace_id = "synthetic-trace"
            rows.append({"index": {"observation_id": observation_id, "trace_id": trace_id},
                "raw": {"id": observation_id, "traceId": trace_id, "type": "GENERATION", "name": name,
                        "input": {"messages": messages()}, "output": {"content": "SYNTHETIC_BASELINE"},
                        "model": "historical-synthetic-model", "modelParameters": {"max_tokens": 999}}})
            selection.append(dict(observation_id=observation_id, trace_id=trace_id, job_type=job, name=name))
    # These are excluded by ID even when content is identical or a recognized name is present.
    for offset, name in enumerate(("llm", "llm-variable-extraction", "node-summary-Synthetic"), 1000):
        row = copy.deepcopy(rows[0])
        row["raw"].update(id=f"{offset:016x}", name=name)
        row["index"]["observation_id"] = row["raw"]["id"]
        rows.append(row)
    return rows, selection


def config(**updates):
    return RunConfig(**(dict(
        endpoint=EndpointConfig(alias="synthetic", base_url="http://synthetic.invalid/v1", model_env="CANDIDATE"),
        generation=GenerationConfig(max_tokens=137), evaluation="exploratory", experiment_label="synthetic",
    ) | updates))


def encoded(rows, selection, *, alternate=False):
    raw = b"".join((json.dumps(row, sort_keys=alternate, ensure_ascii=not alternate,
                               separators=(",", ":") if alternate else None) + "\n").encode()
                   for row in rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(selection[0]) if selection else
                            ["observation_id", "trace_id", "job_type", "name"],
                            quoting=csv.QUOTE_ALL if alternate else csv.QUOTE_MINIMAL,
                            lineterminator="\n" if alternate else "\r\n")
    writer.writeheader()
    writer.writerows(selection)
    return raw, stream.getvalue().encode()


class ReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.selection = self.root / "source.jsonl", self.root / "selection.csv"
        self.rows, self.selected = fixture()

    def load(self, rows=None, selection=None, *, alternate=False):
        raw, csv_data = encoded(self.rows if rows is None else rows,
                                self.selected if selection is None else selection, alternate=alternate)
        self.source.write_bytes(raw)
        self.selection.write_bytes(csv_data)
        return load_ovp34_replay(self.source, selection_path=self.selection)

    def assert_safe_failure(self, function, category=None):
        with self.assertRaises(ReplayError) as caught:
            function()
        error = caught.exception
        if category:
            self.assertIn(category, str(error))
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)
        for private in ("PRIVATE", "SYNTHETIC_BASELINE", str(self.root), "synthetic.invalid"):
            self.assertNotIn(private, repr(error))

    def test_public_loader_full_distribution_csv_order_and_unique_ids(self):
        dataset = self.load(rows=list(reversed(self.rows)))
        self.assertEqual(len(dataset.cases), 242)
        self.assertEqual(Counter(case.contract.value for case in dataset.cases),
                         {contract: count for _, _, contract, _, count in DISTRIBUTION})
        self.assertEqual([case.source_observation_id for case in dataset.cases],
                         [row["observation_id"] for row in self.selected])
        self.assertEqual(len({case.case_id for case in dataset.cases}), 242)
        for case in dataset.cases:
            self.assertEqual(case.case_id, "ovp34-" + case.source_observation_id)
            self.assertEqual(case.captured.source, Source.OVP34_REPLAY)
            self.assertEqual(case.captured.task, next(task for _, _, contract, task, _ in DISTRIBUTION
                                                    if contract == case.contract))
            self.assertIsNone(case.captured.historical_output)
            self.assertIsNone(case.captured.historical_generation)
        self.assertEqual(set(ReplayContract), {contract for _, _, contract, _, _ in DISTRIBUTION})
        # Identical messages represent distinct selected observations; never deduplicate by text.
        self.assertEqual(len({case.message_fingerprint for case in dataset.cases}), 1)

    def test_message_roundtrip_all_fields_and_no_mutation(self):
        original = copy.deepcopy(self.rows)
        dataset = self.load()
        plan = prepare_replay_plan(dataset, config=config(), resolved_model="candidate")
        for case, execution in zip(dataset.cases, plan, strict=True):
            self.assertEqual(canonical_json_bytes(execution.request.messages.to_messages()),
                             canonical_json_bytes(messages()))
            self.assertEqual(execution.request.case_id, case.case_id)
            self.assertNotIn("content", execution.request.messages.to_messages()[2])
            self.assertIsNone(execution.request.messages.to_messages()[3]["content"])
            self.assertIs(execution.request.messages, case.messages)
        self.assertEqual(self.rows, original)
        exported = dataset.cases[0].messages.to_messages()
        exported[4]["unknown"]["nested"].clear()
        self.assertEqual(dataset.cases[0].messages.to_messages(), messages())

    def test_frozen_models_privacy_and_capture_consistency(self):
        dataset = self.load()
        case = dataset.cases[0]
        for value in (dataset, case, case.captured):
            rendered = repr(value) + value.model_dump_json()
            for private in ("SYNTHETIC_PRIVATE_REQUEST", "SYNTHETIC_BASELINE", "synthetic_lookup"):
                self.assertNotIn(private, rendered)
        with self.assertRaises(ValidationError):
            case.contract = ReplayContract.QA_EVALUATION
        with self.assertRaises(ValidationError):
            dataset.cases = ()
        for update in ({"observation_id": "not-hex"}, {"case_id": "other"}, {"task": Task.QA},
                       {"historical_output": "SYNTHETIC_BASELINE"}, {"historical_generation": {}},
                       {"captured_message_fingerprint": "0" * 64}):
            with self.assertRaises(ValidationError):
                CapturedReplayCase(contract=case.contract, captured=case.captured.model_copy(update=update))
        with self.assertRaises(ValidationError):
            ReplayDataset(cases=(case, case), source_artifact_hash="a" * 64, selection_artifact_hash="b" * 64)

    def test_runtime_qualification_only_applied_to_runtime_summary(self):
        for case in self.load().cases:
            if case.contract == ReplayContract.RUNTIME_CONTEXT_SUMMARY:
                self.assertIn("after generation", case.captured.qualification)
                self.assertIn("not independently proven", case.captured.qualification)
            else:
                self.assertIsNone(case.captured.qualification)

    def test_semantic_identity_changes_with_selected_inputs_order_and_contract(self):
        original = self.load()
        changed = copy.deepcopy(self.rows)
        changed[0]["raw"]["input"]["messages"][0]["content"] += " changed"
        self.assertNotEqual(original.dataset_hash, self.load(rows=changed).dataset_hash)
        self.assertNotEqual(original.dataset_hash, self.load(selection=self.selected[::-1]).dataset_hash)
        # Swap extraction/QA mapping for two observations, retaining exact distribution.
        changed_rows, changed_selection = copy.deepcopy((self.rows, self.selected))
        left, right = 0, 118
        for field in ("name", "job_type"):
            changed_selection[left][field], changed_selection[right][field] = (
                changed_selection[right][field], changed_selection[left][field])
        changed_rows[left]["raw"]["name"] = changed_selection[left]["name"]
        changed_rows[right]["raw"]["name"] = changed_selection[right]["name"]
        self.assertNotEqual(original.dataset_hash, self.load(changed_rows, changed_selection).dataset_hash)
        changed_rows, changed_selection = copy.deepcopy((self.rows, self.selected))
        changed_rows[0]["raw"]["id"] = changed_rows[0]["index"]["observation_id"] = "ffffffffffffffff"
        changed_selection[0]["observation_id"] = "ffffffffffffffff"
        self.assertNotEqual(original.dataset_hash, self.load(changed_rows, changed_selection).dataset_hash)

    def test_semantic_identity_ignores_baseline_unselected_rows_and_formatting(self):
        original = self.load()
        variants = []
        baseline = copy.deepcopy(self.rows)
        baseline[0]["raw"]["output"] = {"changed": "SYNTHETIC_OTHER_BASELINE"}
        variants.append(baseline)
        unrelated = copy.deepcopy(self.rows)
        unrelated[-1]["raw"]["input"] = {"unrelated": "SYNTHETIC_UNRELATED"}
        variants.append(unrelated)
        variants.append(self.rows[::-1])
        for rows in variants:
            changed = self.load(rows=rows)
            self.assertEqual(original.dataset_hash, changed.dataset_hash)
            self.assertNotEqual(original.source_artifact_hash, changed.source_artifact_hash)
        changed = self.load(alternate=True)
        self.assertEqual(original.dataset_hash, changed.dataset_hash)
        self.assertNotEqual(original.source_artifact_hash, changed.source_artifact_hash)
        self.assertNotEqual(original.selection_artifact_hash, changed.selection_artifact_hash)

    def test_artifact_hashes_use_the_same_single_read_and_paths_not_identity(self):
        original = self.load()
        source_bytes, csv_bytes = self.source.read_bytes(), self.selection.read_bytes()
        with patch.object(Path, "read_bytes", side_effect=[csv_bytes, source_bytes]) as reading:
            moved = load_ovp34_replay(self.root / "other-source", selection_path=self.root / "other-csv")
        self.assertEqual(reading.call_count, 2)
        self.assertEqual(moved.dataset_hash, original.dataset_hash)
        self.assertEqual(moved.source_artifact_hash, hashlib.sha256(source_bytes).hexdigest())
        self.assertEqual(moved.selection_artifact_hash, hashlib.sha256(csv_bytes).hexdigest())
        self.assertNotIn(str(self.root), repr(moved) + moved.model_dump_json())

    def test_selection_rejections(self):
        for field, value in (("observation_id", "ABCDEF0123456789"), ("observation_id", "x"),
                             ("job_type", "qa_node_summary"), ("job_type", "post_call_qa"),
                             ("name", "llm"), ("trace_id", "foreign-trace")):
            selection = copy.deepcopy(self.selected)
            selection[0][field] = value
            self.assert_safe_failure(lambda: self.load(selection=selection))
        for selection in (self.selected[:-1], self.selected + [self.selected[0]], []):
            self.assert_safe_failure(lambda: self.load(selection=selection))
        foreign = copy.deepcopy(self.selected)
        foreign[0]["observation_id"] = "ffffffffffffffff"
        self.assert_safe_failure(lambda: self.load(selection=foreign), "missing_selected")
        self.assert_safe_failure(lambda: self.load(rows=self.rows[1:]), "missing_selected")
        self.assert_safe_failure(lambda: self.load(rows=self.rows + [self.rows[0]]))

    def test_csv_framing_columns_and_unicode(self):
        self.load()
        for data in (b"observation_id,observation_id,job_type,name,trace_id\n",
                     b"observation_id,job_type,name\n", b"\xff\n",
                     b'observation_id,job_type,name,trace_id\n"PRIVATE_MARKER',
                     b"observation_id,job_type,name,trace_id\na,b,c,d,extra\n",
                     b"observation_id,job_type,name,trace_id\n\n"):
            self.selection.write_bytes(data)
            self.assert_safe_failure(lambda: load_ovp34_replay(self.source, selection_path=self.selection))

    def test_raw_framing_json_and_unrelated_corruption(self):
        self.load()
        original = self.source.read_bytes()
        malformed = (b'{"PRIVATE_MARKER":', b'{"x":1,"x":2}\n', b'{}\n', b'[]\n', b'\xff\n', b'\n',
                     b'{"nested":{"value":NaN}}\n', b'{"x":Infinity}\n', b'{"x":-Infinity}\n',
                     b'{"x":1e999}\n')
        for tail in malformed:
            self.source.write_bytes(original + tail)
            self.assert_safe_failure(lambda: load_ovp34_replay(self.source, selection_path=self.selection))
        self.source.write_bytes(original.rstrip(b"\n"))
        self.assert_safe_failure(lambda: load_ovp34_replay(self.source, selection_path=self.selection))

    def test_selected_input_shapes_and_reference_consistency(self):
        for payload in ({}, {"messages": [], "tools": []}, {"messages": [], "tool_choice": "auto"},
                        {"messages": [], "unknown": 1}, {"messages": None}, {"messages": {}},
                        {"messages": [None]}, {"messages": ["PRIVATE_MARKER"]}):
            rows = copy.deepcopy(self.rows)
            rows[0]["raw"]["input"] = payload
            self.assert_safe_failure(lambda: self.load(rows=rows))
        for section, field, value in (("raw", "type", "SPAN"), ("raw", "name", "llm"),
                                      ("index", "observation_id", "bad"), ("index", "trace_id", "other")):
            rows = copy.deepcopy(self.rows)
            rows[0][section][field] = value
            self.assert_safe_failure(lambda: self.load(rows=rows))

    def test_candidate_settings_and_captured_path_only(self):
        dataset = self.load()
        before = dataset.dataset_hash
        tree = ast.parse(inspect.getsource(replay))
        imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any("contracts" in name or "prompts" in name or "adapter" in name for name in imports))
        with patch("ovp36_benchmark.requests.prepare_generated_request", side_effect=AssertionError("generated")), \
             patch("ovp36_benchmark.contracts.render_contract", side_effect=AssertionError("rendered")):
            first = prepare_replay_plan(dataset, config=config(), resolved_model="candidate-a")
            second = prepare_replay_plan(dataset, config=config(), resolved_model="candidate-b")
            third = prepare_replay_plan(dataset, config=config(generation=GenerationConfig(
                max_tokens=71, temperature=0.4, top_p=0.8, seed=3, token_limit_field="max_completion_tokens")),
                resolved_model="candidate-a")
        for a, b, c in zip(first, second, third, strict=True):
            self.assertEqual(a.request.model, "candidate-a")
            self.assertIsNotNone(a.request.captured)
            self.assertEqual(a.request.message_fingerprint, b.request.message_fingerprint)
            self.assertEqual(a.request.message_fingerprint, c.request.message_fingerprint)
            self.assertNotEqual(a.request.request_fingerprint, b.request.request_fingerprint)
            self.assertNotEqual(a.request.request_fingerprint, c.request.request_fingerprint)
            body = c.request.to_request_body()
            self.assertEqual((body["max_completion_tokens"], body["temperature"], body["top_p"], body["seed"]),
                             (71, 0.4, 0.8, 3))
            self.assertNotIn("max_tokens", body)
            self.assertIn("max_tokens", a.request.to_request_body())
            self.assertNotIn("max_completion_tokens", a.request.to_request_body())
        self.assertEqual(dataset.dataset_hash, before)

    def test_repetition_major_full_count_three_case_example_and_stability(self):
        dataset = self.load()
        for repetitions in (1, 2):
            plan = prepare_replay_plan(dataset, config=config(repetitions=repetitions), resolved_model="candidate")
            self.assertEqual(len(plan), 242 * repetitions)
            self.assertTrue(all(isinstance(item, PlannedExecution) for item in plan))
            expected = [(case.case_id, index) for index in range(repetitions) for case in dataset.cases]
            self.assertEqual([(item.request.case_id, item.repetition_index) for item in plan], expected)
            again = prepare_replay_plan(dataset, config=config(repetitions=repetitions), resolved_model="candidate")
            self.assertEqual(fingerprint_execution_plan([item.identity_projection() for item in plan]),
                             fingerprint_execution_plan([item.identity_projection() for item in again]))
            small = ReplayDataset(cases=dataset.cases[:3], source_artifact_hash=dataset.source_artifact_hash,
                                  selection_artifact_hash=dataset.selection_artifact_hash)
            small_plan = prepare_replay_plan(small, config=config(repetitions=repetitions), resolved_model="candidate")
            self.assertEqual([(item.request.case_id, item.repetition_index) for item in small_plan],
                             [(case.case_id, index) for index in range(repetitions) for case in small.cases])

    def test_expected_filesystem_and_preparation_errors_do_not_retain_private_inputs(self):
        with patch.object(Path, "read_bytes", side_effect=OSError("SYNTHETIC_PRIVATE_REQUEST")):
            self.assert_safe_failure(lambda: load_ovp34_replay(self.source, selection_path=self.selection))
        dataset = self.load()
        for cfg, model in ((config(), "../PRIVATE_MARKER"),
                           (config().model_copy(update={"repetitions": "PRIVATE_MARKER"}), "candidate"),
                           (config(endpoint=EndpointConfig(alias="synthetic", base_url="http://synthetic.invalid/v1",
                                                           model="literal-model")), "different-model")):
            self.assert_safe_failure(lambda: prepare_replay_plan(dataset, config=cfg, resolved_model=model))
        with patch.object(Path, "read_bytes", side_effect=RuntimeError("programming error")):
            with self.assertRaises(RuntimeError):
                load_ovp34_replay(self.source, selection_path=self.selection)


if __name__ == "__main__":
    unittest.main()
