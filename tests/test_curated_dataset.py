"""Small synthetic unit fixtures, NOT the final 126 authored scenarios."""

import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from ovp36_benchmark.dataset import (
    DatasetError, QA_METRIC_FIELDS, curated_review_rows, hash_dataset,
    load_cases, load_curated_cases, validate_curated_cases,
)
from ovp36_benchmark.schemas import BenchmarkCase, ContextMessage


def fixture(task="extraction", *, ordinal=None, kind="model", difficulty=None, critical=True):
    """Deliberately tiny test-local payloads with reviewed matrix metadata."""
    prefix = dict(extraction="ex", context_summary="cs", voicemail="vm", qa="qa",
                  qa_conversation_summary="qs", qa_node_summary="ns")[task]
    ordinal = ordinal if ordinal is not None else (2 if task == "context_summary" else 1)
    difficulty = difficulty or ("normal" if task == "context_summary" else "easy")
    source = "adversarial" if difficulty == "adversarial" else "curated"
    inputs = {
        "extraction": {"messages": [{"role": "user", "content": "The color is blue."}],
                       "variables": [{"name": "color", "type": "string", "hint": "Stated color; null if absent."}]},
        "context_summary": {"messages": [{"role": "user", "content": "The color is blue."}]
                            + [{"role": "assistant", "content": "Acknowledged."}] * 6,
                            "context_compaction_enabled": True, "has_previous_node": True, "is_realtime": False},
        "voicemail": {"messages": [{"role": "user", "content": "Hello?"}]},
        "qa": {"scope": "node", "transcript": "[0.0s] user: Hello?\n[1.0s] assistant: Welcome.",
               "metrics": dict(zip(QA_METRIC_FIELDS, [None, 1, None, None, None])),
               "node_summary": "Greet the caller.", "previous_conversation_summary": ""},
        "qa_conversation_summary": {"prior_transcript": "[0.0s] user: The color is blue."},
        "qa_node_summary": {"node_type": "agent", "node_name": "Synthetic greeting",
                            "agent_prompt": "Greet the caller.", "custom_tools": [], "outgoing_edges": []},
    }
    pointers = dict(context_summary="/messages/0/content", qa_conversation_summary="/prior_transcript",
                    qa_node_summary="/agent_prompt")
    if task == "extraction":
        expected = {"fields": {"color": {"state": "known", "acceptable_values": ["blue"],
                                        "evidence": [{"input_pointer": "/messages/0/content"}]}}}
    elif task == "voicemail":
        expected = {"label": "CONVERSATION", "ambiguous": False, "rationale": "Synthetic human greeting."}
    elif task == "qa":
        expected = {"expected_tags": [], "expected_sentiment": "neutral", "quality_score_range": [8, 10],
                    "tag_evidence": {}, "summary_facts": [{"id": "greeting", "statement": "The assistant greets.",
                                                         "evidence": [{"input_pointer": "/transcript"}]}]}
    else:
        expected = {"required_facts": [{"id": "fact", "statement": "The stated fact or purpose.",
                                       "evidence": [{"input_pointer": pointers[task]}]}], "critical_facts": ["fact"]}
        if task != "context_summary":
            expected["sentence_range"] = [3, 5] if task == "qa_conversation_summary" else [2, 4]
    exercise = {"kind": kind}
    if kind != "model":
        expected = {"assertions": {"source_behavior": True}}
        if kind == "response_contract":
            exercise["raw_output"] = "synthetic supplied output"
        if kind == "transport_contract":
            exercise["response"] = "timeout"
    return {"id": f"{source}-{prefix}-{ordinal:03}", "task": task, "source": source,
            "difficulty": difficulty, "critical": critical, "contract_id": task,
            "exercise": exercise, "input": inputs[task], "expected": expected,
            "tags": ["synthetic", f"matrix:{prefix.upper()}-{ordinal:03}"],
            "notes": "Synthetic unit fixture, not an authored benchmark scenario."}


def case(data=None, **kwargs):
    return BenchmarkCase.model_validate_json(json.dumps(data if data is not None else fixture(**kwargs)))


class CuratedDatasetTests(unittest.TestCase):
    def test_six_contracts_validate_without_mutation(self):
        cases = tuple(case(task=task) for task in (
            "extraction", "context_summary", "voicemail", "qa", "qa_conversation_summary", "qa_node_summary"))
        before = [item.model_dump_json() for item in cases]
        self.assertEqual(validate_curated_cases(cases), cases)
        self.assertEqual(before, [item.model_dump_json() for item in cases])
        self.assertEqual(len(curated_review_rows(cases)), 6)

    def test_control_allocations_and_wrong_exercise_rejected(self):
        controls = [case(ordinal=13, kind="adapter_control", difficulty="normal", critical=False),
                    case(ordinal=24, kind="response_contract", difficulty="normal", critical=False),
                    case(task="context_summary", ordinal=25, kind="transport_contract", difficulty="adversarial", critical=False),
                    case(task="qa", ordinal=26, kind="response_contract", difficulty="adversarial")]
        self.assertEqual(validate_curated_cases(controls), tuple(controls))
        for item in controls:
            data = item.model_dump(mode="json")
            data["exercise"] = {"kind": "adapter_control" if item.exercise.kind != "adapter_control" else "response_contract",
                                **({"raw_output": "x"} if item.exercise.kind == "adapter_control" else {})}
            with self.assertRaises(DatasetError):
                validate_curated_cases([case(data)])

    def test_inventory_metadata_matches_authority_without_authoring_cases(self):
        from ovp36_benchmark.dataset import _SUITE, _CONTROLS
        matrix = (Path(__file__).resolve().parents[1] / "docs/CASE_MATRIX.md").read_text()
        self.assertEqual([len(row[2]) for row in _SUITE], [26, 26, 26, 32, 10, 6])
        self.assertEqual([len(ids) for ids in _CONTROLS.values()], [7, 12, 1])
        self.assertEqual(len(set.union(*_CONTROLS.values())), 20)
        for _, prefix, difficulties, noncritical in _SUITE:
            rows = re.findall(r"\| " + prefix.upper() + r"-(\d{3}) \| (easy|normal|hard|adversarial) \| (yes|no) \|", matrix)
            self.assertEqual("".join(d[0] for _, d, _ in rows), difficulties)
            self.assertEqual({int(i) for i, _, critical in rows if critical == "no"}, noncritical)

    def test_identity_source_contract_and_metadata_fail_closed(self):
        for changes in ({"id": "curated-EX-001"}, {"id": "curated-vm-001"}, {"id": "curated-ex-999"},
                        {"source": "ovp34_replay"}, {"contract_id": "missing"}, {"contract_id": "voicemail"},
                        {"critical": False}, {"difficulty": "hard"}, {"tags": ["synthetic"]},
                        {"provenance": {"source_ref": "synthetic-test", "historical_run_id": "historical"}},
                        {"provenance": {"source_ref": "/synthetic/path"}}):
            with self.subTest(changes=changes), self.assertRaises(DatasetError):
                validate_curated_cases([case(fixture() | changes)])

    def test_unchecked_schema_changes_revalidated(self):
        for changes in ({"critical": "yes"}, {"expected": None}, {"input": {}}, {"source": "bad"}):
            with self.assertRaises(DatasetError):
                validate_curated_cases([case().model_copy(update=changes)])

    def test_evidence_pointer_and_span_resolution(self):
        for pointer, offsets in [("/missing", {}), ("/messages/9/content", {}),
                                 ("/messages/00/content", {}), ("/messages/~2", {}),
                                 ("/messages/0/content", {"start_char": 0, "end_char": 100}),
                                 ("/messages", {"start_char": 0, "end_char": 1})]:
            data = fixture()
            data["expected"]["fields"]["color"]["evidence"] = [{"input_pointer": pointer, **offsets}]
            with self.subTest(pointer=pointer), self.assertRaises(DatasetError):
                validate_curated_cases([case(data)])
        data = fixture()
        data["expected"]["fields"]["color"]["evidence"] = [{"input_pointer": "/messages/0/content", "start_char": 13, "end_char": 17}]
        validate_curated_cases([case(data)])

    def test_required_gold_and_qa_vocabulary(self):
        for task in ("extraction", "context_summary", "qa", "qa_node_summary"):
            data = fixture(task)
            if task == "extraction":
                data["expected"]["fields"]["color"]["evidence"] = []
            elif task == "qa":
                data["expected"].update(expected_tags=["INVENTED_TAG"], tag_evidence={"INVENTED_TAG": [{"input_pointer": "/transcript"}]})
            else:
                data["expected"]["required_facts"] = []
                data["expected"]["critical_facts"] = []
            with self.assertRaises(DatasetError):
                validate_curated_cases([case(data)])

    def test_qa_metric_shape_and_values(self):
        for metrics in ({}, dict(zip(QA_METRIC_FIELDS, [None, True, None, None, None])),
                        dict(zip(QA_METRIC_FIELDS, [2, 1, None, None, None])),
                        dict(zip(QA_METRIC_FIELDS, [None, 1, "slow", None, None]))):
            data = fixture("qa")
            data["input"]["metrics"] = metrics
            with self.assertRaises(DatasetError):
                validate_curated_cases([case(data)])

    def test_order_duplicates_and_completeness(self):
        a, b = case(), case(task="voicemail")
        self.assertEqual(validate_curated_cases([a, b]), (a, b))
        for cases in ([b, a], [a, a]):
            with self.assertRaises(DatasetError):
                validate_curated_cases(cases)
        for cases in ([], [a]):
            with self.assertRaisesRegex(DatasetError, "incomplete"):
                validate_curated_cases(cases, require_complete=True)
        self.assertEqual(validate_curated_cases([]), ())
        with self.assertRaises(DatasetError):
            validate_curated_cases([], require_complete="false")

    def test_loader_explicit_file_order_and_no_partial_return(self):
        calls = []
        def read(path):
            calls.append(path.name)
            return ()
        with patch("ovp36_benchmark.dataset.load_cases", side_effect=read):
            with self.assertRaisesRegex(DatasetError, "incomplete"):
                load_curated_cases("synthetic-root")
        self.assertEqual(calls, [f"{task}.jsonl" for task in (
            "extraction", "context_summary", "voicemail", "qa", "qa_conversation_summary", "qa_node_summary")])
        with patch("ovp36_benchmark.dataset.load_cases", return_value=(case(task="voicemail"),)):
            with self.assertRaisesRegex(DatasetError, "file_task"):
                load_curated_cases("synthetic-root")

    def test_semantic_hash_and_public_repr_unchanged(self):
        a = case()
        data = a.model_dump(mode="json")
        reordered = BenchmarkCase.model_validate_json(json.dumps(data, sort_keys=True, indent=2))
        self.assertEqual(hash_dataset([a]), hash_dataset([reordered]))
        self.assertIn("The color is blue.", repr(a))
        for name, value in (("notes", "Different synthetic note"), ("critical", False)):
            self.assertNotEqual(hash_dataset([a]), hash_dataset([case(data | {name: value})]))
        data["expected"]["fields"]["color"]["acceptable_values"] = ["green"]
        self.assertNotEqual(hash_dataset([a]), hash_dataset([case(data)]))

    def test_review_projection_is_deterministic_bounded_and_contains_no_prompt(self):
        data = fixture()
        data["notes"] = "Synthetic scenario " * 100
        rows = curated_review_rows([case(data)])
        self.assertEqual(rows, curated_review_rows([case(data)]))
        self.assertEqual(rows[0]["matrix_id"], "EX-001")
        self.assertLessEqual(len(rows[0]["short_scenario"]), 160)
        self.assertNotIn("The color is blue.", json.dumps(rows))
        self.assertEqual(set(rows[0]), {"case_id", "matrix_id", "contract", "source", "exercise_kind",
                                       "difficulty", "critical", "short_scenario", "expected_annotation_summary"})

    def test_dataset_errors_drop_path_payload_cause_and_context(self):
        data = fixture()
        data["SYNTHETIC_REJECTED_FIELD"] = "SYNTHETIC_VALUE"
        for text in ("{", json.dumps(data), '{"a":1,"a":2}', '{"a":NaN}'):
            with patch.object(Path, "open", return_value=io.StringIO(text)):
                try:
                    load_cases("/synthetic-private-path/cases.jsonl")
                except DatasetError as exc:
                    self.assertIsNone(exc.__cause__)
                    self.assertIsNone(exc.__context__)
                    self.assertNotIn("synthetic-private-path", str(exc))
                    self.assertNotIn("SYNTHETIC", str(exc))
                else:
                    self.fail("malformed case accepted")

    def test_dataset_io_unicode_and_duplicate_errors_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unit.jsonl"
            path.write_text(json.dumps(fixture()) + "\n" + json.dumps(fixture()) + "\n")
            for action in (lambda: load_cases(path), lambda: load_cases(path / "missing")):
                try:
                    action()
                except DatasetError as exc:
                    self.assertIsNone(exc.__context__)
                    self.assertIsNone(exc.__cause__)
                    self.assertNotIn(directory, str(exc))
                    self.assertNotIn("curated-ex-001", str(exc))
                else:
                    self.fail("bad source accepted")
            path.write_bytes(b"\xff")
            try:
                load_cases(path)
            except DatasetError as exc:
                self.assertIsNone(exc.__context__)
                self.assertIn("UTF-8", str(exc))

    def test_curated_errors_drop_context(self):
        try:
            validate_curated_cases([case().model_copy(update={"critical": "synthetic-bad-value"})])
        except DatasetError as exc:
            self.assertIsNone(exc.__context__)
            self.assertIsNone(exc.__cause__)
            self.assertNotIn("synthetic-bad-value", str(exc))

    def test_developer_role_narrow_extension(self):
        message = ContextMessage(role="developer", content='{"type":"async_tool","status":"finished","tool_call_id":"test"}')
        self.assertEqual(message.role, "developer")
        for data in ({"role": "arbitrary", "content": "x"}, {"role": "developer"},
                     {"role": "developer", "content": "x", "unexpected": True}):
            with self.assertRaises(ValidationError):
                ContextMessage(**data)
