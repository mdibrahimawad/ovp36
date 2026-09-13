"""Infrastructure unit fixtures and the fixed, draft human-review inventory."""

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


class AuthoredDatasetTests(unittest.TestCase):
    """Freeze approved allocation; semantic judgments still need human review."""

    # Independent authoring-plan expectations, not imports of private dataset metadata.
    FAMILIES = (
        ("extraction", "ex", "eeennnhnnhhnnnnhhnnhaaannh", {9, 13, 14, 15, 16, 23, 24, 25}),
        ("context_summary", "cs", "ennnnnnhhhnnhhhhhaahnhhaah", {1, 11, 12, 17, 21, 25}),
        ("voicemail", "vm", "eeennhhheennnnnhhhhhaaanhh", {19, 20, 21, 22, 23, 24}),
        ("qa", "qa", "eeeeenennnehhhhhhhhhhhhnaannaahh", {24, 27, 28, 29, 30}),
        ("qa_conversation_summary", "qs", "ennhhhnhaa", {7}),
        ("qa_node_summary", "ns", "ennhha", set()),
    )
    CONTROLS = {
        "adapter_control": {"EX-013", "EX-014", "EX-015", "EX-016", "CS-001", "CS-021", "CS-023"},
        "response_contract": {"EX-024", "EX-025", "EX-026", "CS-024", "VM-022", "VM-023", "VM-024",
                              "QA-026", "QA-027", "QA-028", "QA-029", "QA-030"},
        "transport_contract": {"CS-025"},
    }

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1] / "data" / "curated"
        cls.cases = load_curated_cases(cls.root)
        cls.by_matrix = {next(t.removeprefix("matrix:") for t in c.tags if t.startswith("matrix:")): c
                         for c in cls.cases}

    def test_exact_inventory_ids_order_and_allocation(self):
        from collections import Counter
        expected_ids, expected_critical = [], set()
        for task, prefix, difficulties, noncritical in self.FAMILIES:
            for ordinal, difficulty in enumerate(difficulties, 1):
                source = "adversarial" if difficulty == "a" else "curated"
                case_id = f"{source}-{prefix}-{ordinal:03}"
                expected_ids.append(case_id)
                if ordinal not in noncritical:
                    expected_critical.add(case_id)
                actual = self.by_matrix[f"{prefix.upper()}-{ordinal:03}"]
                self.assertEqual(actual.contract_id, task)
                self.assertEqual(actual.difficulty.value, dict(e="easy", n="normal", h="hard", a="adversarial")[difficulty])
        self.assertEqual([c.id for c in self.cases], expected_ids)
        self.assertEqual(len(set(expected_ids)), 126)
        self.assertEqual([sum(c.task == task for c in self.cases) for task, *_ in self.FAMILIES], [26, 26, 26, 32, 10, 6])
        self.assertEqual(Counter(c.source.value for c in self.cases), {"curated": 109, "adversarial": 17})
        self.assertEqual(Counter(c.exercise.kind for c in self.cases),
                         {"model": 106, "adapter_control": 7, "response_contract": 12, "transport_contract": 1})
        self.assertEqual({c.id for c in self.cases if c.critical}, expected_critical)
        self.assertEqual([sum(c.critical and c.task == task for c in self.cases) for task, *_ in self.FAMILIES], [18, 20, 20, 27, 9, 6])
        self.assertEqual(sum(c.critical and c.is_model_quality_case for c in self.cases), 96)
        self.assertEqual({c.id for c in self.cases if c.critical and not c.is_model_quality_case},
                         {"curated-ex-026", "curated-cs-023", "adversarial-cs-024", "adversarial-qa-026"})
        for kind, ids in self.CONTROLS.items():
            self.assertEqual({mid for mid, c in self.by_matrix.items() if c.exercise.kind == kind}, ids)
        controls = set.union(*self.CONTROLS.values())
        self.assertTrue(all(c.exercise.kind == "model" for mid, c in self.by_matrix.items() if mid not in controls))

    def test_six_exact_files_and_synthetic_annotations(self):
        from ovp36_benchmark.schemas import validate_safe_identifier
        self.assertEqual({p.name for p in self.root.iterdir()}, {f"{task}.jsonl" for task, *_ in self.FAMILIES})
        for c in self.cases:
            with self.subTest(case=c.id):
                validate_safe_identifier(c.id)
                self.assertEqual(c.id, c.id.lower())
                self.assertIn("synthetic", c.tags)
                self.assertIsNone(c.provenance)
                self.assertTrue(c.notes)
                self.assertIsNotNone(c.expected)
        # Guard obvious accidental identifiers; this does not prove synthetic origin.
        text = "\n".join(p.read_text() for p in sorted(self.root.iterdir()))
        self.assertNotRegex(text, r"https?://|/Users/|/home/|\b\d{1,3}(?:\.\d{1,3}){3}\b")
        self.assertNotRegex(text, r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\bsk-[A-Za-z0-9]{12,}")

    def test_complete_validation_resolves_evidence_without_mutating_cases(self):
        before = [c.model_dump_json() for c in self.cases]
        self.assertEqual(validate_curated_cases(self.cases, require_complete=True), self.cases)
        self.assertEqual(before, [c.model_dump_json() for c in self.cases])

    def test_runtime_evidence_reaches_selected_model_input(self):
        from ovp36_benchmark.adapters import select_summary_context, format_summary_transcript, render_runtime_summary
        count = 0
        for c in self.cases:
            if c.task != "context_summary" or not c.is_model_quality_case:
                continue
            count += 1
            selected = select_summary_context(c.input)
            transcript = format_summary_transcript(selected.messages)
            self.assertTrue(transcript)
            for f in c.expected.required_facts + c.expected.superseded_facts:
                for evidence in f.evidence:
                    self.assertRegex(evidence.input_pointer, r"^/messages/[0-9]+/content$")
                    index = int(evidence.input_pointer.split("/")[2])
                    self.assertGreaterEqual(index, 1)
                    self.assertLessEqual(index, selected.last_summarized_index)
                    self.assertIn(c.input.messages[index].content, transcript)
            self.assertEqual(render_runtime_summary(c.input)[1]["content"], "Conversation history:\n" + transcript)
        self.assertEqual(count, 21)
        self.assertEqual(select_summary_context(self.by_matrix["CS-007"].input).last_summarized_index, 1)
        self.assertEqual(select_summary_context(self.by_matrix["CS-016"].input).last_summarized_index, 5)
        long = self.by_matrix["CS-020"]
        self.assertEqual(len(long.input.messages), 128)
        self.assertEqual(select_summary_context(long.input).last_summarized_index, 125)
        self.assertNotIn("...(truncated)", render_runtime_summary(long.input)[1]["content"])

    def test_authored_extraction_controls_and_typed_gold(self):
        from ovp36_benchmark.adapters import render_extraction
        for mid in ("EX-013", "EX-014", "EX-015"):
            c = self.by_matrix[mid]
            history = render_extraction(c.input)[1]["content"].split("Conversation history:\n", 1)[1]
            self.assertEqual(history, c.expected.assertions["formatted_history"])
        spelling = self.by_matrix["EX-015"].input.messages
        self.assertEqual(spelling[0].content.strip(), '{"status": "done"}')
        self.assertEqual(spelling[1].content, '{"status":"done"}')
        c = self.by_matrix["EX-016"]
        raw = c.input.messages[0].content
        self.assertGreater(len(raw), 2000)
        self.assertTrue(render_extraction(c.input)[1]["content"].endswith(
            raw[:2000] + "...(truncated)\n" + c.expected.assertions["following_message"]))
        self.assertIs(type(self.by_matrix["EX-018"].expected.fields["party_size"].acceptable_values[0]), int)
        self.assertIs(self.by_matrix["EX-019"].expected.fields["confirmed"].acceptable_values[0], False)
        for mid in ("EX-002", "EX-003", "EX-021"):
            for gold in self.by_matrix[mid].expected.fields.values():
                if gold.state == "missing":
                    self.assertEqual(gold.evidence, ())
                    self.assertEqual(gold.missing_policy.values, ("unknown",))
                    self.assertFalse(gold.missing_policy.allow_absent)

    def test_authored_summary_controls_preserve_current_context(self):
        from ovp36_benchmark.adapters import select_summary_context, render_runtime_summary, apply_context_summary
        self.assertIsNone(render_runtime_summary(self.by_matrix["CS-001"].input))
        c = self.by_matrix["CS-021"]
        selected = select_summary_context(c.input)
        self.assertEqual(selected.messages, c.input.messages[1:6])
        self.assertEqual(selected.last_summarized_index, 5)
        c = self.by_matrix["CS-023"]
        current = c.exercise.current_messages
        assertions = c.expected.assertions
        result = apply_context_summary(current, assertions["last_summarized_index"], assertions["summary_text"])
        self.assertEqual(result[0], current[0])
        self.assertNotEqual(current[0], c.input.messages[0])
        self.assertEqual(result[1].content, assertions["expected_summary_message"])
        self.assertEqual(result[2:], current[6:])
        self.assertEqual(len(result), 6)
        c = self.by_matrix["CS-024"]
        self.assertEqual(apply_context_summary(c.input.messages, 5, c.exercise.raw_output), c.input.messages)
        self.assertEqual(self.by_matrix["CS-025"].exercise.response, "timeout")

    def test_supplied_response_stimuli_match_parser_annotations(self):
        from ovp36_benchmark.parsing import parse_extraction, parse_summary, parse_voicemail, parse_qa
        parsers = dict(extraction=parse_extraction, context_summary=parse_summary, voicemail=parse_voicemail, qa=parse_qa)
        for c in self.cases:
            if c.exercise.kind != "response_contract" or c.id == "adversarial-qa-026":
                continue
            parsed = parsers[c.task.value](c.exercise.raw_output)
            assertions = c.expected.assertions
            for key in ("strict_json_valid", "strict_format_valid", "parser_path", "production_usable", "parsed_value", "normalized_value"):
                if key in assertions:
                    self.assertEqual(getattr(parsed, key), assertions[key], (c.id, key))
            if "decision" in assertions:
                self.assertEqual(parsed.normalized_value, assertions["decision"])
            if "source_defaults" in assertions:
                defaults = dict(tags=[], summary="", call_quality_score=None, overall_sentiment=None)
                for key, value in assertions["source_defaults"].items():
                    self.assertEqual(parsed.normalized_value.get(key, defaults[key]), value)

    def test_qa_invented_evidence_is_supplied_not_model_generated(self):
        from ovp36_benchmark.parsing import parse_qa
        c = self.by_matrix["QA-026"]
        result = parse_qa(c.exercise.raw_output)
        self.assertTrue(result.production_usable)
        self.assertIn("threatened to hang up", result.parsed_value["tags"][0]["reason"])
        self.assertEqual(result.parsed_value["tags"][0]["tag"], "USER_FRUSTRATED")
        self.assertEqual(result.parsed_value["overall_sentiment"], "negative")
        self.assertEqual(c.exercise.raw_output,
                         '{"tags":[{"tag":"USER_FRUSTRATED","reason":"The user threatened to hang up in anger."}],'
                         '"overall_sentiment":"negative","call_quality_score":2,"summary":"The user threatened to hang up."}')
        self.assertNotIn("hang up", c.input.transcript)
        self.assertEqual(c.input.transcript, self.by_matrix["QA-001"].input.transcript)
        self.assertIn("Thank you, that answers my question.", c.input.transcript)
        self.assertEqual(c.expected.expected_sentiment, "positive")
        self.assertEqual(c.expected.expected_tags, ())
        self.assertIn("USER_FRUSTRATED", c.expected.forbidden_tags)
        self.assertEqual(c.expected.quality_score_range, (9, 10))

    def test_qa_pair_relationships_change_only_intended_context(self):
        for first, second, field in (("QA-001", "QA-031", "node_summary"),
                                     ("QA-019", "QA-032", "previous_conversation_summary")):
            a, b = self.by_matrix[first], self.by_matrix[second]
            left, right = a.input.model_dump(), b.input.model_dump()
            self.assertNotEqual(left.pop(field), right.pop(field))
            self.assertEqual(left, right)
            self.assertNotEqual(a.expected.expected_tags, b.expected.expected_tags)
        a, b = self.by_matrix["QA-001"], self.by_matrix["QA-031"]
        self.assertEqual(a.expected.expected_sentiment, "positive")
        self.assertEqual(b.expected.expected_sentiment, a.expected.expected_sentiment)
        self.assertEqual(a.expected.expected_tags, ())
        self.assertEqual(b.expected.expected_tags, ("ASSISTANT_REPLY_IMPROPER",))
        self.assertEqual(b.expected.quality_score_range, (3, 5))

    def test_qa_distant_loop_repeats_an_answered_assistant_question(self):
        c = self.by_matrix["QA-021"]
        lines = c.input.transcript.splitlines()
        question_indices = [i for i, line in enumerate(lines)
                            if line.endswith("assistant: What date do you want?")]
        self.assertGreaterEqual(len(question_indices), 2)
        first, last = question_indices[0], question_indices[-1]
        self.assertEqual(first, 1)
        self.assertIn("user: I am ready to provide my date.", lines[first - 1])
        self.assertIn("user: My date is 2030-10-18.", lines[first + 1])
        self.assertIn("assistant: I have noted the date.", lines[first + 2])
        self.assertGreaterEqual(last - first, 40)  # Long intervening context, not adjacent repetition.
        self.assertIn("user: I already said 2030-10-18. This is frustrating.", lines[last + 1])
        self.assertEqual(len(lines), 48)
        self.assertEqual(c.input.metrics, dict(call_duration_seconds=None, num_turns=24,
                         avg_latency_seconds=0.5, avg_ttfb_seconds=0.2, max_latency_seconds=0.8))
        self.assertEqual(c.expected.expected_tags, ("ASSISTANT_IN_LOOP", "USER_FRUSTRATED"))
        self.assertEqual(c.expected.forbidden_tags, ("HEARING_ISSUES", "DEAD_AIR"))
        self.assertEqual(c.expected.expected_sentiment, "negative")
        self.assertEqual(c.expected.quality_score_range, (3, 5))

    def test_qa_timestamps_and_precomputed_metrics_are_compatible(self):
        for c in self.cases:
            if c.task != "qa":
                continue
            rows = re.findall(r"^\[([0-9.]+)s\] (user|assistant):", c.input.transcript, re.MULTILINE)
            gaps = []
            self.assertEqual(len(rows), 2 * c.input.metrics["num_turns"])
            for index in range(0, len(rows), 2):
                self.assertEqual((rows[index][1], rows[index + 1][1]), ("user", "assistant"))
                gaps.append(float(rows[index + 1][0]) - float(rows[index][0]))
            self.assertAlmostEqual(sum(gaps) / len(gaps), c.input.metrics["avg_latency_seconds"])
            self.assertAlmostEqual(max(gaps), c.input.metrics["max_latency_seconds"])
            duration = c.input.metrics["call_duration_seconds"]
            if duration is not None:
                self.assertGreaterEqual(duration, float(rows[-1][0]))
        for mid in ("QA-008", "QA-023"):
            self.assertEqual(self.by_matrix[mid].input.metrics["max_latency_seconds"], 24.0)

    def test_ambiguous_voicemail_and_qa_annotations_require_human_review(self):
        from ovp36_benchmark.schemas import QAEvaluationExpected
        for i, label, ambiguous in ((17, "VOICEMAIL", False), (18, "CONVERSATION", True),
                                    (19, "VOICEMAIL", True), (20, "CONVERSATION", True), (21, None, True)):
            c = self.by_matrix[f"VM-{i:03}"]
            self.assertIn("human-review-required", c.tags)
            self.assertEqual(c.expected.label, label)
            self.assertEqual(c.expected.ambiguous, ambiguous)
            self.assertEqual(c.exercise.kind, "model")
        self.assertEqual(self.by_matrix["VM-021"].input.messages[0].content, "")
        for c in self.cases:
            if isinstance(c.expected, QAEvaluationExpected):
                self.assertIn("human-review-required", c.tags)
                self.assertIn("Draft QA", c.notes)
        self.assertIn("USER_NOT_UNDERSTANDING", self.by_matrix["QA-016"].expected.expected_tags)

    def test_summary_gold_is_factual_and_optional_behavior_stays_optional(self):
        from ovp36_benchmark.schemas import SummaryExpected
        for c in self.cases:
            if isinstance(c.expected, SummaryExpected):
                self.assertTrue(c.expected.required_facts)
                if c.task == "qa_conversation_summary":
                    self.assertEqual(c.expected.sentence_range, (3, 5))
                if c.task == "qa_node_summary":
                    self.assertEqual(c.expected.sentence_range, (2, 4))
        c = self.by_matrix["NS-006"]
        facts = {f.id: f for f in c.expected.required_facts}
        self.assertEqual(c.expected.optional_behaviors, ("optional-offer", "optional-lookup"))
        for name in c.expected.optional_behaviors:
            self.assertIn("optional", facts[name].statement)
            self.assertIn("conditional", facts[name].statement)
        self.assertIn("confirmation-required", c.expected.key_behaviors)
        self.assertIn("required", facts["confirmation-required"].statement)

    def test_hash_and_all_126_review_rows_are_stable(self):
        again = load_curated_cases(self.root)
        self.assertEqual(hash_dataset(self.cases), hash_dataset(again))
        reordered_keys = tuple(BenchmarkCase.model_validate_json(json.dumps(c.model_dump(mode="json"), sort_keys=True)) for c in self.cases)
        self.assertEqual(hash_dataset(self.cases), hash_dataset(reordered_keys))
        rows = curated_review_rows(self.cases)
        self.assertEqual(len(rows), 126)
        self.assertEqual(rows, curated_review_rows(again))
        self.assertEqual([r["case_id"] for r in rows], [c.id for c in self.cases])
        for row in rows:
            self.assertEqual(set(row), {"case_id", "matrix_id", "contract", "source", "exercise_kind",
                                       "difficulty", "critical", "short_scenario", "expected_annotation_summary"})
            self.assertLessEqual(len(row["short_scenario"]), 160)

    def test_full_generated_plan_counts_order_and_no_execution(self):
        from contextlib import ExitStack
        from ovp36_benchmark.adapters import prepare_curated_plan
        from ovp36_benchmark.requests import prepare_generated_request
        from ovp36_benchmark.schemas import RunConfig
        model_ids = [c.id for c in self.cases if c.is_model_quality_case]
        for repetitions in (1, 2):
            config = RunConfig.model_validate_json(json.dumps({
                "endpoint": {"alias": "synthetic", "base_url_env": "UNUSED_SYNTHETIC_ENDPOINT", "model": "synthetic-model"},
                "generation": {"temperature": 0.0, "max_tokens": 71, "top_p": 1.0, "seed": 7},
                "repetitions": repetitions, "experiment_label": "synthetic-preparation", "evaluation": "exploratory"}))
            with ExitStack() as stack:
                for target in ("ovp36_benchmark.requests.prepare_captured_request", "ovp36_benchmark.requests.CapturedReplayRequest",
                               "ovp36_benchmark.client.ModelClient", "ovp36_benchmark.runner.run_plan",
                               "ovp36_benchmark.persistence.ResultJournal.open", "socket.socket.connect", "socket.getaddrinfo"):
                    stack.enter_context(patch(target, side_effect=AssertionError("execution or replay forbidden")))
                generated = stack.enter_context(patch("ovp36_benchmark.adapters.prepare_generated_request", wraps=prepare_generated_request))
                plan = prepare_curated_plan(self.cases, config=config, resolved_model="synthetic-model")
            self.assertEqual(generated.call_count, 106)
            self.assertEqual(len(plan), 106 * repetitions)
            self.assertEqual([(p.request.case_id, p.repetition_index) for p in plan],
                             [(case_id, rep) for rep in range(repetitions) for case_id in model_ids])
            for item in plan:
                self.assertIsNone(item.request.captured)
                self.assertEqual(item.request.generation, config.generation)
