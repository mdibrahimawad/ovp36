import json
import unittest

from pydantic import TypeAdapter, ValidationError

from helpers import case_data, parse_case
from ovp36_benchmark.schemas import (
    BenchmarkError, EvidenceRef, ExerciseSpec, ParseResult, ScoreResult,
    SummaryExpected, TASK_MODELS, Task,
)


class SchemaTests(unittest.TestCase):
    def test_all_six_tasks_round_trip(self):
        for task in Task:
            with self.subTest(task=task):
                case = parse_case(case_data(task.value))
                self.assertIsInstance(case.input, TASK_MODELS[task][0])
                self.assertIsInstance(case.expected, TASK_MODELS[task][1])
                self.assertEqual(type(case).model_validate_json(case.model_dump_json()), case)

    def test_strict_envelope(self):
        for field, value in [
            ("id", ""), ("id", " \t"), ("id", 12), ("critical", "true"),
            ("critical", 1), ("source", "replay"), ("difficulty", "extreme"),
            ("task", "dialogue"), ("unexpected", "value"), ("schema_version", 1),
        ]:
            with self.subTest(field=field, value=value):
                data = case_data()
                data[field] = value
                with self.assertRaises(ValidationError):
                    parse_case(data)

    def test_invalid_input_combinations(self):
        for actual in Task:
            for other in Task:
                if actual == other:
                    continue
                with self.subTest(actual=actual, other=other):
                    data = case_data(actual.value)
                    data["input"] = case_data(other.value)["input"]
                    with self.assertRaises(ValidationError):
                        parse_case(data)

    def test_invalid_expected_combination(self):
        data = case_data()
        data["expected"] = case_data("voicemail")["expected"]
        with self.assertRaises(ValidationError):
            parse_case(data)

    def test_variable_types_and_known_values(self):
        for variable_type, value in [("string", "Alex"), ("number", 3.5), ("boolean", True)]:
            data = case_data()
            data["input"]["variables"][0]["type"] = variable_type
            data["expected"]["fields"]["name"]["acceptable_values"] = [value]
            self.assertEqual(parse_case(data).input.variables[0].type, variable_type)
        for variable_type in ["array", "object", "integer", "bool", 1]:
            data = case_data()
            data["input"]["variables"][0]["type"] = variable_type
            with self.assertRaises(ValidationError):
                parse_case(data)
        for variable_type, bad_value in [("boolean", "true"), ("number", True), ("string", 2)]:
            data = case_data()
            data["input"]["variables"][0]["type"] = variable_type
            data["expected"]["fields"]["name"]["acceptable_values"] = [bad_value]
            with self.assertRaises(ValidationError):
                parse_case(data)

    def test_extraction_names_and_gold_coverage(self):
        data = case_data()
        data["input"]["variables"] *= 2
        with self.assertRaises(ValidationError):
            parse_case(data)
        data = case_data()
        data["expected"]["fields"] = {}
        with self.assertRaises(ValidationError):
            parse_case(data)

    def test_missing_policy_is_explicit_and_per_case(self):
        for variable_type in ("number", "boolean", "string"):
            for policy in ({"allow_absent": True, "values": []},
                           {"allow_absent": False, "values": [None]},
                           {"allow_absent": False, "values": ["unknown"]}):
                data = case_data()
                data["input"]["variables"][0]["type"] = variable_type
                data["expected"]["fields"]["name"] = {"state": "missing", "missing_policy": policy}
                self.assertEqual(parse_case(data).expected.fields["name"].missing_policy.values,
                                 tuple(policy["values"]))
        data = case_data()
        data["expected"]["fields"]["name"] = {"state": "missing"}
        with self.assertRaises(ValidationError):
            parse_case(data)

    def test_none_expected_only_for_replay(self):
        for source in ("curated", "adversarial", "ovp34_replay"):
            data = case_data()
            data.update(source=source, expected=None)
            if source == "ovp34_replay":
                self.assertIsNone(parse_case(data).expected)
            else:
                with self.assertRaises(ValidationError):
                    parse_case(data)

    def test_exercises_explicit_and_separate(self):
        cases = [
            {"kind": "model"}, {"kind": "adapter_control"},
            {"kind": "response_contract", "raw_output": None},
            {"kind": "transport_contract", "response": "timeout"},
            {"kind": "transport_contract", "response": "http_error", "http_status": 503},
        ]
        for exercise in cases:
            data = case_data()
            data["exercise"] = exercise
            case = parse_case(data)
            self.assertEqual(case.is_model_quality_case, exercise["kind"] == "model")
        for exercise in ({}, {"kind": "other"}, {"kind": "response_contract"},
                         {"kind": "model", "raw_output": "injected"},
                         {"kind": "transport_contract", "response": "http_error"},
                         {"kind": "transport_contract", "response": "timeout", "http_status": 503}):
            with self.subTest(exercise=exercise), self.assertRaises(ValidationError):
                TypeAdapter(ExerciseSpec).validate_json(json.dumps(exercise))
        data = case_data()
        del data["exercise"]
        with self.assertRaises(ValidationError):
            parse_case(data)

    def test_contract_assertions_cannot_be_model_gold(self):
        data = case_data()
        data["expected"] = {"assertions": {"strict_format_valid": False}}
        with self.assertRaises(ValidationError):
            parse_case(data)
        data["exercise"] = {"kind": "response_contract", "raw_output": "not JSON"}
        self.assertFalse(parse_case(data).is_model_quality_case)

    def test_runtime_message_shapes_without_execution(self):
        data = case_data("context_summary")
        data["input"]["messages"] = [
            {"role": "system", "content": "Instructions"},
            {"role": "assistant", "content": [{"type": "text", "text": "Checking"}],
             "tool_calls": [{"id": "call-1", "type": "function",
                             "function": {"name": "lookup", "arguments": "{}"}}]},
            {"role": "tool", "content": "Result", "tool_call_id": "call-1"},
            {"kind": "llm_specific", "payload": {"opaque": True}},
        ]
        self.assertEqual(len(parse_case(data).input.messages), 4)
        data["input"]["is_realtime"] = "false"
        with self.assertRaises(ValidationError):
            parse_case(data)

    def test_qa_whole_call_and_score_constraints(self):
        data = case_data("qa")
        data["input"]["scope"] = "whole_call"
        with self.assertRaises(ValidationError):
            parse_case(data)
        data["input"]["node_summary"] = ""
        parse_case(data)
        for score in ([10, 1], [0, 5], [1, 11], ["1", 5]):
            data["expected"]["quality_score_range"] = score
            with self.assertRaises(ValidationError):
                parse_case(data)

    def test_opaque_json_numbers_must_be_finite(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            data = case_data("qa")
            data["input"]["metrics"] = {"nested": [{"value": value}]}
            with self.assertRaises(ValidationError):
                parse_case(data)


    def test_fact_and_evidence_references(self):
        for span in ({"start_char": 1}, {"start_char": 5, "end_char": 2},
                     {"start_char": -1, "end_char": 2}):
            with self.assertRaises(ValidationError):
                EvidenceRef(input_pointer="/transcript", **span)
        EvidenceRef(input_pointer="/transcript", start_char=0, end_char=5)
        with self.assertRaises(ValidationError):
            SummaryExpected.model_validate_json('{"required_facts":[],"critical_facts":["missing"]}')

    def test_frozen_models_and_tuple_sequences(self):
        case = parse_case(case_data())
        with self.assertRaises(ValidationError):
            case.id = "modified"
        with self.assertRaises(ValidationError):
            case.input.variables[0].type = "number"
        self.assertIsInstance(case.input.messages, tuple)

    def test_result_foundations_require_exercise_identity(self):
        data = {"case_id": "synthetic", "status": "complete", "metrics": {}}
        with self.assertRaises(ValidationError):
            ScoreResult.model_validate_json(json.dumps(data))
        data["exercise_kind"] = "response_contract"
        self.assertEqual(ScoreResult.model_validate_json(json.dumps(data)).exercise_kind,
                         "response_contract")
        error = BenchmarkError(stage="parser", code="raw_fallback", message="No structured output")
        result = ParseResult(strict_format_valid=False, strict_json_valid=False,
                             schema_valid=False, production_parse_success=False,
                             production_usable=False, parser_path="raw_fallback",
                             parsed_value={"raw": "x"}, normalized_value={}, diagnostics=(error,))
        self.assertFalse(result.production_usable)


if __name__ == "__main__":
    unittest.main()
