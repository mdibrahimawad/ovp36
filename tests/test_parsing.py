import json
import math
import unittest

from ovp36_benchmark.schemas import ParseResult

from ovp36_benchmark.parsing import (
    _first_balanced, check_strict_json, parse_extraction, parse_production_json,
    parse_qa, parse_summary, parse_voicemail,
)


class JSONParsingTests(unittest.TestCase):
    def test_direct_objects_and_arrays(self):
        for raw in ('{"name":"Alex"}', '[1, "two"]', '{}', '[]', ' \n{"x":1}\t'):
            with self.subTest(raw=raw):
                result = parse_production_json(raw)
                self.assertEqual(result.parser_path, "direct")
                self.assertEqual(result.parsed_value, json.loads(raw))
                self.assertEqual(result.raw_response, raw)
                self.assertTrue(result.strict_json_valid)
                self.assertTrue(result.production_parse_success)
                self.assertIsNone(result.schema_valid)

    def test_scalar_syntax_separate_from_production(self):
        for raw, kind in [('"hello"', "string"), ('42', "number"), ('1.5', "number"),
                          ('true', "boolean"), ('false', "boolean"), ('null', "null")]:
            with self.subTest(raw=raw):
                strict = check_strict_json(raw)
                self.assertTrue(strict.valid)
                self.assertTrue(strict.format_valid)
                self.assertEqual(strict.top_level_type, kind)
                self.assertFalse(check_strict_json(raw, object_required=True).format_valid)
                result = parse_production_json(raw)
                self.assertEqual(result.parser_path, "raw_fallback")
                self.assertFalse(result.production_parse_success)
                self.assertEqual(result.parsed_value, {"raw": raw})
                self.assertIn("direct_scalar_rejected", [d.code for d in result.diagnostics])

    def test_empty_inputs(self):
        for raw in (None, "", " \t\n"):
            result = parse_production_json(raw)
            self.assertEqual(result.parsed_value, {})
            self.assertEqual(result.raw_response, raw)
            self.assertEqual(result.parser_path, "empty")
            self.assertFalse(result.strict_json_valid)
            self.assertFalse(result.production_usable)
            self.assertIn("blank_output", [d.code for d in result.diagnostics])

    def test_fenced_json_strict_failure_production_success(self):
        for raw, value in [('```json\n{"x":1}\n```', {"x": 1}),
                           ('```\n[1,2]\n```', [1, 2]),
                           ('Here:\n```json\n{"x":1}\n```\nDone.', {"x": 1}),
                           ('```{"x":1}```', {"x": 1})]:
            with self.subTest(raw=raw):
                result = parse_production_json(raw)
                self.assertEqual(result.parser_path, "fence")
                self.assertEqual(result.parsed_value, value)
                self.assertFalse(result.strict_json_valid)
                self.assertTrue(result.production_parse_success)
                self.assertIn("markdown_wrapping", [d.code for d in result.diagnostics])

    def test_first_fenced_block_only(self):
        first = '```json\n{"first":1}\n```\n```json\n{"second":2}\n```'
        self.assertEqual(parse_production_json(first).parsed_value, {"first": 1})
        malformed = '```json\n{"bad":}\n```\n```json\n{"second":2}\n```'
        self.assertEqual(parse_production_json(malformed).parser_path, "raw_fallback")
        # A later object can be the FIRST embedded object if the invalid fence
        # contained no object. This is object recovery, not a second fence attempt.
        later_object = '```text\ninvalid\n```\n```json\n{"second":2}\n```'
        self.assertEqual(parse_production_json(later_object).parser_path, "object")

    def test_fenced_scalars_rejected_then_object_array_recovery(self):
        cases = [
            ("A", '```\n42\n```', "raw_fallback", None),
            ("B", '```json\nnull\n```', "raw_fallback", None),
            ("C", '```json\n42\n```\n{"x":1}', "object", {"x": 1}),
            ("D", '```json\n42\n```\n[1,2]', "array", [1, 2]),
        ]
        for name, raw, path, value in cases:
            with self.subTest(case=name):
                result = parse_production_json(raw)
                self.assertEqual(result.parser_path, path)
                self.assertEqual(result.parsed_value, {"raw": raw} if path == "raw_fallback" else value)
                self.assertEqual(result.raw_response, raw)
                self.assertEqual(result.production_parse_success, path != "raw_fallback")
                self.assertEqual(result.production_usable, path != "raw_fallback")
                self.assertFalse(result.strict_json_valid)
                self.assertIn("first_fence_scalar_rejected", [d.code for d in result.diagnostics])

    def test_scalar_fence_does_not_trigger_second_fence_search(self):
        raw = '```json\n42\n```\n```json\n{"x":1}\n```'
        result = parse_production_json(raw)
        self.assertEqual(result.parser_path, "object")
        self.assertEqual(result.parsed_value, {"x": 1})

    def test_malformed_backslash_outside_string_preserves_production_boundary(self):
        for raw, opening, closing, candidate in [
            (r'prefix {\} later }', '{', '}', r'{\} later }'),
            (r'prefix [\] later ]', '[', ']', r'[\] later ]'),
        ]:
            with self.subTest(raw=raw):
                # The escaped first closing delimiter does not end the candidate,
                # even though the escape occurs outside a JSON string.
                start, end = _first_balanced(raw, opening, closing)
                self.assertEqual(raw[start:end], candidate)
                result = parse_production_json(raw)
                self.assertEqual(result.parser_path, "raw_fallback")
                self.assertEqual(result.parsed_value, {"raw": raw})

    def test_exact_case_sensitive_fence_regex(self):
        for raw, path, value in [('```json42```', 'raw_fallback', None),
                                 ('```json{"x":1}```', 'fence', {"x": 1}),
                                 ('```JSON\n{"x":1}\n```', 'object', {"x": 1}),
                                 ('```python\n42\n```', 'raw_fallback', None),
                                 ('```python\n{"x":1}\n```', 'object', {"x": 1}),
                                 ('```json\n[1,]\n```\n```json\n[2]\n```', 'raw_fallback', None)]:
            with self.subTest(raw=raw):
                result = parse_production_json(raw)
                self.assertEqual(result.parser_path, path)
                self.assertEqual(result.parsed_value, {"raw": raw} if path == 'raw_fallback' else value)

    def test_production_strips_unicode_whitespace_but_preserves_raw(self):
        raw = '\u00a0{"x":1}\u00a0'
        result = parse_production_json(raw)
        self.assertEqual(result.parser_path, "direct")
        self.assertEqual(result.parsed_value, {"x": 1})
        self.assertEqual(result.raw_response, raw)
        self.assertFalse(result.strict_json_valid)

    def test_embedded_json_and_string_delimiters(self):
        values = [{"nested": {"list": [1, {"x": "y"}]}},
                  {"text": 'braces { } and escaped "quote" and backslash \\'},
                  {"text": 'escaped backslash before quote \\" }'}]
        for value in values:
            raw = "Before: " + json.dumps(value) + " After."
            result = parse_production_json(raw)
            self.assertEqual(result.parser_path, "object")
            self.assertEqual(result.parsed_value, value)
            self.assertFalse(result.strict_json_valid)
            self.assertIn("leading_prose", [d.code for d in result.diagnostics])
            self.assertIn("trailing_prose", [d.code for d in result.diagnostics])
        result = parse_production_json('Before [1, "[brackets]"] After')
        self.assertEqual(result.parser_path, "array")
        self.assertEqual(result.parsed_value, [1, "[brackets]"])

    def test_first_balanced_candidate_not_any_valid_json(self):
        for raw in ('bad {"x":} later {"good":1}', 'unclosed { later {"good":1}',
                    'bad [1,] later [2]'):
            with self.subTest(raw=raw):
                self.assertEqual(parse_production_json(raw).parser_path, "raw_fallback")
        result = parse_production_json('bad {"x":} but [1,2]')
        self.assertEqual(result.parser_path, "array")
        self.assertEqual(result.parsed_value, [1, 2])
        # Object recovery precedes array recovery even when an array starts first.
        self.assertEqual(parse_production_json('prose [{"x":1}]').parsed_value, {"x": 1})

    def test_trailing_content_and_object_requirement(self):
        strict = check_strict_json('{"x":1} additional text')
        self.assertFalse(strict.valid)
        self.assertIn("trailing_prose", [d.code for d in strict.diagnostics])
        strict = check_strict_json('[1]', object_required=True)
        self.assertTrue(strict.valid)
        self.assertFalse(strict.format_valid)
        self.assertEqual(strict.top_level_type, "array")
        self.assertIn("object_required", [d.code for d in strict.diagnostics])

    def test_fallback_exact_and_no_repairs(self):
        for raw in ("  completely invalid\n", "{'single': 'quotes'}", '{"x":1,}'):
            with self.subTest(raw=raw):
                result = parse_production_json(raw)
                self.assertEqual(result.parsed_value, {"raw": raw})
                self.assertEqual(result.raw_response, raw)
                self.assertEqual(result.parser_path, "raw_fallback")
                self.assertFalse(result.production_usable)
                self.assertFalse(result.strict_json_valid)

    def test_numeric_representability_is_separate_from_json_syntax(self):
        result = parse_production_json('{"x":1e999}')
        self.assertTrue(result.strict_json_valid)
        self.assertEqual(result.parser_path, "direct")
        self.assertEqual(result.raw_response, '{"x":1e999}')
        self.assertEqual(result.parsed_value, {"x": float("inf")})
        self.assertTrue(result.production_usable)

    def test_nonfinite_production_values_and_result_round_trip(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            for template, path in [('{"x":%s}', "direct"), ('[%s]', "direct"),
                                   ('```json\n{"x":%s}\n```', "fence"),
                                   ('before {"x":%s} after', "object"),
                                   ('before [%s] after', "array")]:
                raw = template % token
                with self.subTest(raw=raw):
                    result = parse_production_json(raw)
                    self.assertEqual(result.parser_path, path)
                    self.assertEqual(result.raw_response, raw)
                    self.assertTrue(result.production_parse_success)
                    self.assertTrue(result.production_usable)
                    self.assertFalse(result.strict_json_valid)
                    serialized = result.model_dump_json()
                    self.assertIn(token, serialized)
                    restored = ParseResult.model_validate_json(serialized)
                    for parsed in (result, restored):
                        for value in (parsed.parsed_value, parsed.normalized_value):
                            number = value["x"] if isinstance(value, dict) else value[0]
                            if token == "NaN":
                                self.assertTrue(math.isnan(number))
                            else:
                                self.assertEqual(number, json.loads(token))

    def test_task_normalization_keeps_nonfinite_values(self):
        for parser in (parse_extraction, parse_qa):
            result = parser('{"x":NaN}')
            self.assertEqual(result.parser_path, "direct")
            self.assertTrue(result.production_usable)
            self.assertTrue(math.isnan(result.normalized_value["x"]))

    def test_no_type_coercion(self):
        result = parse_extraction('{"number":"1","boolean":"true"}')
        self.assertEqual(result.normalized_value, {"number": "1", "boolean": "true"})
        self.assertIsNone(result.schema_valid)


class TaskNormalizationTests(unittest.TestCase):
    def test_extraction_retains_value_and_fallback(self):
        for raw, usable in [('{}', True), ('[]', False), ('oops', False), ('', False)]:
            result = parse_extraction(raw)
            self.assertEqual(result.parsed_value, result.normalized_value)
            self.assertEqual(result.production_usable, usable)
        result = parse_extraction('{"raw":"an actual requested field"}')
        self.assertTrue(result.production_usable)
        self.assertEqual(result.parser_path, "direct")

    def test_qa_dict_preserved_without_defaults_or_schema_scoring(self):
        raw = '{"tags":"wrong type","custom":1}'
        result = parse_qa(raw)
        self.assertEqual(result.parsed_value, json.loads(raw))
        self.assertEqual(result.normalized_value, json.loads(raw))
        self.assertIsNone(result.schema_valid)

    def test_qa_non_dict_fallback_and_empty(self):
        for raw in ('[1,2]', '42', 'true', 'null', 'invalid', None, '', ' \n'):
            result = parse_qa(raw)
            self.assertEqual(result.normalized_value, {})
            self.assertFalse(result.production_usable)
        result = parse_qa('[1,2]')
        self.assertEqual(result.parsed_value, [1, 2])
        self.assertTrue(result.production_parse_success)
        self.assertIn("non_dict_qa_output", [d.code for d in result.diagnostics])
        result = parse_qa('invalid')
        self.assertEqual(result.parsed_value, {"raw": "invalid"})
        self.assertEqual(result.parser_path, "raw_fallback")

    def test_voicemail_decision_and_strict_label(self):
        rows = [('CONVERSATION', 'CONVERSATION', True), ('VOICEMAIL', 'VOICEMAIL', True),
                ('conversation', 'CONVERSATION', False), ('voicemail', 'VOICEMAIL', False),
                (' \nVOICEMAIL\t', 'VOICEMAIL', True), ('It is VOICEMAIL.', 'VOICEMAIL', False),
                ('VOICEMAIL then CONVERSATION', 'CONVERSATION', False),
                ('No label', None, False), ('NOTVOICEMAIL', 'VOICEMAIL', False),
                ('CONVERSATIONAL', 'CONVERSATION', False), (None, None, False), ('', None, False)]
        for raw, label, strict in rows:
            with self.subTest(raw=raw):
                result = parse_voicemail(raw)
                self.assertEqual(result.normalized_value, label)
                self.assertEqual(result.strict_format_valid, strict)
                self.assertEqual(result.raw_response, raw)
                self.assertEqual(result.production_parse_success, label is not None)
                self.assertIsNone(result.strict_json_valid)
                if label is None:
                    self.assertIn("undecided", [d.code for d in result.diagnostics])

    def test_summary_text_retained_without_json_parsing(self):
        for raw in ('  A summary.\n', '{"looks":"like JSON"}', None, '', ' \n'):
            result = parse_summary(raw)
            self.assertEqual(result.parsed_value, raw)
            self.assertEqual(result.normalized_value, raw)
            self.assertEqual(result.raw_response, raw)
            self.assertEqual(result.production_usable, raw is not None and bool(raw.strip()))
            self.assertIsNone(result.strict_json_valid)
            self.assertIsNone(result.schema_valid)

    def test_non_string_input_rejected(self):
        for parser in (check_strict_json, parse_production_json, parse_extraction, parse_qa,
                       parse_voicemail, parse_summary):
            with self.assertRaises(TypeError):
                parser({"not": "raw text"})


if __name__ == "__main__":
    unittest.main()
