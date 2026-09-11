import json
from pathlib import Path
import tempfile
import unittest

from helpers import case_data, parse_case
from ovp36_benchmark.dataset import (
    DatasetError, hash_dataset, load_cases, select_cases, validate_matrix_coverage,
)
from ovp36_benchmark.schemas import Difficulty, Source, Task


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic.jsonl"

    def write_cases(self, rows):
        self.path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def test_utf8_and_no_mutation(self):
        data = case_data()
        data["input"]["messages"][0]["content"] = "مرحبا Alex"
        self.write_cases([data])
        original = self.path.read_bytes()
        cases = load_cases(self.path)
        snapshot = cases[0].model_dump_json()
        self.assertIn("مرحبا", cases[0].input.messages[0].content)
        self.assertIs(select_cases(cases)[0], cases[0])
        hash_dataset(cases)
        validate_matrix_coverage(cases, [data["id"]])
        self.assertEqual(cases[0].model_dump_json(), snapshot)
        self.assertEqual(self.path.read_bytes(), original)

    def test_duplicate_ids_within_and_across_files(self):
        self.write_cases([case_data(), case_data()])
        with self.assertRaisesRegex(DatasetError, "duplicate case ID"):
            load_cases(self.path)
        self.write_cases([case_data()])
        with self.assertRaisesRegex(DatasetError, "duplicate case ID"):
            load_cases([self.path, self.path])

    def test_malformed_or_invalid_lines_not_skipped(self):
        for bad_line in ("{", "\n", "[]", '{"id":"a","id":"b"}',
                         '{"value":NaN}', json.dumps({**case_data(), "critical": "true"})):
            with self.subTest(bad_line=bad_line):
                self.path.write_text(json.dumps(case_data()) + "\n" + bad_line, encoding="utf-8")
                with self.assertRaisesRegex(DatasetError, r":2: invalid case"):
                    load_cases(self.path)

    def test_non_utf8_rejected(self):
        self.path.write_bytes(b"\xff")
        with self.assertRaisesRegex(DatasetError, "UTF-8"):
            load_cases(self.path)

    def test_overflowing_json_number_rejected(self):
        data = case_data("qa")
        data["input"]["metrics"] = {"duration": "OVERFLOW"}
        self.path.write_text(json.dumps(data).replace('"OVERFLOW"', '1e999'), encoding="utf-8")
        with self.assertRaises(DatasetError):
            load_cases(self.path)

    def test_hash_stable_across_layout_and_case_order(self):
        rows = [case_data(case_id="b"), case_data("voicemail", "a")]
        self.write_cases(rows)
        first = load_cases(self.path)
        self.path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in reversed(rows)), encoding="utf-8")
        second = load_cases(self.path)
        self.assertEqual(hash_dataset(first), hash_dataset(second))
        self.assertEqual(len(hash_dataset(first)), 64)
        rows[0]["notes"] = "A changed annotation"
        self.write_cases(rows)
        self.assertNotEqual(hash_dataset(first), hash_dataset(load_cases(self.path)))

    def test_hash_includes_message_order_gold_and_exercise(self):
        data = case_data()
        data["input"]["messages"].append({"role": "user", "content": "Another turn"})
        original = hash_dataset([parse_case(data)])
        data["input"]["messages"].reverse()
        self.assertNotEqual(original, hash_dataset([parse_case(data)]))
        data["input"]["messages"].reverse()
        data["expected"]["fields"]["name"]["acceptable_values"] = ["Different"]
        self.assertNotEqual(original, hash_dataset([parse_case(data)]))
        data["expected"]["fields"]["name"]["acceptable_values"] = ["Alex"]
        data["exercise"] = {"kind": "response_contract", "raw_output": "{}"}
        self.assertNotEqual(original, hash_dataset([parse_case(data)]))

    def test_filters(self):
        a = parse_case(case_data())
        b_data = case_data("voicemail", "second")
        b_data.update(source="adversarial", difficulty="hard", critical=False, tags=["synthetic", "edge"])
        b_data["exercise"] = {"kind": "response_contract", "raw_output": "both labels"}
        b = parse_case(b_data)
        cases = (a, b)
        self.assertEqual(select_cases(cases, tasks=[Task.EXTRACTION]), (a,))
        self.assertEqual(select_cases(cases, sources=[Source.ADVERSARIAL],
                                      difficulties=[Difficulty.HARD], critical=False,
                                      tags=["synthetic", "edge"]), (b,))
        self.assertEqual(select_cases(cases, exercise_kinds=["model"]), (a,))
        self.assertEqual(select_cases(cases, tasks=[]), ())
        self.assertEqual(select_cases(cases, tags=["missing"]), ())
        self.assertEqual(select_cases(cases, tasks=[Task.EXTRACTION], critical=False), ())
        for kwargs in ({"tasks": ["bad"]}, {"exercise_kinds": ["bad"]}, {"critical": "false"}):
            with self.assertRaises(ValueError):
                select_cases(cases, **kwargs)

    def test_coverage_explicit_and_partial(self):
        cases = (parse_case(case_data()),)
        self.assertTrue(validate_matrix_coverage(cases, ["synthetic-1"]).complete)
        coverage = validate_matrix_coverage(cases, ["not-created"], require_complete=False)
        self.assertEqual(coverage.missing_ids, ("not-created",))
        self.assertEqual(coverage.unexpected_ids, ("synthetic-1",))
        with self.assertRaises(DatasetError):
            validate_matrix_coverage(cases, ["not-created"])
        with self.assertRaises(DatasetError):
            hash_dataset(cases * 2)


if __name__ == "__main__":
    unittest.main()
