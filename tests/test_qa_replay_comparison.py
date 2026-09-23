"""Synthetic full replay + real SDK/runner/persistence with in-memory HTTP only."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from ovp36_benchmark import __main__ as cli, adapters, qa_replay_comparison as comparison
from ovp36_benchmark.client import ModelClient
from ovp36_benchmark.persistence import load_manifest
from ovp36_benchmark.replay import ReplayError, load_ovp34_replay
from test_replay import encoded, fixture, messages
from test_local_e2e import envelope


IDS = (
    "c276db54ca990a74", "31b24d3491a50db8", "34da3fd07076ab1e", "4162e59759efd5c8",
    "de5409211c79ea8c", "fd352d76033d559b", "c032b38b0cac1cca", "8afbd319f5e0b261",
    "81b63df3a246dd12", "9fbf1f50391d2a43",
)


def qa_output(index=0, **updates):
    return json.dumps(dict(tags=[dict(tag="ASSISTANT_IN_LOOP", reason="Synthetic reason.")],
        overall_sentiment="negative", call_quality_score=4, summary=f"Synthetic summary {index}.") | updates)


class QAReplayComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary).resolve()
        self.source, self.selection = self.root / "source.jsonl", self.root / "selection.csv"
        self.results = self.root / "results"
        self.rows, self.selections = fixture()
        qa_rows = [r for r in self.rows if r["raw"]["name"] == "qa-node-Synthetic"]
        qa_selections = [r for r in self.selections if r["job_type"] == "post_call_qa"]
        for i, (row, selected, observation_id) in enumerate(zip(qa_rows, qa_selections, IDS)):
            row["raw"].update(id=observation_id, model="qwen3.5")
            row["index"]["observation_id"] = observation_id
            selected["observation_id"] = observation_id
            row["raw"]["output"]["content"] = (qa_output(i) if i % 2 == 0
                                                    else " \n```json\n" + qa_output(i) + "\n```\n ")
        self.write_sources()
        # Only synthetic test hashes are substituted; production has no bypass.
        self.enterContext(patch.object(comparison, "CANONICAL_SOURCE_HASH", hashlib.sha256(self.source.read_bytes()).hexdigest()))
        self.enterContext(patch.object(comparison, "CANONICAL_SELECTION_HASH", hashlib.sha256(self.selection.read_bytes()).hexdigest()))
        self.enterContext(patch.dict(os.environ, {"QA_TEST_URL": "http://127.0.0.1:12345/v1",
                                                  "QA_TEST_MODEL": "synthetic-candidate"}, clear=True))
        self.enterContext(patch("socket.getaddrinfo", side_effect=AssertionError("No DNS allowed")))
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("No sockets allowed")))
        self.config = self.root / "candidate.local.toml"
        self.config.write_text('''experiment_label = "candidate-smoke-qa-test"
evaluation = "exploratory"
repetitions = 1
concurrency = 1
[endpoint]
alias = "synthetic"
base_url_env = "QA_TEST_URL"
model_env = "QA_TEST_MODEL"
[generation]
temperature = 0.25
top_p = 0.75
max_tokens = 512
token_limit_field = "max_completion_tokens"
seed = 0
''')

    def write_sources(self):
        # Raw order deliberately differs from CSV order: outputs must join by ID.
        raw, selection = encoded(list(reversed(self.rows)), self.selections)
        self.source.write_bytes(raw)
        self.selection.write_bytes(selection)

    def invoke(self, handler=None):
        sent = []
        def respond(request):
            sent.append(json.loads(request.content))
            return handler(len(sent)) if handler else httpx.Response(200, json=envelope(qa_output()))
        def client(endpoint, **kwargs):
            return ModelClient(endpoint, **kwargs, transport=httpx.MockTransport(respond))
        output, error = io.StringIO(), io.StringIO()
        with patch.object(cli, "ModelClient", side_effect=client) as construction, \
             redirect_stdout(output), redirect_stderr(error):
            code = cli.main(["qa-replay-compare", "--config", str(self.config), "--source", str(self.source),
                             "--selection", str(self.selection), "--results-root", str(self.results)])
        for private in (str(self.root), "Synthetic summary", "SYNTHETIC_PRIVATE_REQUEST", "http://127.0.0.1"):
            self.assertNotIn(private, output.getvalue() + error.getvalue())
        return code, output.getvalue(), error.getvalue(), sent, construction

    def records(self, result):
        run = self.results / result["run_id"]
        return [json.loads((run / "qa-comparison" / (key + ".json")).read_bytes()) for key in IDS]

    def test_full_validation_before_subsetting_including_unselected_rows(self):
        # Even malformed rows outside the ten IDs must fail the existing loader.
        original_rows, original_selection = copy.deepcopy((self.rows, self.selections))
        for mutate in (lambda: self.rows[0]["raw"].update(input={}),
                       lambda: self.rows[-1]["index"].update(trace_id="mismatched"),
                       lambda: self.selections.pop(0)):
            with self.subTest(mutation=mutate):
                self.rows, self.selections = copy.deepcopy((original_rows, original_selection))
                mutate()
                self.write_sources()
                with patch.object(comparison, "select_qa_cases") as subset:
                    with self.assertRaises(ReplayError):
                        comparison.load_qa_comparison_inputs(self.source, self.selection)
                    subset.assert_not_called()

    def test_canonical_checksum_gate_before_subset_or_client(self):
        # Semantically valid unrelated output changes still invalidate provenance.
        self.rows[0]["raw"]["output"]["content"] = "Synthetic changed output"
        self.write_sources()
        with patch.object(comparison, "select_qa_cases") as subset:
            code, _, _, sent, client = self.invoke()
            self.assertEqual(code, 1)
            self.assertEqual(sent, [])
            client.assert_not_called()
            subset.assert_not_called()
        self.assertFalse(self.results.exists())

    def test_exact_ten_selection_and_historical_join(self):
        full, selected, baselines = comparison.load_qa_comparison_inputs(self.source, self.selection)
        self.assertEqual(len(full.cases), 242)
        self.assertEqual(tuple(c.source_observation_id for c in selected.cases), IDS)
        self.assertEqual(comparison.QA_OBSERVATION_IDS, IDS)
        self.assertEqual(set(baselines), set(IDS))
        self.assertNotEqual(full.dataset_hash, selected.dataset_hash)
        for i, key in enumerate(IDS):
            raw = next(row["raw"] for row in self.rows if row["raw"]["id"] == key)
            self.assertEqual(baselines[key]["raw_output"], raw["output"]["content"])
            side = comparison._side(baselines[key]["raw_output"])
            self.assertEqual(side["parsed_result"]["summary"], f"Synthetic summary {i}.")
            self.assertEqual(side["parse_status"]["parser_path"], "direct" if i % 2 == 0 else "fence")
            self.assertEqual(side["parse_status"]["strict_format_valid"], i % 2 == 0)

    def test_reject_unknown_non_qa_and_duplicate_ids(self):
        dataset = load_ovp34_replay(self.source, selection_path=self.selection)
        for ids in (("f" * 16,), (dataset.cases[0].source_observation_id,), (IDS[0], IDS[0])):
            with self.subTest(ids=ids), self.assertRaises(ReplayError):
                comparison.select_qa_cases(dataset, ids)

    def test_join_rejects_source_change_and_missing_or_wrong_model_baseline(self):
        original_loader = comparison.load_ovp34_replay
        def change_after_validation(*args, **kwargs):
            dataset = original_loader(*args, **kwargs)
            self.source.write_bytes(self.source.read_bytes() + b"\n")
            return dataset
        with patch.object(comparison, "load_ovp34_replay", side_effect=change_after_validation):
            with self.assertRaisesRegex(ReplayError, "baseline_source_changed"):
                comparison.load_qa_comparison_inputs(self.source, self.selection)
        row = next(row["raw"] for row in self.rows if row["raw"]["id"] == IDS[0])
        for update in (dict(output=None), dict(output={"content": ""}),
                       dict(output={"content": qa_output()}, model="different-model")):
            row.update(update)
            self.write_sources()
            with patch.object(comparison, "CANONICAL_SOURCE_HASH", hashlib.sha256(self.source.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(ReplayError, "invalid_historical_qa_output"):
                    comparison.load_qa_comparison_inputs(self.source, self.selection)

    def test_sdk_runner_comparisons_permissions_resume_and_no_adapters(self):
        candidate = qa_output(tags=[dict(tag="ASSISTANT_IN_LOOP", reason="Synthetic loop."),
                                    dict(tag="USER_FRUSTRATED", reason="Synthetic frustration.")],
                              call_quality_score=7, overall_sentiment="positive")
        with patch.object(cli, "prepare_curated_plan", side_effect=AssertionError("No adapters")), \
             patch.object(adapters, "prepare_curated_plan", side_effect=AssertionError("No adapters")):
            code, output, error, sent, _ = self.invoke(lambda _: httpx.Response(200, json=envelope(candidate)))
        self.assertEqual((code, error), (0, ""))
        result = json.loads(output)
        self.assertEqual(len(sent), 10)
        for request in sent:
            self.assertEqual(request["messages"], messages())
            self.assertEqual(request["model"], "synthetic-candidate")
            self.assertEqual({k: v for k, v in request.items() if k not in ("model", "messages")},
                             dict(temperature=0.25, top_p=0.75, max_completion_tokens=512, seed=0, stream=False))
        full = load_ovp34_replay(self.source, selection_path=self.selection)
        self.assertEqual(load_manifest(self.results, result["run_id"]).dataset_hash, full.dataset_hash)
        records = self.records(result)
        for i, record in enumerate(records):
            self.assertEqual(record["observation_id"], IDS[i])
            self.assertEqual(record["qa_observation_name"], "qa-node-Synthetic")
            self.assertEqual(record["captured_request_messages"], messages())
            self.assertEqual(record["qwen"]["role"], "historical_baseline")
            self.assertEqual(record["qwen"]["model"], "qwen3.5")
            self.assertEqual(record["qwen"]["summary"], f"Synthetic summary {i}.")
            self.assertEqual(record["candidate"]["raw_output"], candidate)
            self.assertEqual(record["candidate_model"], "synthetic-candidate")
            self.assertEqual(record["differences"], dict(tags=dict(overlap=["ASSISTANT_IN_LOOP"],
                qwen_only=[], candidate_only=["USER_FRUSTRATED"]), sentiment_agreement=False, absolute_score_difference=3))
            self.assertEqual(record["transport"]["status"], "success")
        run = self.results / result["run_id"]
        before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
        for path in run.rglob("*"):
            self.assertEqual(path.stat().st_mode & 0o077, 0)
        self.assertNotIn("SYNTHETIC_PRIVATE_REQUEST", (run / "journal.jsonl").read_text())
        code, output, _, sent, _ = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(sent, [])
        self.assertEqual(json.loads(output)["execution"]["skipped_completed"], 10)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_errors_invalid_fields_and_nonfinite_outputs_stay_explicit(self):
        def response(index):
            if index == 1:
                return httpx.Response(503, json={"error": "Synthetic private failure"})
            content = {2: "not json", 3: qa_output(call_quality_score=float("nan")),
                       4: qa_output(tags=[], call_quality_score=True),
                       5: qa_output(tags="bad", overall_sentiment={}), 6: "[]"}.get(index, qa_output(tags=[]))
            body = envelope(content)
            if index == 2:
                body["choices"][0]["finish_reason"] = "length"
            return httpx.Response(200, json=body)
        code, output, error, sent, _ = self.invoke(response)
        self.assertEqual((code, error, len(sent)), (0, "", 10))
        records = self.records(json.loads(output))
        self.assertEqual(records[0]["transport"]["status"], "http_error")
        self.assertEqual(records[0]["transport"]["http_status"], 503)
        self.assertIsNone(records[0]["candidate"]["raw_output"])
        self.assertIsNone(records[0]["differences"]["absolute_score_difference"])
        self.assertEqual(records[1]["candidate"]["parse_status"]["parser_path"], "raw_fallback")
        self.assertEqual(records[1]["transport"]["response_metadata"]["finish_reason"], "length")
        self.assertEqual(records[2]["candidate"]["parse_status"]["parsed_result_encoding"], "nonfinite_parsed_result_omitted")
        self.assertIn("NaN", records[2]["candidate"]["raw_output"])
        self.assertEqual(records[2]["candidate"]["omitted_nonfinite_fields"], ["call_quality_score"])
        for index in (1, 2, 3, 5):
            self.assertIsNone(records[index]["differences"]["absolute_score_difference"])
        self.assertIsNone(records[4]["differences"]["tags"]["overlap"])
        self.assertIsNone(records[4]["differences"]["sentiment_agreement"])
        self.assertEqual(records[6]["differences"]["tags"]["qwen_only"], ["ASSISTANT_IN_LOOP"])

    def test_cli_requires_both_private_artifact_paths(self):
        for args in (["qa-replay-compare", "--config", "unused"],
                     ["qa-replay-compare", "--config", "unused", "--source", "private"],
                     ["candidate-smoke", "--config", "unused", "--selection", "private"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                cli._arguments(args)
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
