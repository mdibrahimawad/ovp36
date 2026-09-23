"""Curated candidate CLI composition; synthetic guarded or in-memory HTTP only."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from ovp36_benchmark import __main__ as cli
from ovp36_benchmark.adapters import prepare_curated_plan
from ovp36_benchmark.client import ModelClient
from ovp36_benchmark.config import load_config
from ovp36_benchmark.dataset import hash_dataset, load_curated_cases
from ovp36_benchmark.evaluation import CaseEvaluation, EvaluationError, evaluate_model_output, evaluation_fingerprint
from ovp36_benchmark.persistence import load_manifest, read_events
from ovp36_benchmark.reporting import aggregate_evaluations, evaluate_run
from ovp36_benchmark.test_server import DeterministicChatServer, StubExchange
from test_local_e2e import LoopbackGuard, MODEL, OUTPUTS, SMOKE_IDS, envelope


class CandidateSmokeCLITests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        self.guard = LoopbackGuard()
        self.guard.install(stack)
        clean = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI_")}
        stack.enter_context(patch.dict(os.environ, clean, clear=True))
        namespaces = ("openai", "httpx", "httpcore", "asyncio")
        names = set(namespaces) | {n for n in logging.root.manager.loggerDict
                                  if n.startswith(tuple(p + "." for p in namespaces))}
        for name in names:
            logger = logging.getLogger(name)
            stack.enter_context(patch.object(logger, "disabled", True))
            stack.callback(logger.setLevel, logger.level)
            logger.setLevel(logging.CRITICAL + 1)
        self.temporary = tempfile.TemporaryDirectory(prefix="SYNTHETIC_PRIVATE_PATH-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.results = self.root / "results"
        self.config_path = self.root / "candidate.local.toml"
        self.cases = load_curated_cases(cli.REPOSITORY_ROOT / "data/curated")

    def tearDown(self):
        self.assertEqual(self.guard.forbidden, [])
        self.temporary.cleanup()
        self.assertFalse(self.root.exists())

    def config(self, **updates):
        fields = dict(experiment_label="candidate-smoke-test", evaluation="exploratory",
                      repetitions=1, concurrency=1, timeout_seconds=30.0)
        endpoint = dict(alias="local-smoke", base_url_env="SMOKE_URL", model=MODEL, api_key_env="SMOKE_KEY")
        generation = dict(temperature=0.25, top_p=0.75, max_tokens=179,
                          token_limit_field="max_completion_tokens", seed=0)
        server = dict(runtime="synthetic", runtime_version="1", model_revision="test-revision")
        for name, value in updates.items():
            if name in endpoint:
                endpoint[name] = value
            elif name in generation:
                generation[name] = value
            else:
                fields[name] = value
        def lines(values):
            return "\n".join(f"{key} = {json.dumps(value)}" for key, value in values.items())
        self.config_path.write_text(lines(fields) + "\n[endpoint]\n" + lines(endpoint)
                                    + "\n[generation]\n" + lines(generation) + "\n[server]\n" + lines(server))
        return load_config(self.config_path)

    def invoke(self, arguments=None, *, url="http://127.0.0.1:12345/v1"):
        output, errors = io.StringIO(), io.StringIO()
        arguments = (["candidate-smoke", "--config", str(self.config_path),
                      "--results-root", str(self.results)] if arguments is None else arguments)
        with patch.dict(os.environ, {"SMOKE_URL": url, "SMOKE_KEY": "SYNTHETIC_PRIVATE_KEY"}), \
             redirect_stdout(output), redirect_stderr(errors):
            try:
                code = cli.main(arguments)
            except SystemExit as exit:
                code = exit.code
        return code, output.getvalue(), errors.getvalue()

    def assert_safe(self, output):
        for marker in ("SYNTHETIC_PRIVATE", str(self.root), "Authorization", "Bearer",
                       "http://127.0.0.1", "raw_content", "The caller requested"):
            self.assertNotIn(marker, output)

    def test_arguments_and_help_without_client(self):
        with patch.object(cli, "ModelClient") as client:
            for args in ([], ["candidate-smoke"], ["unknown", "--config", "SYNTHETIC_PRIVATE_ARG"],
                         ["candidate-smoke", "--config", "unused", "--unknown", "SYNTHETIC_PRIVATE_ARG"]):
                with self.subTest(args_count=len(args)):
                    code, output, errors = self.invoke(args)
                    self.assertEqual(code, 2)
                    self.assert_safe(output + errors)
            code, output, errors = self.invoke(["candidate-smoke", "--help"])
            self.assertEqual(code, 0)
            self.assertIn("candidate-smoke", output)
            self.assertEqual(cli._arguments(["candidate-smoke", "--config", "unused"]).results_root, "results")
            client.assert_not_called()

    def test_invalid_config_and_unexpected_failure_are_safe(self):
        with patch.object(cli, "ModelClient") as client:
            for text in ("SYNTHETIC_PRIVATE_BROKEN", "experiment_label = 'SYNTHETIC_PRIVATE_CONFIG'"):
                self.config_path.write_text(text)
                code, output, errors = self.invoke()
                self.assertEqual(code, 1)
                self.assertIn("invalid_configuration", errors)
                self.assert_safe(output + errors)
            self.config()
            with patch.object(cli, "load_curated_cases", side_effect=RuntimeError("SYNTHETIC_PRIVATE_BUG")):
                code, output, errors = self.invoke()
                self.assertEqual(code, 1)
                self.assertIn("orchestration_failed", errors)
                self.assert_safe(output + errors)
            client.assert_not_called()
        self.assertFalse(self.results.exists())

    def test_nonloopback_and_malformed_urls_fail_before_client(self):
        self.config()
        # LAN rejection is generated numerically so no private endpoint literal is tracked.
        lan = ".".join(map(str, (192, 168, 1, 10)))
        urls = ["http://localhost:8000/v1", "http://0.0.0.0:8000/v1", f"http://{lan}:8000/v1",
                "https://example.com/v1", "http://127.0.0.1:8000/v1?x=1",
                "http://user:pass@127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1#fragment",
                "http://127.0.0.1:8000/v1/chat/completions", "http://127.0.0.1/v1",
                "http://127.0.0.1:0/v1", "http://[::1]:8000/v1", "https://127.0.0.1:8000/v1"]
        with patch.object(cli, "ModelClient") as client:
            for url in urls:
                with self.subTest(index=urls.index(url)):
                    code, output, errors = self.invoke(url=url)
                    self.assertEqual(code, 1)
                    self.assert_safe(output + errors)
            client.assert_not_called()
        self.assertFalse(self.results.exists())

    def test_config_invariants_are_not_rewritten(self):
        with patch.object(cli, "ModelClient") as client:
            for updates, error in ((dict(repetitions=2), "one_repetition"),
                                   (dict(concurrency=2), "one_repetition"),
                                   (dict(evaluation="canonical"), "exploratory_designation"),
                                   (dict(experiment_label="unrelated"), "candidate_smoke_experiment_label"),
                                   (dict(model="SYNTHETIC_PRIVATE_PATH/model/file"), "orchestration_failed")):
                self.config(**updates)
                before = self.config_path.read_bytes()
                code, output, errors = self.invoke()
                self.assertEqual(code, 1)
                self.assertIn(error, errors)
                self.assert_safe(output + errors)
                self.assertEqual(self.config_path.read_bytes(), before)
            client.assert_not_called()

    def test_dataset_and_plan_fail_closed(self):
        self.config()
        variants = [self.cases[1:], self.cases + (self.cases[0],), self.cases[::-1]]
        with patch.object(cli, "ModelClient") as client:
            for cases in variants:
                with patch.object(cli, "load_curated_cases", return_value=cases):
                    self.assertEqual(self.invoke()[0], 1)
            with patch.object(cli, "hash_dataset", return_value="0" * 64):
                self.assertIn("frozen_dataset_mismatch", self.invoke()[2])
            selected = next(c for c in self.cases if c.id == SMOKE_IDS[0])
            control = next(c for c in self.cases if c.exercise.kind != "model")
            altered = tuple(c.model_copy(update={"exercise": control.exercise}) if c == selected else c for c in self.cases)
            with patch.object(cli, "load_curated_cases", return_value=altered), \
                 patch.object(cli, "hash_dataset", return_value=cli.FROZEN_DATASET_HASH):
                self.assertIn("six_model_cases_required", self.invoke()[2])
            with patch.object(cli, "prepare_curated_plan", return_value=()):
                self.assertIn("six_execution_plan_required", self.invoke()[2])
            client.assert_not_called()
        self.assertFalse(self.results.exists())

    def test_fingerprints_use_actual_documented_bytes(self):
        paths = sorted((cli.REPOSITORY_ROOT / "src/ovp36_benchmark").glob("*.py"))
        paths += [cli.REPOSITORY_ROOT / "src/ovp36_benchmark/prompts/contracts.json", cli.REPOSITORY_ROOT / "uv.lock"]
        components = {p.relative_to(cli.REPOSITORY_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in paths}
        expected = evaluation_fingerprint("implementation", components)
        self.assertEqual(cli._implementation_fingerprints(), (expected, components["uv.lock"]))
        original = Path.read_bytes
        for changed in (paths[0], paths[-2], paths[-1]):
            def read(path, changed=changed):
                return original(path) + (b"\nsynthetic-change" if path == changed else b"")
            with patch.object(Path, "read_bytes", read):
                self.assertNotEqual(cli._implementation_fingerprints()[0], expected)

    def test_results_root_default_and_private_location(self):
        self.config()
        with patch.object(cli, "ModelClient") as client:
            # Rejected paths are never created, including a lexical '..' escape.
            for path in (cli.REPOSITORY_ROOT / "unignored-smoke", cli.REPOSITORY_ROOT / "results/../unignored-smoke"):
                code, _, errors = self.invoke(["candidate-smoke", "--config", str(self.config_path), "--results-root", str(path)])
                self.assertEqual(code, 1)
                self.assertIn("results_root_must_be_ignored_or_external", errors)
            link = self.root / "linked"
            link.symlink_to(self.root, target_is_directory=True)
            code, _, errors = self.invoke(["candidate-smoke", "--config", str(self.config_path), "--results-root", str(link / "results")])
            self.assertEqual(code, 1)
            self.assertIn("results_root_symlink", errors)
            client.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "private journal requires POSIX")
    def test_cli_real_http_artifacts_purposes_and_resume(self):
        cfg = self.config()
        selected = tuple(c for c in self.cases if c.id in cli.SMOKE_CASE_IDS)
        self.assertEqual(tuple(c.id for c in selected), SMOKE_IDS)
        self.assertTrue(all(c.exercise.kind == "model" for c in selected))
        self.assertEqual(hash_dataset(self.cases), cli.FROZEN_DATASET_HASH)
        plan = prepare_curated_plan(selected, config=cfg, resolved_model=MODEL)
        responses = [envelope(output) for output in OUTPUTS]
        # Marker in response metadata is private and must not reach the console.
        responses[0]["id"] = "SYNTHETIC_PRIVATE_RESPONSE_ID"
        server = DeterministicChatServer([StubExchange(p.request, response)
                                         for p, response in zip(plan, responses, strict=True)])
        with server:
            self.guard.port = server._server.server_port
            url = server.base_url
            code, output, errors = self.invoke(url=url)
            self.assertEqual(code, 0, errors)
            self.assertEqual(errors, "")
            self.assert_safe(output)
            summary = json.loads(output)
            self.assertEqual((server.received_count, server.matched_count, server.responded_count), (6, 6, 6))
            self.assertEqual((summary["selected_case_count"], summary["execution"]["planned"],
                              summary["execution"]["attempts_sent"], summary["execution"]["executions_finalized"],
                              summary["evaluation_count"]), (6, 6, 6, 6, 6))
            self.assertEqual(summary["purpose"], "candidate_smoke")
            self.assertGreater(summary["pending_human_review_checks"], 0)
            self.assertEqual(set(summary["contracts"]), {"candidate_smoke:synthetic:" + c.contract_id for c in selected})
            run_id = summary["run_id"]
            manifest = load_manifest(self.results, run_id)
            implementation, lock = cli._implementation_fingerprints()
            self.assertEqual(manifest.dataset_hash, cli.FROZEN_DATASET_HASH)
            self.assertEqual(manifest.harness_code_fingerprint, implementation)
            self.assertEqual(manifest.dependency_lock_fingerprint, lock)
            self.assertEqual(manifest.safe_config.generation, cfg.generation)
            self.assertEqual(manifest.safe_config.requested_model, MODEL)
            self.assertEqual(manifest.safe_config.evaluation, "exploratory")
            self.assert_safe(manifest.model_dump_json())
            events = read_events(self.results, run_id)
            self.assertEqual([e.kind for e in events], ["attempt", "finalized"] * 6)
            self.assertTrue(all(e.attempt_index == 0 for e in events))
            self.assertEqual(tuple(e.execution_key.case_id for e in events[::2]), SMOKE_IDS)
            artifacts = list((self.results / run_id / "evaluations").glob("*/automatic.jsonl"))
            self.assertEqual(len(artifacts), 1)
            rows = tuple(CaseEvaluation.model_validate_json(line) for line in artifacts[0].read_bytes().splitlines())
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(r.purpose == "candidate_smoke" and r.evaluator_fingerprint == implementation for r in rows))
            self.assertFalse(list(self.results.rglob("reviews")))
            self.assertFalse((self.results / "control-evaluations").exists())

            # Same evidence has distinct purpose identities and separate reports.
            variants = [evaluate_run(self.cases, plan=plan, results_root=self.results, expected_manifest=manifest,
                        evaluator_fingerprint=implementation, purpose=purpose)
                        for purpose in ("candidate_smoke", "candidate", "functional_stub")]
            self.assertEqual(variants[0], rows)
            self.assertEqual(len({batch[0].evaluation_id for batch in variants}), 3)
            for purpose, batch in zip(("candidate_smoke", "candidate", "functional_stub"), variants, strict=True):
                self.assertTrue(all(name.startswith(purpose + ":") for name in aggregate_evaluations(batch)["contracts"]))
                self.assertEqual(batch[0].score, rows[0].score)
            with self.assertRaisesRegex(EvaluationError, "mixed_evaluation_purpose"):
                aggregate_evaluations((variants[0][0], variants[1][0]))
            direct = evaluate_model_output(selected[0], request=plan[0].request, attempt=events[0],
                dataset_hash=manifest.dataset_hash, evaluator_fingerprint=implementation, purpose="candidate_smoke")
            self.assertEqual(direct, rows[0])
            for purpose in ("unknown", "control", "candidate-smoke"):
                with self.assertRaises(EvaluationError):
                    evaluate_run(self.cases, plan=plan, results_root=self.results, expected_manifest=manifest,
                                 evaluator_fingerprint=implementation, purpose=purpose)
                with self.assertRaises(EvaluationError):
                    evaluate_model_output(selected[0], request=plan[0].request, attempt=events[0],
                        dataset_hash=manifest.dataset_hash, evaluator_fingerprint=implementation, purpose=purpose)
            with self.assertRaises(ValidationError):
                CaseEvaluation.model_validate_json(rows[0].model_dump_json().replace('"candidate_smoke"', '"unknown"'))

            before = {p: p.read_bytes() for p in self.results.rglob("*") if p.is_file()}
            code, output, errors = self.invoke(url=url)
            self.assertEqual(code, 0, errors)
            self.assertEqual(json.loads(output)["execution"]["skipped_completed"], 6)
            self.assertEqual(json.loads(output)["execution"]["attempts_sent"], 0)
            self.assertEqual(server.received_count, 6)
            self.assertEqual(before, {p: p.read_bytes() for p in self.results.rglob("*") if p.is_file()})
            journal_path = self.results / run_id / "journal.jsonl"
            with journal_path.open("ab") as stream:
                stream.write(b'{"SYNTHETIC_PRIVATE_CORRUPTION":')
            corrupt = journal_path.read_bytes()
            code, output, errors = self.invoke(url=url)
            self.assertEqual(code, 1)
            self.assert_safe(output + errors)
            self.assertEqual(journal_path.read_bytes(), corrupt)
            self.assertEqual(server.received_count, 6)
        self.assertFalse(server._thread.is_alive())
        self.assertEqual(server._server.socket.fileno(), -1)

    @unittest.skipUnless(os.name == "posix", "private journal requires POSIX")
    def test_handled_failure_and_bad_quality_exit_zero_without_retry(self):
        cfg = self.config()
        selected = tuple(c for c in self.cases if c.id in SMOKE_IDS)
        plan = prepare_curated_plan(selected, config=cfg, resolved_model=MODEL)
        responses = [envelope("SYNTHETIC_PRIVATE_BAD_OUTPUT")] + [envelope(output) for output in OUTPUTS[1:]]
        server = DeterministicChatServer([StubExchange(p.request, response, 500 if i == 1 else 200)
                                         for i, (p, response) in enumerate(zip(plan, responses, strict=True))])
        with server:
            self.guard.port = server._server.server_port
            code, output, errors = self.invoke(url=server.base_url)
            self.assertEqual(code, 0, errors)
            self.assert_safe(output + errors)
            summary = json.loads(output)
            self.assertEqual(summary["execution"]["failures"], 1)
            events = read_events(self.results, summary["run_id"])
            self.assertEqual(len(events), 12)
            self.assertEqual((events[2].result.status, events[2].result.retryable), ("http_error", True))
            self.assertEqual(server.received_count, 6)
            path, = (self.results / summary["run_id"] / "evaluations").glob("*/automatic.jsonl")
            rows = [CaseEvaluation.model_validate_json(line) for line in path.read_bytes().splitlines()]
            self.assertEqual(rows[1].score.status, "not_scored")
            self.assertTrue(any(c.status == "fail" for c in rows[0].checks))
            self.assertGreater(summary["pending_human_review_checks"], 0)
        self.assertFalse(server._thread.is_alive())

    def test_suite_inspection_does_not_construct_client_or_run(self):
        with patch.object(cli, "ModelClient") as client, patch.object(cli, "run_plan") as runner:
            args = cli._arguments(["candidate-suite", "--config", "unused"])
            self.assertEqual((args.command, args.results_root), ("candidate-suite", "results"))
            code, output, errors = self.invoke(["candidate-suite", "--help"])
            self.assertEqual(code, 0)
            self.assertIn("candidate-suite", output)
            self.assertEqual(errors, "")
            for flag in ("--source", "--selection"):
                self.assertEqual(self.invoke(["candidate-suite", "--config", "unused", flag, "private"])[0], 2)
            client.assert_not_called()
            runner.assert_not_called()
        self.assertFalse(self.results.exists())

    def test_suite_full_validation_and_hash_precede_plan(self):
        self.config()
        args = ["candidate-suite", "--config", str(self.config_path), "--results-root", str(self.results)]
        fixtures = self.root / "synthetic-fixture-copy"
        shutil.copytree(cli.REPOSITORY_ROOT / "data/curated", fixtures)
        control = next(c for c in self.cases if c.exercise.kind != "model")
        path = fixtures / (control.task.value + ".jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        # Corruption in a control must be rejected, even though it cannot be sent.
        next(r for r in rows if r["id"] == control.id)["unexpected_field"] = True
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.assertEqual(len(self.cases), 126)
        with patch.object(cli, "prepare_curated_plan") as planner, patch.object(cli, "ModelClient") as client:
            with patch.object(cli, "load_curated_cases", side_effect=lambda _: load_curated_cases(fixtures)):
                self.assertEqual(self.invoke(args)[0], 1)
            with patch.object(cli, "hash_dataset", return_value="0" * 64):
                code, _, errors = self.invoke(args)
                self.assertEqual(code, 1)
                self.assertIn("frozen_dataset_mismatch", errors)
            planner.assert_not_called()
            client.assert_not_called()
        self.assertFalse(self.results.exists())

    def test_suite_plan_requires_106_ordered_model_ids_before_client(self):
        cfg = self.config()
        args = ["candidate-suite", "--config", str(self.config_path), "--results-root", str(self.results)]
        plan = prepare_curated_plan(self.cases, config=cfg, resolved_model=MODEL)
        control = next(c for c in self.cases if c.exercise.kind != "model")
        injected = SimpleNamespace(request=SimpleNamespace(case_id=control.id), repetition_index=0)
        with patch.object(cli, "ModelClient") as client:
            for bad_plan in (plan[:-1], plan + (plan[0],), plan[::-1], (injected,) + plan[1:]):
                with patch.object(cli, "prepare_curated_plan", return_value=bad_plan):
                    code, _, errors = self.invoke(args)
                    self.assertEqual(code, 1)
                    self.assertIn("full_suite_execution_plan_required", errors)
            with patch.object(cli, "load_curated_cases", return_value=self.cases[:-1]), \
                 patch.object(cli, "hash_dataset", return_value=cli.FROZEN_DATASET_HASH):
                self.assertIn("full_suite_inventory_required", self.invoke(args)[2])
            client.assert_not_called()
        self.assertFalse(self.results.exists())

    def test_suite_mock_execution_controls_purpose_resume_and_six_case_smoke(self):
        cfg = self.config()
        config_bytes = self.config_path.read_bytes()
        model_cases = tuple(c for c in self.cases if c.exercise.kind == "model")
        control_ids = {c.id for c in self.cases if c.exercise.kind != "model"}
        plan = prepare_curated_plan(self.cases, config=cfg, resolved_model=MODEL)
        self.assertEqual((len(plan), len(control_ids)), (106, 20))
        received = []
        def response(request):
            received.append(json.loads(request.content))
            # Deliberately imperfect, entirely synthetic completion; no model.
            return httpx.Response(200, json=envelope("{}"))
        def client(endpoint, **kwargs):
            return ModelClient(endpoint, **kwargs, transport=httpx.MockTransport(response))
        args = ["candidate-suite", "--config", str(self.config_path), "--results-root", str(self.results)]
        with patch.object(cli, "ModelClient", side_effect=client), \
             patch.object(cli, "prepare_curated_plan", wraps=prepare_curated_plan) as planner, \
             patch.object(cli, "evaluate_control_case", wraps=cli.evaluate_control_case) as control_eval:
            code, output, errors = self.invoke(args)
            self.assertEqual((code, errors), (0, ""))
            self.assert_safe(output)
            self.assertEqual(planner.call_args.args[0], self.cases)
            self.assertEqual({call.args[0].id for call in control_eval.call_args_list}, control_ids)
            self.assertEqual(control_eval.call_count, 20)
            self.assertEqual(received, [p.request.to_request_body() for p in plan])
            summary = json.loads(output)
            self.assertEqual(summary["purpose"], "candidate")
            self.assertEqual((summary["selected_case_count"], summary["execution"]["planned"],
                              summary["execution"]["attempts_sent"], summary["evaluation_count"],
                              summary["control_evaluation_count"]), (106, 106, 106, 106, 20))
            self.assertGreater(summary["pending_human_review_checks"], 0)
            self.assertGreater(summary["pending_control_human_review_checks"], 0)
            self.assertEqual({k: v["selected"] for k, v in summary["controls"].items()},
                             dict(adapter_control=7, response_contract=12, transport_contract=1))
            expected_counts = dict(extraction=19, context_summary=21, voicemail=23, qa=27,
                                   qa_conversation_summary=10, qa_node_summary=6)
            self.assertEqual({k: v["selected"] for k, v in summary["contracts"].items()},
                             {"candidate:synthetic:" + k: v for k, v in expected_counts.items()})
            run_id = summary["run_id"]
            events = read_events(self.results, run_id)
            self.assertEqual(len(events), 212)
            self.assertEqual(tuple(e.execution_key.case_id for e in events[::2]), tuple(c.id for c in model_cases))
            self.assertTrue(control_ids.isdisjoint(e.execution_key.case_id for e in events))
            manifest = load_manifest(self.results, run_id)
            self.assertEqual(manifest.dataset_hash, cli.FROZEN_DATASET_HASH)
            self.assertEqual(manifest.safe_config.generation, cfg.generation)
            automatic, = (self.results / run_id / "evaluations").glob("*/automatic.jsonl")
            models = [CaseEvaluation.model_validate_json(line) for line in automatic.read_bytes().splitlines()]
            controls_path, = (self.results / "control-evaluations").glob("*/automatic.jsonl")
            controls = [CaseEvaluation.model_validate_json(line) for line in controls_path.read_bytes().splitlines()]
            self.assertEqual((len(models), len(controls)), (106, 20))
            self.assertTrue(all(r.purpose == "candidate" and r.exercise_kind == "model" for r in models))
            self.assertTrue(all(r.purpose == "control" and r.execution_key is None for r in controls))
            self.assertTrue(all(c.status == "pending" for row in models + controls
                                for c in row.checks if c.method == "human"))
            model_report = json.loads((automatic.parent / "summary-automatic.json").read_bytes())
            control_report = json.loads((controls_path.parent / "summary-automatic.json").read_bytes())
            self.assertEqual(sum(v["selected"] for v in model_report["contracts"].values()), 106)
            self.assertEqual(sum(v["selected"] for v in model_report["controls"].values()), 0)
            self.assertEqual(control_report["contracts"], {})

            before = {p: p.read_bytes() for p in self.results.rglob("*") if p.is_file()}
            code, output, errors = self.invoke(args)
            self.assertEqual((code, errors), (0, ""))
            self.assertEqual(json.loads(output)["execution"]["skipped_completed"], 106)
            self.assertEqual(json.loads(output)["execution"]["attempts_sent"], 0)
            self.assertEqual(len(received), 106)
            self.assertEqual(before, {p: p.read_bytes() for p in self.results.rglob("*") if p.is_file()})

            control_eval.reset_mock()
            code, output, errors = self.invoke()
            self.assertEqual((code, errors), (0, ""))
            smoke = json.loads(output)
            self.assertEqual((smoke["purpose"], smoke["execution"]["attempts_sent"]), ("candidate_smoke", 6))
            self.assertEqual(tuple(c.id for c in planner.call_args.args[0]), SMOKE_IDS)
            self.assertEqual(len(received), 112)
            self.assertNotIn("control_evaluation_count", smoke)
            control_eval.assert_not_called()
        self.assertEqual(self.config_path.read_bytes(), config_bytes)


if __name__ == "__main__":
    unittest.main()
