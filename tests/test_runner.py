"""Runner orchestration using synthetic requests, real private journals, and no network."""

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from pydantic import ValidationError

from ovp36_benchmark.client import AttemptResult, ModelClient, ModelResponse
from ovp36_benchmark.config import ResolvedEndpoint
from ovp36_benchmark.identity import ExecutionKey, fingerprint_execution_plan
from ovp36_benchmark.persistence import (
    AttemptRecorded, JournalCorruptionError, PersistenceError, ResultJournal, build_manifest,
)
from ovp36_benchmark.requests import (
    CapturedReplayRequest, MessageSnapshot, prepare_captured_request, prepare_generated_request,
)
from ovp36_benchmark.runner import (
    MAX_ATTEMPTS, PlannedExecution, RunnerValidationError, RunSummary, run_plan,
)
from ovp36_benchmark.schemas import EndpointConfig, GenerationConfig, RunConfig, ServerMetadata, Source, Task


def execution(case_id="case-a", repetition_index=0, **updates):
    request = prepare_generated_request(
        [{"role": "user", "content": "SYNTHETIC_PRIVATE_REQUEST"}],
        **(dict(case_id=case_id, task=Task.QA, source=Source.CURATED, model="synthetic-model",
                generation=GenerationConfig(max_tokens=137)) | updates),
    )
    return PlannedExecution(request=request, repetition_index=repetition_index)


def manifest(plan, *, plan_hash=None, **config_updates):
    config = RunConfig(**(dict(
        endpoint=EndpointConfig(alias="synthetic", base_url="http://synthetic.invalid/v1",
                                model="synthetic-model"),
        generation=GenerationConfig(max_tokens=137), experiment_label="synthetic",
        evaluation="canonical", server=ServerMetadata(model_revision="revision-a"),
    ) | config_updates))
    return build_manifest(
        config, resolved_model="synthetic-model", dataset_hash="a" * 64,
        ordered_execution_plan_hash=plan_hash or fingerprint_execution_plan(
            [item.identity_projection() for item in plan]),
        contract_hashes={"qa": "b" * 64}, harness_code_fingerprint="c" * 64,
        dependency_lock_fingerprint="d" * 64,
    )


def result(**updates):
    return AttemptResult(**(dict(
        status="success", latency_ms=1.25,
        response=ModelResponse(raw_content="SYNTHETIC_PRIVATE_CANDIDATE", content_present=True,
                               finish_reason="stop", response_model="synthetic-model", usage=None),
    ) | updates))


def key(item, expected):
    return ExecutionKey(run_id=expected.run_id, **item.identity_projection())


class FakeClient:
    def __init__(self, *outcomes):
        self.send_once = AsyncMock(side_effect=outcomes)
        self.aclose = AsyncMock()


@unittest.skipUnless(os.name == "posix", "local POSIX persistence target")
class RunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "results"
        self.plan = (execution(), execution("case-b"))
        self.expected = manifest(self.plan)
        for target in ("socket.getaddrinfo", "socket.socket.connect", "socket.socket.connect_ex"):
            guard = patch(target, side_effect=AssertionError("real network forbidden"))
            mock = guard.start()
            self.addCleanup(mock.assert_not_called)
            self.addCleanup(guard.stop)

    async def run_with(self, client, plan=None, expected=None):
        return await run_plan(self.plan if plan is None else plan, client=client,
                              results_root=self.root, expected_manifest=expected or self.expected)

    def seed(self, item, results, *, finalized=False, expected=None):
        expected = expected or self.expected
        with ResultJournal.open(self.root, expected) as journal:
            for index, value in enumerate(results):
                journal.append_attempt(key(item, expected), index, value)
            if finalized:
                journal.finalize(key(item, expected), len(results) - 1)

    def records(self, expected=None):
        with ResultJournal.open(self.root, expected or self.expected) as journal:
            return journal.records(), journal.completed_keys()

    async def assert_preflight_failure(self, plan, expected, message):
        client = FakeClient()
        with patch.object(ResultJournal, "open") as opening:
            with self.assertRaisesRegex(RunnerValidationError, message) as caught:
                await self.run_with(client, plan, expected)
        opening.assert_not_called()
        client.send_once.assert_not_called()
        client.aclose.assert_not_called()
        self.assertFalse(self.root.exists())
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn("PRIVATE", repr(caught.exception))

    def test_strict_frozen_execution_and_summary(self):
        item = self.plan[0]
        self.assertIs(PlannedExecution(request=item.request, repetition_index=0).request, item.request)
        self.assertEqual(set(item.identity_projection()),
                         {"case_id", "request_fingerprint", "repetition_index"})
        self.assertNotIn("SYNTHETIC_PRIVATE_REQUEST", repr(item))
        self.assertNotIn("request", item.model_dump())
        for value in (True, "0", -1, 0.0):
            with self.assertRaises(ValidationError):
                PlannedExecution(request=item.request, repetition_index=value)
        with self.assertRaises(ValidationError):
            PlannedExecution(request={}, repetition_index=0)
        with self.assertRaises(ValidationError):
            item.repetition_index = 1
        fields = {name: 0 for name in ("planned", "skipped_completed", "resumed_unfinished",
                  "attempts_sent", "executions_finalized", "successes", "failures")}
        summary = RunSummary(**fields)
        self.assertEqual(summary.model_dump(), fields)
        for name in fields:
            for value in (True, "0", -1, 0.0):
                with self.assertRaises(ValidationError):
                    RunSummary(**(fields | {name: value}))
        for change in ({"planned": 1}, {"successes": 1}, {"extra": 0}):
            with self.assertRaises(ValidationError):
                RunSummary(**(fields | change))
        with self.assertRaises(ValidationError):
            summary.planned = 1

    async def test_fresh_statuses_always_one_attempt_and_finalized(self):
        self.assertEqual(MAX_ATTEMPTS, 1)
        variants = [dict(status=status, retryable=retryable, http_status=http_status)
                    for status, retryable, http_status in (
                        ("success", False, None), ("success", True, 503),
                        ("timeout", True, None), ("connection_error", True, None),
                        ("http_error", True, 429), ("http_error", True, 503),
                        ("http_error", False, 400), ("protocol_error", False, None),
                        ("protocol_error", True, None))]
        for index, update in enumerate(variants):
            with self.subTest(update=update):
                plan = (execution(f"status-{index}"),)
                expected = manifest(plan)
                evidence = result(**update)
                client = FakeClient(evidence)
                summary = await self.run_with(client, plan, expected)
                client.send_once.assert_awaited_once_with(plan[0].request)
                client.aclose.assert_not_called()
                self.assertEqual(summary.model_dump(), dict(
                    planned=1, skipped_completed=0, resumed_unfinished=0, attempts_sent=1,
                    executions_finalized=1, successes=int(evidence.status == "success"),
                    failures=int(evidence.status != "success")))
                records, completed = self.records(expected)
                self.assertEqual([event.kind for event in records], ["attempt", "finalized"])
                self.assertEqual([event.attempt_index for event in records], [0, 0])
                self.assertEqual(records[0].result.retryable, evidence.retryable)
                self.assertEqual(records[0].result.http_status, evidence.http_status)
                self.assertEqual(completed, {key(plan[0], expected)})

    async def test_candidate_content_never_controls_attempts_or_summary(self):
        for index, content in enumerate(("", None, "not JSON", "NaN", "SYNTHETIC_PRIVATE_CANDIDATE")):
            plan = (execution(f"content-{index}"),)
            expected = manifest(plan)
            response = ModelResponse(raw_content=content, content_present=content is not None,
                                     finish_reason="length", response_model=None, usage=None)
            client = FakeClient(result(response=response))
            summary = await self.run_with(client, plan, expected)
            self.assertEqual((summary.attempts_sent, summary.successes), (1, 1))
            records, _ = self.records(expected)
            self.assertEqual(records[0].result.response.raw_content, content)
            artifacts = self.root / expected.run_id
            self.assertEqual({path.name for path in artifacts.iterdir()}, {"manifest.json", "journal.jsonl"})
            self.assertNotIn("SYNTHETIC_PRIVATE", (artifacts / "manifest.json").read_text())
            self.assertNotIn("SYNTHETIC_PRIVATE_REQUEST", (artifacts / "journal.jsonl").read_text())
            self.assertNotIn("SYNTHETIC_PRIVATE", repr(summary) + summary.model_dump_json())

    async def test_resume_every_status_finalizes_without_send(self):
        for index, status in enumerate(("success", "timeout", "connection_error", "http_error", "protocol_error")):
            for retryable in (False, True):
                plan = (execution(f"resume-{index}-{int(retryable)}"),)
                expected = manifest(plan)
                self.seed(plan[0], [result(status=status, retryable=retryable)], expected=expected)
                client = FakeClient()
                summary = await self.run_with(client, plan, expected)
                self.assertEqual((summary.resumed_unfinished, summary.attempts_sent,
                                  summary.executions_finalized), (1, 0, 1))
                self.assertEqual(summary.successes, int(status == "success"))
                self.assertEqual(summary.failures, int(status != "success"))
                client.send_once.assert_not_called()
                again = await self.run_with(client, plan, expected)
                self.assertEqual(again.model_dump(), dict(planned=1, skipped_completed=1,
                    resumed_unfinished=0, attempts_sent=0, executions_finalized=0, successes=0, failures=0))
                self.assertEqual(len(self.records(expected)[0]), 2)

    async def test_finalized_multi_attempt_history_skipped_and_mixed_counts(self):
        plan = (*self.plan, execution("case-c"))
        expected = manifest(plan)
        self.seed(plan[0], [result(status="timeout"), result()], finalized=True, expected=expected)
        self.seed(plan[1], [result(status="protocol_error")], expected=expected)
        client = FakeClient(result())
        summary = await self.run_with(client, plan, expected)
        self.assertEqual(summary.model_dump(), dict(planned=3, skipped_completed=1,
            resumed_unfinished=1, attempts_sent=1, executions_finalized=2, successes=1, failures=1))
        client.send_once.assert_awaited_once_with(plan[2].request)

    async def test_plan_validation_before_artifact_creation(self):
        cases = [
            ("not a plan", self.expected, "invalid_plan_sequence"),
            ([{}], self.expected, "invalid_planned_execution"),
            (self.plan, self.expected.model_copy(update={"dataset_hash": "PRIVATE_MARKER"}), "invalid_expected_manifest"),
            ((self.plan[0].model_copy(update={"repetition_index": True}),), self.expected, "invalid_planned_execution"),
            ((self.plan[0].model_copy(update={"repetition_index": -1}),), self.expected, "invalid_planned_execution"),
            ((self.plan[0].model_copy(update={"request": "PRIVATE_MARKER"}),), self.expected, "invalid_planned_execution"),
            ((PlannedExecution(request=self.plan[0].request.model_copy(update={"case_id": "../PRIVATE_MARKER"}),
                               repetition_index=0),), self.expected, "invalid_planned_execution"),
            ((execution(repetition_index=1),), self.expected, "repetition_out_of_range"),
            ((execution(model="different-model"),), self.expected, "request_model_mismatch"),
            ((self.plan[0], self.plan[0]), self.expected, "duplicate_execution"),
            (self.plan, manifest(self.plan, concurrency=2), "sequential_concurrency_required"),
            (self.plan[::-1], self.expected, "execution_plan_hash_mismatch"),
            (self.plan[:1], self.expected, "execution_plan_hash_mismatch"),
            ((execution(generation=GenerationConfig(max_tokens=138)), self.plan[1]),
             self.expected, "execution_plan_hash_mismatch"),
            ((), self.expected, "execution_plan_hash_mismatch"),
        ]
        for plan, expected, message in cases:
            with self.subTest(message=message):
                await self.assert_preflight_failure(plan, expected, message)

    async def test_generation_preserved_with_different_manifest_defaults(self):
        plan = (execution(generation=GenerationConfig(max_tokens=4000, temperature=0.4, seed=3)),
                execution("case-b", generation=GenerationConfig(max_tokens=17,
                    token_limit_field="max_completion_tokens", top_p=0.5)))
        expected = manifest(plan)
        self.assertEqual(expected.safe_config.generation.max_tokens, 137)
        client = FakeClient(result(), result())
        await self.run_with(client, plan, expected)
        self.assertEqual([call.args[0] for call in client.send_once.await_args_list],
                         [item.request for item in plan])
        changed = (execution(generation=GenerationConfig(max_tokens=4001)), plan[1])
        self.assertNotEqual(manifest(changed).run_id, expected.run_id)

    async def test_repetitions_distinct_and_input_order_snapshotted(self):
        plan = [execution(repetition_index=1), execution(repetition_index=0)]
        original = tuple(plan)
        expected = manifest(plan, repetitions=2)
        client = FakeClient()
        async def send(request):
            plan.clear()
            return result()
        client.send_once.side_effect = send
        summary = await self.run_with(client, plan, expected)
        self.assertEqual(summary.attempts_sent, 2)
        attempts = [event for event in self.records(expected)[0] if isinstance(event, AttemptRecorded)]
        self.assertEqual([event.execution_key for event in attempts], [key(item, expected) for item in original])

    async def test_empty_plan_identity_validated_and_zero_summary(self):
        expected = manifest(())
        client = FakeClient()
        summary = await self.run_with(client, (), expected)
        self.assertEqual(set(summary.model_dump().values()), {0})
        client.send_once.assert_not_called()
        self.assertEqual(self.records(expected), ((), frozenset()))

    async def test_foreign_journal_keys_rejected_before_any_dispatch(self):
        for index, finalized in enumerate((False, True)):
            plan = (execution(f"foreign-plan-{index}"),)
            expected = manifest(plan)
            self.seed(execution("outside-plan"), [result()], finalized=finalized, expected=expected)
            client = FakeClient()
            with self.assertRaisesRegex(RunnerValidationError, "journal_execution_outside_plan") as caught:
                await self.run_with(client, plan, expected)
            client.send_once.assert_not_called()
            self.assertIsNone(caught.exception.__context__)
        expected = manifest(())
        self.seed(execution("outside-empty-plan"), [result()], expected=expected)
        with self.assertRaisesRegex(RunnerValidationError, "journal_execution_outside_plan"):
            await self.run_with(FakeClient(), (), expected)

    async def test_multiple_unfinished_attempts_rejected_before_fresh_first_entry(self):
        self.seed(self.plan[1], [result(status="timeout"), result()])
        client = FakeClient()
        with self.assertRaisesRegex(RunnerValidationError, "incompatible_unfinished_attempts") as caught:
            await self.run_with(client)
        client.send_once.assert_not_called()
        self.assertIsNone(caught.exception.__context__)
        self.assertIsNone(caught.exception.__cause__)

    async def test_sequential_send_append_finalize_and_journal_ownership(self):
        journal = ResultJournal.open(self.root, self.expected)
        events = []
        client = FakeClient()
        async def send(request):
            events.append("send")
            if request.case_id == "case-b":
                self.assertIn(key(self.plan[0], self.expected), journal.completed_keys())
            return result()
        client.send_once.side_effect = send
        append, finalize = journal.append_attempt, journal.finalize
        def recording_append(*args):
            events.append("append")
            return append(*args)
        def recording_finalize(*args):
            events.append("finalize")
            return finalize(*args)
        with patch.object(ResultJournal, "open", return_value=journal), \
             patch.object(journal, "append_attempt", side_effect=recording_append), \
             patch.object(journal, "finalize", side_effect=recording_finalize), \
             patch.object(journal, "close", wraps=journal.close) as closing:
            await self.run_with(client)
            closing.assert_called_once()
        self.assertEqual(events, ["send", "append", "finalize"] * 2)
        client.aclose.assert_not_called()

    async def test_persistence_failures_stop_and_close_without_later_send(self):
        for method in ("append_attempt", "finalize"):
            plan = (execution(method), execution(f"{method}-later"))
            expected = manifest(plan)
            journal = ResultJournal.open(self.root, expected)
            client = FakeClient(result(), result())
            failure = PersistenceError("synthetic_write_failure")
            with patch.object(ResultJournal, "open", return_value=journal), \
                 patch.object(journal, method, side_effect=failure), \
                 patch.object(journal, "close", wraps=journal.close) as closing:
                with self.assertRaises(PersistenceError) as caught:
                    await self.run_with(client, plan, expected)
                self.assertIs(caught.exception, failure)
                closing.assert_called_once()
            client.send_once.assert_awaited_once()
            client.aclose.assert_not_called()
            records, completed = self.records(expected)
            self.assertEqual(len(records), int(method == "finalize"))
            self.assertFalse(completed)

    async def test_failed_finalize_can_survive_and_resume_trusts_journal(self):
        plan = self.plan[:1]
        expected = manifest(plan)
        journal = ResultJournal.open(self.root, expected)
        finalize = journal.finalize
        def finalize_then_fail(*args):
            finalize(*args)
            raise PersistenceError("synthetic_uncertain_write")
        with patch.object(ResultJournal, "open", return_value=journal), \
             patch.object(journal, "finalize", side_effect=finalize_then_fail):
            with self.assertRaises(PersistenceError):
                await self.run_with(FakeClient(result()), plan, expected)
        client = FakeClient()
        summary = await self.run_with(client, plan, expected)
        self.assertEqual(summary.skipped_completed, 1)
        client.send_once.assert_not_called()

    async def test_corrupt_journal_propagates_without_repair_or_dispatch(self):
        self.seed(self.plan[0], [result()])
        path = self.root / self.expected.run_id / "journal.jsonl"
        with path.open("ab") as stream:
            stream.write(b'{"SYNTHETIC_PRIVATE_CORRUPT":')
        before = path.read_bytes()
        client = FakeClient()
        with self.assertRaises(JournalCorruptionError):
            await self.run_with(client)
        client.send_once.assert_not_called()
        self.assertEqual(path.read_bytes(), before)

    async def test_unexpected_client_exception_propagates_and_no_fabricated_evidence(self):
        journal = ResultJournal.open(self.root, self.expected)
        failure = RuntimeError("synthetic programming failure")
        client = FakeClient(failure)
        with patch.object(ResultJournal, "open", return_value=journal), \
             patch.object(journal, "close", wraps=journal.close) as closing:
            with self.assertRaises(RuntimeError) as caught:
                await self.run_with(client)
            self.assertIs(caught.exception, failure)
            closing.assert_called_once()
        self.assertEqual(self.records(), ((), frozenset()))
        client.send_once.assert_awaited_once()
        client.aclose.assert_not_called()

    async def test_cancellation_during_send_closes_only_journal(self):
        journal = ResultJournal.open(self.root, self.expected)
        entered = asyncio.Event()
        client = FakeClient()
        async def send(request):
            entered.set()
            await asyncio.Future()
        client.send_once.side_effect = send
        with patch.object(ResultJournal, "open", return_value=journal), \
             patch.object(journal, "close", wraps=journal.close) as closing:
            task = asyncio.create_task(self.run_with(client))
            await asyncio.wait_for(entered.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            closing.assert_called_once()
        self.assertEqual(self.records(), ((), frozenset()))
        client.aclose.assert_not_called()

    async def test_captured_request_reaches_mock_transport_unchanged(self):
        messages = [{"role": "system", "content": "  SYNTHETIC_CAPTURE\n"},
                    {"role": "assistant", "tool_calls": [{"id": "call-a", "type": "function",
                     "function": {"name": "lookup", "arguments": ' { "x" : 1 } '}}]},
                    {"role": "tool", "tool_call_id": "call-a", "content": None,
                     "extension": [1, "1", True, {"text": " café ☎\t "}]}]
        snapshot = MessageSnapshot(messages)
        captured = CapturedReplayRequest(case_id="synthetic-replay", task=Task.QA, messages=snapshot,
            captured_message_fingerprint=snapshot.fingerprint, source_ref="synthetic-export",
            historical_output="SYNTHETIC_BASELINE", historical_generation={"temperature": 0.9})
        request = prepare_captured_request(captured, model="synthetic-model",
            generation=GenerationConfig(max_tokens=4000, seed=7, token_limit_field="max_completion_tokens"))
        plan = (PlannedExecution(request=request, repetition_index=0),)
        expected = manifest(plan)
        dispatch = Mock(return_value=httpx.Response(200, json={
            "id": "synthetic", "object": "chat.completion", "created": 1, "model": "reported-model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "candidate"}}],
        }))
        with patch.dict(os.environ, {}, clear=True):
            async with ModelClient(ResolvedEndpoint(alias="synthetic", base_url="http://synthetic.invalid/v1",
                    model="synthetic-model", api_key=None), timeout_seconds=30.0,
                    transport=httpx.MockTransport(dispatch)) as client:
                with patch.object(client, "aclose", wraps=client.aclose) as closing:
                    summary = await self.run_with(client, plan, expected)
                    closing.assert_not_called()
        self.assertEqual(summary.successes, 1)
        dispatch.assert_called_once()
        self.assertEqual(json.loads(dispatch.call_args.args[0].content), request.to_request_body())
        self.assertEqual(json.loads(dispatch.call_args.args[0].content)["messages"], messages)
        journal = (self.root / expected.run_id / "journal.jsonl").read_text()
        self.assertNotIn("SYNTHETIC_CAPTURE", journal)
        self.assertNotIn("SYNTHETIC_BASELINE", journal)


if __name__ == "__main__":
    unittest.main()
