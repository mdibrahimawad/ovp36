"""Synthetic functional plumbing over guarded loopback HTTP, without a model."""

from contextlib import contextmanager, ExitStack, redirect_stderr, redirect_stdout
import http.client
import io
import json
import logging
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from ovp36_benchmark.adapters import prepare_curated_plan
from ovp36_benchmark.client import ModelClient
from ovp36_benchmark.config import resolve_endpoint
from ovp36_benchmark.contracts import hash_contract, load_contract
from ovp36_benchmark.dataset import hash_dataset, load_curated_cases
from ovp36_benchmark.evaluation import ReviewDecision, apply_review_decisions, evaluate_control_case
from ovp36_benchmark.identity import canonical_json_bytes, fingerprint_execution_plan
from ovp36_benchmark.persistence import AttemptRecorded, ExecutionFinalized, ResultJournal, build_manifest, read_events
from ovp36_benchmark.reporting import aggregate_evaluations, evaluate_run, write_evaluation_artifacts
from ovp36_benchmark.requests import prepare_generated_request
from ovp36_benchmark.runner import run_plan
from ovp36_benchmark.schemas import EndpointConfig, GenerationConfig, RunConfig, Source, Task
from ovp36_benchmark.test_server import DeterministicChatServer, MAX_REQUEST_BYTES, StubExchange, StubServerError


MODEL = "ovp36-deterministic-stub"
DATASET_HASH = "519b02e5b0ceb1b32882377d7dd58178dead8a22e6305821e3ba2633d008b9ba"
# Controlled synthetic implementation identity, not a candidate experiment.
FINGERPRINT = "a" * 64
SMOKE_IDS = ("curated-ex-001", "curated-cs-002", "curated-vm-001", "curated-qa-001",
             "curated-qs-001", "adversarial-ns-006")
# Fixed plumbing outputs, deliberately independent of the gold schema.
OUTPUTS = (
    '{"origin":"Dubai","destination":"London","travel_date":"2030-10-18"}',
    "The caller requested and agreed to an afternoon callback. The exact callback time remains unset.",
    "CONVERSATION",
    '{"tags":[],"overall_sentiment":"positive","call_quality_score":9,'
    '"summary":"Morning and afternoon departures were explained. '
    'The information step ended, leaving selection to the next node."}',
    "The caller requested an afternoon callback. An afternoon callback was agreed. The exact time remains unset.",
    "The node must obtain explicit confirmation of the selected option before continuing. "
    "Offering alternatives and using synthetic_lookup_alternatives are optional and only appropriate "
    "if the caller wants alternatives. Neither is a prerequisite for confirmation.",
)


def envelope(content="Synthetic answer.", *, usage=True):
    body = dict(id="stub-001", object="chat.completion", created=0, model=MODEL,
                choices=[dict(index=0, message=dict(role="assistant", content=content), finish_reason="stop")])
    if usage:
        body["usage"] = dict(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    return body


def prepared(*, messages=None, **generation):
    return prepare_generated_request(
        messages if messages is not None else [{"role": "user", "content": "Synthetic request."}],
        case_id="synthetic-http", task=Task.EXTRACTION, source=Source.CURATED, model=MODEL,
        generation=GenerationConfig(**(dict(max_tokens=137, temperature=0.25, top_p=0.75) | generation)))


def config():
    # This small token limit is functional configuration only, not model policy.
    return RunConfig(endpoint=EndpointConfig(alias="local-stub", base_url_env="OVP36_TEST_URL", model=MODEL),
                     generation=GenerationConfig(max_tokens=137), experiment_label="local-http-smoke",
                     evaluation="exploratory", repetitions=1, concurrency=1, timeout_seconds=30.0)


def manifest(cases, plan):
    return build_manifest(config(), resolved_model=MODEL, dataset_hash=hash_dataset(cases),
        ordered_execution_plan_hash=fingerprint_execution_plan([p.identity_projection() for p in plan]),
        contract_hashes={c.contract_id: hash_contract(load_contract(c.contract_id)) for c in cases},
        harness_code_fingerprint="b" * 64, dependency_lock_fingerprint="c" * 64)


class LoopbackGuard:
    """Test-only interception; failures are latched even when libraries catch them."""

    def __init__(self):
        self.port = None
        self.forbidden = []
        self.allowed = 0

    def install(self, stack):
        for name in ("connect", "connect_ex"):
            original = getattr(socket.socket, name)

            def connect(sock, address, original=original):
                if sock.family in (socket.AF_INET, socket.AF_INET6):
                    if (sock.family != socket.AF_INET or sock.type != socket.SOCK_STREAM or self.port is None
                            or address != ("127.0.0.1", self.port)):
                        self.forbidden.append("outbound_blocked")
                        raise AssertionError("outbound_blocked")
                    self.allowed += 1
                return original(sock, address)

            stack.enter_context(patch.object(socket.socket, name, connect))
        for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "getfqdn", "gethostbyaddr"):
            def resolve(*args, **kwargs):
                self.forbidden.append("dns_blocked")
                raise AssertionError("dns_blocked")
            stack.enter_context(patch.object(socket, name, resolve))


class LocalHTTPTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        self.guard = LoopbackGuard()
        self.guard.install(stack)
        environment = {k: v for k, v in os.environ.items()
                       if not k.startswith("OPENAI_") and k.lower() not in
                       ("http_proxy", "https_proxy", "all_proxy", "no_proxy")}
        stack.enter_context(patch.dict(os.environ, environment, clear=True))
        # IsolatedAsyncioTestCase's slow-callback logger is wall-clock dependent;
        # keep it out of assertions about server access logs and safe errors.
        namespaces = ("openai", "httpx", "httpcore", "asyncio")
        names = set(namespaces) | {name for name in logging.root.manager.loggerDict
                                  if name.startswith(tuple(n + "." for n in namespaces))}
        for name in names:
            logger = logging.getLogger(name)
            # SDK logging may already have been configured at import. Disabling
            # only a parent does not suppress records from its existing children.
            stack.enter_context(patch.object(logger, "disabled", True))
            stack.callback(logger.setLevel, logger.level)
            logger.setLevel(logging.CRITICAL + 1)
        self.servers = []

    def tearDown(self):
        self.assertEqual(self.guard.forbidden, [])
        for server in self.servers:
            if server._thread is not None:
                self.assertFalse(server._thread.is_alive())
            if server._server is not None:
                self.assertEqual(server._server.socket.fileno(), -1)
            with self.assertRaisesRegex(StubServerError, "server_not_running"):
                _ = server.base_url

    @contextmanager
    def serving(self, exchanges):
        server = DeterministicChatServer(exchanges)
        self.servers.append(server)
        with server:
            self.assertEqual(server._server.server_address[0], "127.0.0.1")
            self.assertGreater(server._server.server_port, 0)
            self.guard.port = server._server.server_port
            yield server

    def client(self, server):
        endpoint = resolve_endpoint(config().endpoint, {"OVP36_TEST_URL": server.base_url})
        return ModelClient(endpoint, timeout_seconds=30.0)

    def wire(self, server, body=b"", *, method="POST", path="/v1/chat/completions", headers=None):
        # Raw malformed-request tests avoid HTTPConnection's getaddrinfo call.
        headers = [("Content-Length", str(len(body)))] if headers is None else headers
        head = f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        head += "".join(f"{key}: {value}\r\n" for key, value in headers) + "\r\n"
        with socket.socket() as connection:
            connection.settimeout(5)
            connection.connect(("127.0.0.1", server._server.server_port))
            connection.sendall(head.encode("ascii") + body)
            connection.shutdown(socket.SHUT_WR)
            with http.client.HTTPResponse(connection) as response:
                response.begin()
                data = response.read()
                self.assertEqual(response.getheader("Content-Type"), "application/json")
                self.assertEqual(int(response.getheader("Content-Length")), len(data))
                self.assertEqual(response.getheader("Connection"), "close")
                return response.status, data

    def counts(self, server):
        return server.received_count, server.matched_count, server.responded_count

    def test_loopback_ephemeral_startup_without_dns(self):
        with self.serving([]) as server:
            self.assertEqual(server.base_url, f"http://127.0.0.1:{self.guard.port}/v1")
            self.assertEqual(self.counts(server), (0, 0, 0))
        self.assertEqual(self.guard.allowed, 0)

    async def test_ordered_exchanges_match_prepared_requests(self):
        requests = [prepared(messages=[{"role": "user", "content": text}]) for text in ("First", "Second")]
        with self.serving([StubExchange(r, envelope(str(i))) for i, r in enumerate(requests)]) as server:
            async with self.client(server) as client:
                for i, request in enumerate(requests):
                    result = await client.send_once(request)
                    self.assertEqual(result.status, "success")
                    self.assertEqual(result.response.raw_content, str(i))
            self.assertEqual(self.counts(server), (2, 2, 2))

    async def test_request_field_fidelity(self):
        messages = [{"role": "user", "content": "  Synthetic café\n"},
                    {"role": "assistant"}, {"role": "assistant", "content": None},
                    {"role": "user", "content": "Continue."}]
        for field in ("max_tokens", "max_completion_tokens"):
            for seed in (None, 0):
                with self.subTest(field=field, seed=seed):
                    request = prepared(messages=messages, token_limit_field=field, seed=seed, max_tokens=173)
                    body = request.to_request_body()
                    self.assertEqual(body[field], 173)
                    self.assertNotIn("max_completion_tokens" if field == "max_tokens" else "max_tokens", body)
                    self.assertEqual("seed" in body, seed is not None)
                    self.assertNotIn("stop", body)
                    self.assertNotIn("content", body["messages"][1])
                    self.assertIsNone(body["messages"][2]["content"])
                    with self.serving([StubExchange(request, envelope())]) as server:
                        async with self.client(server) as client:
                            self.assertEqual((await client.send_once(request)).status, "success")
                        self.assertEqual(self.counts(server), (1, 1, 1))

    def test_mismatches_are_safe_and_latched(self):
        request = prepared()
        body = request.to_request_body()
        variants = [
            (body | {"model": "SYNTHETIC_PRIVATE_MODEL"}, "model_mismatch"),
            (body | {"messages": [{"role": "user", "content": "SYNTHETIC_PRIVATE_REQUEST"}]}, "messages_mismatch"),
            (body | {"max_tokens": 138}, "generation_mismatch"),
            (body | {"stream": 0}, "generation_mismatch"),
            (body | {"extra": "SYNTHETIC_PRIVATE_EXTRA"}, "request_shape_mismatch"),
            ({k: v for k, v in body.items() if k != "top_p"}, "request_shape_mismatch"),
        ]
        for changed, code in variants:
            with self.subTest(code=code), self.assertRaisesRegex(StubServerError, code):
                with self.serving([StubExchange(request, envelope())]) as server:
                    status, data = self.wire(server, canonical_json_bytes(changed))
                    self.assertEqual(status, 400)
                    self.assertEqual(json.loads(data), {"error": {"code": code}})
                    self.assertEqual(self.counts(server), (1, 0, 0))
                    with self.assertRaisesRegex(StubServerError, code):
                        server.assert_complete()

    def test_http_shape_and_bounded_reading(self):
        valid = canonical_json_bytes(prepared().to_request_body())
        variants = [
            (dict(path="/wrong/SYNTHETIC_PRIVATE_PATH"), "wrong_path"),
            (dict(path="/v1/chat/completions?SYNTHETIC_PRIVATE_QUERY"), "wrong_path"),
            (dict(method="GET"), "wrong_method"),
            (dict(method="SYNTHETIC_PRIVATE_METHOD"), "wrong_method"),
            (dict(body=b'{"SYNTHETIC_PRIVATE_JSON":'), "malformed_json"),
            (dict(body=b'{"x":NaN}'), "malformed_json"),
            (dict(body=b'{"x":1,"x":2}'), "malformed_json"),
            (dict(body=b'[]'), "request_shape_mismatch"),
            (dict(headers=[]), "invalid_content_length"),
            (dict(headers=[("Content-Length", "-1")]), "invalid_content_length"),
            (dict(headers=[("Content-Length", "bad")]), "invalid_content_length"),
            (dict(headers=[("Content-Length", "1"), ("Content-Length", "1")]), "invalid_content_length"),
            (dict(headers=[("Content-Length", str(MAX_REQUEST_BYTES + 1))]), "request_too_large"),
            (dict(headers=[("Content-Length", str(len(valid) + 1))]), "truncated_body"),
        ]
        for arguments, code in variants:
            with self.subTest(code=code, variant=list(arguments)), self.assertRaisesRegex(StubServerError, code):
                with self.serving([StubExchange(prepared(), envelope())]) as server:
                    status, data = self.wire(server, **(dict(body=valid) | arguments))
                    self.assertEqual(status, 400)
                    self.assertEqual(json.loads(data)["error"]["code"], code)

    def test_request_order_and_extra_and_unconsumed_exchanges(self):
        first, second = prepared(), prepared(messages=[{"role": "user", "content": "Second"}])
        with self.assertRaisesRegex(StubServerError, "messages_mismatch"):
            with self.serving([StubExchange(first, envelope()), StubExchange(second, envelope())]) as server:
                self.wire(server, canonical_json_bytes(second.to_request_body()))
        with self.assertRaisesRegex(StubServerError, "unexpected_request"):
            with self.serving([StubExchange(first, envelope())]) as server:
                for expected_status in (200, 400):
                    self.assertEqual(self.wire(server, canonical_json_bytes(first.to_request_body()))[0], expected_status)
                self.assertEqual(self.counts(server), (2, 1, 1))
        with self.assertRaisesRegex(StubServerError, "unconsumed_exchanges"):
            with self.serving([StubExchange(first, envelope())]):
                pass

    def test_body_read_deadline_without_client_eof(self):
        with self.assertRaisesRegex(StubServerError, "truncated_body"):
            with patch("ovp36_benchmark.test_server.READ_TIMEOUT_SECONDS", 0.05):
                with self.serving([StubExchange(prepared(), envelope())]) as server:
                    with socket.socket() as connection:
                        connection.settimeout(5)
                        connection.connect(("127.0.0.1", server._server.server_port))
                        connection.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n"
                                           b"Host: 127.0.0.1\r\nContent-Length: 10\r\n\r\n{")
                        # No shutdown or sleep: the server's read deadline ends the request.
                        with http.client.HTTPResponse(connection) as response:
                            response.begin()
                            self.assertEqual(response.status, 400)
                            self.assertEqual(json.loads(response.read())["error"]["code"], "truncated_body")

    def test_invalid_exchange_and_startup_errors_are_safe(self):
        for body in ({"value": float("nan")}, {"value": float("inf")}, {"value": object()}):
            with self.subTest(kind=type(body["value"]).__name__):
                with self.assertRaisesRegex(StubServerError, "invalid_exchange") as caught:
                    DeterministicChatServer([StubExchange(prepared(), body)])
                self.assertIsNone(caught.exception.__context__)
        with patch("ovp36_benchmark.test_server._Server.server_bind",
                   side_effect=OSError("SYNTHETIC_PRIVATE_BIND")):
            with self.assertRaisesRegex(StubServerError, "startup_failed") as caught:
                with self.serving([]):
                    self.fail("startup unexpectedly succeeded")
            self.assertIsNone(caught.exception.__context__)
            self.assertNotIn("SYNTHETIC_PRIVATE_BIND", repr(caught.exception))

    async def test_exchange_payloads_are_snapshotted(self):
        request, response = prepared(), envelope("Original é response")
        original = request.to_request_body()
        exchange = StubExchange(request, response)
        server = DeterministicChatServer([exchange])
        response["choices"][0]["message"]["content"] = "SYNTHETIC_PRIVATE_MUTATION"
        # Simulate mutation even through the normally frozen request boundary.
        with patch.object(type(request), "to_request_body", return_value={"mutated": True}):
            self.servers.append(server)
            with server:
                self.guard.port = server._server.server_port
                status, data = self.wire(server, json.dumps(dict(reversed(list(original.items())))).encode())
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(data)["choices"][0]["message"]["content"], "Original é response")

    async def test_usage_round_trip_and_missing_details(self):
        request = prepared()
        with self.serving([StubExchange(request, envelope(usage=usage)) for usage in (True, False)]) as server:
            async with self.client(server) as client:
                first = await client.send_once(request)
                second = await client.send_once(request)
            self.assertEqual(first.response.usage.model_dump(), dict(prompt_tokens=10, completion_tokens=2,
                total_tokens=12, cache_read_input_tokens=None, reasoning_tokens=None))
            self.assertIsNone(second.response.usage)
        for value in (True, "10", 1.5, -1):
            with self.subTest(value=value):
                response = envelope()
                response["usage"]["prompt_tokens"] = value
                with self.serving([StubExchange(request, response)]) as server:
                    async with self.client(server) as client:
                        result = await client.send_once(request)
                    self.assertEqual((result.status, result.error_code), ("protocol_error", "invalid_usage_schema"))

    async def test_errors_and_logs_omit_payload_markers(self):
        marker = "SYNTHETIC_PRIVATE_PAYLOAD"
        request = prepared(messages=[{"role": "user", "content": marker}])
        exchange = StubExchange(request, envelope(marker))
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            with self.assertRaisesRegex(StubServerError, "messages_mismatch") as caught:
                with self.serving([exchange]) as server:
                    self.assertNotIn(marker, repr(exchange) + repr(server))
                    async with self.client(server) as client:
                        result = await client.send_once(prepared())
                    self.assertEqual(result.status, "http_error")
                    self.assertNotIn(marker, str(result) + repr(result))
            self.assertNotIn(marker, str(caught.exception) + repr(caught.exception))
            self.assertIsNone(caught.exception.__context__)
        self.assertEqual(output.getvalue(), "")

    async def test_shutdown_on_client_and_test_exception(self):
        for from_client in (True, False):
            with self.subTest(from_client=from_client):
                original = RuntimeError("synthetic_test_exception")
                with self.assertRaises(RuntimeError) as caught:
                    with self.serving([StubExchange(prepared(), envelope())]) as server:
                        async with self.client(server) as client:
                            if from_client:
                                with patch.object(client, "send_once", side_effect=original):
                                    await client.send_once(prepared())
                            else:
                                raise original
                self.assertIs(caught.exception, original)

    def test_thread_exception_is_safe_and_latched(self):
        output = io.StringIO()
        with redirect_stderr(output), redirect_stdout(output):
            with self.assertRaisesRegex(StubServerError, "server_error") as caught:
                with self.serving([StubExchange(prepared(), envelope())]) as server:
                    with patch("ovp36_benchmark.test_server._Handler.do_POST",
                               side_effect=RuntimeError("SYNTHETIC_PRIVATE_EXCEPTION")):
                        with self.assertRaises(http.client.RemoteDisconnected):
                            self.wire(server, canonical_json_bytes(prepared().to_request_body()))
            self.assertNotIn("SYNTHETIC_PRIVATE_EXCEPTION", repr(caught.exception))
        self.assertEqual(output.getvalue(), "")

    def test_nonloopback_wrong_port_and_dns_attempts_are_blocked(self):
        with self.serving([]) as server:
            for family, address in ((socket.AF_INET, ("192.0.2.1", 80)),
                                    (socket.AF_INET, ("127.0.0.1", 0)),
                                    (socket.AF_INET6, ("::1", self.guard.port))):
                with socket.socket(family) as connection:
                    for name in ("connect", "connect_ex"):
                        with self.assertRaisesRegex(AssertionError, "outbound_blocked"):
                            getattr(connection, name)(address)
            for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "getfqdn", "gethostbyaddr"):
                with self.assertRaisesRegex(AssertionError, "dns_blocked"):
                    getattr(socket, name)("synthetic.invalid")
            self.assertEqual(self.guard.forbidden, ["outbound_blocked"] * 6 + ["dns_blocked"] * 5)
            self.assertEqual(self.guard.allowed, 0)
            # These deliberate guard probes were asserted, not library-hidden attempts.
            self.guard.forbidden.clear()

    @unittest.skipUnless(os.name == "posix", "private journal requires POSIX")
    async def test_six_case_http_runner_journal_evaluation_reporting(self):
        cases = load_curated_cases(Path(__file__).resolve().parents[1] / "data/curated")
        self.assertEqual((len(cases), sum(c.exercise.kind == "model" for c in cases)), (126, 106))
        self.assertEqual(hash_dataset(cases), DATASET_HASH)
        selected = tuple(c for c in cases if c.id in SMOKE_IDS)
        self.assertEqual(tuple(c.id for c in selected), SMOKE_IDS)
        plan = prepare_curated_plan(selected, config=config(), resolved_model=MODEL)
        self.assertEqual(len(plan), 6)
        expected = manifest(cases, plan)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "results"
            with self.serving([StubExchange(p.request, envelope(output))
                               for p, output in zip(plan, OUTPUTS, strict=True)]) as server:
                async with self.client(server) as client:
                    summary = await run_plan(plan, client=client, results_root=root, expected_manifest=expected)
                self.assertEqual((summary.planned, summary.attempts_sent, summary.executions_finalized,
                                  summary.successes, summary.failures), (6, 6, 6, 6, 0))
                self.assertEqual(self.counts(server), (6, 6, 6))
            events = read_events(root, expected.run_id)
            attempts = [e for e in events if isinstance(e, AttemptRecorded)]
            finalizations = [e for e in events if isinstance(e, ExecutionFinalized)]
            self.assertEqual((len(events), len(attempts), len(finalizations)), (12, 6, 6))
            self.assertTrue(all(e.attempt_index == 0 for e in events))
            self.assertEqual([e.kind for e in events], ["attempt", "finalized"] * 6)
            self.assertEqual(tuple(a.result.response.raw_content for a in attempts), OUTPUTS)
            with ResultJournal.open(root, expected) as journal:
                self.assertEqual(len(journal.completed_keys()), 6)
            rows = evaluate_run(cases, plan=plan, results_root=root, expected_manifest=expected,
                                evaluator_fingerprint=FINGERPRINT, purpose="functional_stub")
            self.assertEqual(tuple(r.case_id for r in rows), SMOKE_IDS)
            self.assertTrue(all(r.purpose == "functional_stub" and r.execution_state == "finalized" for r in rows))
            self.assertTrue(all(r.production_usable for r in rows))
            self.assertEqual(rows[0].score.metrics["whole_object_match"].value, 1)
            self.assertEqual(rows[2].score.metrics["binary_accuracy"].value, 1)
            self.assertTrue(rows[3].structure_valid)
            self.assertEqual(rows[3].score.metrics["sentiment_match"].value, 1)
            self.assertEqual(rows[3].score.metrics["score_in_range"].value, 1)
            for row in (rows[1], rows[3], rows[4], rows[5]):
                self.assertEqual(row.score.status, "pending_review")
                self.assertTrue(any(c.kind == "required_fact" and c.status == "pending" for c in row.checks))
                self.assertTrue(any(c.kind == "critical_contract" and c.status == "pending" for c in row.checks))
            for row in (rows[4], rows[5]):
                self.assertEqual(row.score.metrics["sentence_range_by_rule_ok"].value, 1)

            # Controls neither dispatch nor create journal evidence.
            before = (root / expected.run_id / "journal.jsonl").read_bytes()
            control_cases = [next(c for c in cases if c.id == name)
                             for name in ("curated-ex-024", "adversarial-cs-025")]
            with patch("ovp36_benchmark.adapters.prepare_curated_request", side_effect=AssertionError("control_dispatch")), \
                 patch.object(ModelClient, "send_once", side_effect=AssertionError("control_dispatch")):
                controls = [evaluate_control_case(c, dataset_hash=DATASET_HASH, evaluator_fingerprint=FINGERPRINT)
                            for c in control_cases]
            self.assertEqual([c.control_status for c in controls], ["pass", "pass"])
            self.assertFalse(controls[0].strict_format_valid)
            self.assertTrue(controls[0].production_usable)
            self.assertEqual(controls[0].parser_path, "fence")
            self.assertEqual(controls[1].transport_status, "timeout")
            self.assertTrue(all(c.purpose == "control" and c.execution_key is None for c in controls))
            report = aggregate_evaluations((*rows, *controls))
            self.assertEqual(set(report["contracts"]), {"functional_stub:synthetic:" + c.contract_id for c in selected})
            self.assertTrue(all(g["selected"] == 1 for g in report["contracts"].values()))
            self.assertEqual(report["controls"]["response_contract"]["selected"], 1)
            self.assertEqual(report["controls"]["transport_contract"]["selected"], 1)

            # A test-local review decision exercises a separate view, not a semantic judge.
            automatic = tuple(r.model_dump_json() for r in rows)
            decisions = [ReviewDecision(evaluation_id=row.evaluation_id, check_id="required_fact.0", status="pass")
                         for row in (rows[3], rows[5])]
            for row, decision in zip((rows[3], rows[5]), decisions, strict=True):
                reviewed = apply_review_decisions(row, [decision])
                self.assertEqual(next(c.status for c in reviewed.checks if c.check_id == decision.check_id), "pass")
                self.assertEqual(next(c.status for c in row.checks if c.check_id == decision.check_id), "pending")
                self.assertIsNot(reviewed, row)
            batch = write_evaluation_artifacts(root, evaluations=rows, review_decisions=decisions)
            control_batch = write_evaluation_artifacts(root, evaluations=controls)
            self.assertEqual(len((batch / "automatic.jsonl").read_text().splitlines()), 6)
            self.assertEqual(control_batch.parent.name, "control-evaluations")
            self.assertEqual(tuple(r.model_dump_json() for r in rows), automatic)
            self.assertEqual((root / expected.run_id / "journal.jsonl").read_bytes(), before)
            self.assertEqual(self.counts(server), (6, 6, 6))
            self.assertEqual(self.guard.forbidden, [])
        self.assertFalse(Path(temporary).exists())

    @unittest.skipUnless(os.name == "posix", "private journal requires POSIX")
    async def test_fixed_http_500_is_journaled_once(self):
        cases = load_curated_cases(Path(__file__).resolve().parents[1] / "data/curated")
        selected = (next(c for c in cases if c.id == "curated-ex-001"),)
        plan = prepare_curated_plan(selected, config=config(), resolved_model=MODEL)
        expected = manifest(selected, plan)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "results"
            with self.serving([StubExchange(plan[0].request, {"error": {"code": "synthetic_failure"}}, 500)]) as server:
                async with self.client(server) as client:
                    summary = await run_plan(plan, client=client, results_root=root, expected_manifest=expected)
                self.assertEqual(self.counts(server), (1, 1, 1))
                self.assertEqual((summary.attempts_sent, summary.executions_finalized, summary.failures), (1, 1, 1))
            events = read_events(root, expected.run_id)
            self.assertEqual([e.kind for e in events], ["attempt", "finalized"])
            self.assertTrue(all(e.attempt_index == 0 for e in events))
            result = events[0].result
            self.assertEqual((result.status, result.error_code, result.http_status, result.retryable),
                             ("http_error", "http_status_500", 500, True))
            self.assertIsNone(result.response)
            row, = evaluate_run(selected, plan=plan, results_root=root, expected_manifest=expected,
                                 evaluator_fingerprint=FINGERPRINT, purpose="functional_stub")
            self.assertEqual(row.score.status, "not_scored")
            self.assertEqual(row.critical_status, "unavailable")
            self.assertTrue(all(m.value is None for m in row.score.metrics.values()))
        self.assertFalse(Path(temporary).exists())


if __name__ == "__main__":
    unittest.main()
