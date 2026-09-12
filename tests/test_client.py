import asyncio
import inspect
import json
import logging
import math
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import openai
from pydantic import SecretStr, ValidationError
from openai._legacy_response import LegacyAPIResponse

from ovp36_benchmark.client import (
    AttemptResult, ClientConfigurationError, ModelClient, ModelResponse, TokenUsage,
)
from ovp36_benchmark.config import ResolvedEndpoint
from ovp36_benchmark.requests import (
    CapturedReplayRequest, MessageSnapshot, prepare_captured_request, prepare_generated_request,
)
from ovp36_benchmark.schemas import GenerationConfig, Source, Task


def messages():
    return [
        {"role": "system", "content": "  Synthetic instruction.\n\n"},
        {"role": "user", "content": " café ☎ مرحبا\t "},
        {"role": "assistant", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {
                "name": "lookup", "arguments": ' { "x" : 1, "y": "\\n" } '}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": None,
         "unknown": {"nested": [1, "1", True, None, {"text": "  résumé\n "}]}},
    ]


def prepared(*, replay=False, **generation):
    settings = GenerationConfig(max_tokens=137, temperature=0.25, top_p=0.75, **generation)
    if replay:
        snapshot = MessageSnapshot(messages())
        capture = CapturedReplayRequest(
            case_id="synthetic-replay", task=Task.QA, messages=snapshot,
            captured_message_fingerprint=snapshot.fingerprint, source_ref="synthetic-export",
        )
        return prepare_captured_request(capture, model="synthetic-model", generation=settings)
    return prepare_generated_request(
        messages(), case_id="synthetic-generated", task=Task.QA, source=Source.CURATED,
        model="synthetic-model", generation=settings,
    )


def envelope(**updates):
    return {
        "id": "synthetic-completion", "object": "chat.completion", "created": 1,
        "model": "reported-model",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "answer"}}],
    } | updates


class ClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.network_guards = []
        for target in ("socket.getaddrinfo", "socket.socket.connect", "socket.socket.connect_ex"):
            guard = patch(target, side_effect=AssertionError("real network forbidden"))
            mock = guard.start()
            self.addCleanup(guard.stop)
            # Even if an SDK exception wrapper catches the assertion, fail the test.
            self.addCleanup(mock.assert_not_called)
            self.network_guards.append(mock)
        self.dispatched = []

    def make_client(self, handler=None, *, timeout=7.5, key=None, base_url="http://unit.test/v1"):
        async def dispatch(request):
            self.dispatched.append(request)
            result = handler(request) if handler else httpx.Response(200, json=envelope())
            return await result if inspect.isawaitable(result) else result

        client = ModelClient(
            ResolvedEndpoint(alias="synthetic", base_url=base_url, model="synthetic-model",
                             api_key=SecretStr(key) if key is not None else None),
            timeout_seconds=timeout, transport=httpx.MockTransport(dispatch),
        )
        self.addAsyncCleanup(client.aclose)
        return client

    async def test_exact_dependency_versions(self):
        self.assertEqual(openai.__version__, "2.54.0")
        self.assertEqual(httpx.__version__, "0.28.1")

    async def test_explicit_owned_client_and_sdk_configuration(self):
        for timeout in (5.0, 7.5):
            with self.subTest(timeout=timeout):
                client = self.make_client(timeout=timeout)
                self.assertIs(client._sdk._client, client._http_client)
                self.assertEqual(client._sdk.max_retries, 0)
                self.assertEqual(str(client._sdk.base_url), "http://unit.test/v1/")
                self.assertEqual(client._sdk.timeout, timeout)
                self.assertEqual(client._http_client.timeout, httpx.Timeout(timeout))
                self.assertFalse(client._http_client.follow_redirects)
                self.assertFalse(client._http_client.trust_env)
                self.assertIsInstance(client._http_client._transport, httpx.MockTransport)
                self.assertEqual((await client.send_once(prepared())).status, "success")
                self.assertEqual(self.dispatched[-1].extensions["timeout"],
                                 dict(connect=timeout, read=timeout, write=timeout, pool=timeout))

    async def test_default_transport_has_zero_retries_and_no_environment(self):
        with patch("ovp36_benchmark.client.httpx.AsyncHTTPTransport",
                   wraps=httpx.AsyncHTTPTransport) as factory:
            client = ModelClient(
                ResolvedEndpoint(alias="synthetic", base_url="http://unit.test/v1", model="m"),
                timeout_seconds=5.0,
            )
            self.addAsyncCleanup(client.aclose)
            factory.assert_called_once_with(retries=0, trust_env=False)

    async def test_placeholder_and_explicit_key_override_ambient_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "ambient-test-key"}):
            for configured, expected in ((None, "ovp36-local-no-auth"),
                                         ("explicit-test-key", "explicit-test-key")):
                with self.subTest(configured=configured):
                    result = await self.make_client(key=configured).send_once(prepared())
                    self.assertEqual(result.status, "success")
                    self.assertEqual(self.dispatched[-1].headers["authorization"], f"Bearer {expected}")

    async def test_ambient_header_settings_rejected_safely_without_mutation(self):
        settings = {
            name: ("synthetic-private-setting", "", " \t ")
            for name in ("OPENAI_CUSTOM_HEADERS", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID")
        }
        settings["OPENAI_LOG"] = ("debug", "info", "", "   ")
        for name, values in settings.items():
            for value in values:
                with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}):
                    before = dict(os.environ)
                    with patch("ovp36_benchmark.client.AsyncOpenAI") as sdk:
                        with self.assertRaises(ClientConfigurationError) as caught:
                            self.make_client()
                        sdk.assert_not_called()
                    self.assertEqual(str(caught.exception), f"Unexpected OpenAI SDK environment setting: {name}")
                    self.assertNotIn("synthetic-private-setting", repr(caught.exception))
                    self.assertEqual(dict(os.environ), before)
                    self.assertEqual(self.dispatched, [])

    async def test_absent_guarded_and_unrelated_environment_allowed(self):
        settings = dict(UNRELATED_SETTING="synthetic", OPENAI_BASE_URL="http://ambient.test/v1",
                        OPENAI_ADMIN_KEY="synthetic-admin", OPENAI_WEBHOOK_SECRET="synthetic-webhook",
                        HTTPS_PROXY="http://proxy.test", HTTP_PROXY="http://proxy.test")
        with patch.dict(os.environ, settings):
            before = dict(os.environ)
            self.assertNotIn("OPENAI_LOG", os.environ)
            result = await self.make_client().send_once(prepared())
            self.assertEqual(result.status, "success")
            self.assertEqual(str(self.dispatched[0].url), "http://unit.test/v1/chat/completions")
            headers = self.dispatched[0].headers
            self.assertEqual(headers["authorization"], "Bearer ovp36-local-no-auth")
            self.assertNotIn("openai-organization", headers)
            self.assertNotIn("openai-project", headers)
            self.assertEqual(dict(os.environ), before)

    async def test_invalid_timeout_rejected_before_construction(self):
        for timeout in (0, -1, True, "5", None, math.nan, math.inf):
            with self.subTest(timeout=timeout), patch("ovp36_benchmark.client.AsyncOpenAI") as sdk:
                with self.assertRaises(ClientConfigurationError):
                    self.make_client(timeout=timeout)
                sdk.assert_not_called()

    async def test_actual_dispatch_preserves_captured_and_generated_json(self):
        client = self.make_client()
        for replay in (False, True):
            with self.subTest(replay=replay):
                request = prepared(replay=replay)
                before = request.request_fingerprint
                original_builder = type(request).to_request_body
                with patch.object(type(request), "to_request_body", autospec=True,
                                  side_effect=original_builder) as builder:
                    with patch("ovp36_benchmark.contracts.render_contract",
                               side_effect=AssertionError("rendering forbidden")):
                        result = await client.send_once(request)
                    builder.assert_called_once_with(request)
                self.assertEqual(result.status, "success")
                sent = self.dispatched[-1]
                actual = json.loads(sent.content)
                self.assertEqual(sent.method, "POST")
                self.assertEqual(sent.url.path, "/v1/chat/completions")
                self.assertEqual(actual, request.to_request_body())
                self.assertEqual(actual["messages"], messages())
                self.assertNotIn("content", actual["messages"][2])
                self.assertIsNone(actual["messages"][3]["content"])
                self.assertEqual(request.request_fingerprint, before)
        self.assertEqual(len(self.dispatched), 2)

    async def test_generation_fields_and_no_unexpected_body_fields(self):
        client = self.make_client()
        for field in ("max_tokens", "max_completion_tokens"):
            for seed in (None, 0, 19):
                with self.subTest(field=field, seed=seed):
                    request = prepared(replay=True, token_limit_field=field, seed=seed)
                    await client.send_once(request)
                    body = json.loads(self.dispatched[-1].content)
                    expected = {"model": "synthetic-model", "messages": messages(),
                                "temperature": 0.25, "top_p": 0.75, field: 137, "stream": False}
                    if seed is not None:
                        expected["seed"] = seed
                    self.assertEqual(body, expected)
                    self.assertIs(body["stream"], False)
        self.assertEqual(len(self.dispatched), 6)

    async def test_text_empty_null_and_absent_content_are_successful(self):
        for message, present, content in (
            ({"role": "assistant", "content": "answer"}, True, "answer"),
            ({"role": "assistant", "content": ""}, True, ""),
            ({"role": "assistant", "content": None}, True, None),
            ({"role": "assistant"}, False, None),
        ):
            with self.subTest(message=message):
                body = envelope(choices=[{"index": 0, "finish_reason": "length", "message": message}])
                result = await self.make_client(lambda r: httpx.Response(200, json=body)).send_once(prepared())
                self.assertEqual(result.status, "success")
                self.assertEqual(result.response.raw_content, content)
                self.assertIs(result.response.content_present, present)
                self.assertEqual(result.response.finish_reason, "length")
                self.assertEqual(result.response.response_model, "reported-model")
                self.assertIsNone(result.error_code)
                self.assertIsNone(result.http_status)
                self.assertFalse(result.retryable)

    async def test_optional_response_metadata_missing_or_null(self):
        for null in (False, True):
            body = envelope(choices=[{"message": {"role": "assistant", "content": "ok"}}])
            body.pop("model")
            if null:
                body["model"] = None
                body["choices"][0]["finish_reason"] = None
            result = await self.make_client(lambda r: httpx.Response(200, json=body)).send_once(prepared())
            self.assertEqual(result.status, "success")
            self.assertIsNone(result.response.response_model)
            self.assertIsNone(result.response.finish_reason)

    async def test_complete_usage_preserved(self):
        usage = {"prompt_tokens": 521, "completion_tokens": 19, "total_tokens": 540,
                 "prompt_tokens_details": {"cached_tokens": 480},
                 "completion_tokens_details": {"reasoning_tokens": 7}}
        result = await self.make_client(lambda r: httpx.Response(200, json=envelope(usage=usage))).send_once(prepared())
        self.assertEqual(result.response.usage, TokenUsage(
            prompt_tokens=521, completion_tokens=19, total_tokens=540,
            cache_read_input_tokens=480, reasoning_tokens=7))

    async def test_usage_missing_or_null(self):
        for extra in ({}, {"usage": None}):
            result = await self.make_client(lambda r: httpx.Response(200, json=envelope(**extra))).send_once(prepared())
            self.assertEqual(result.status, "success")
            self.assertIsNone(result.response.usage)

    async def test_missing_counts_are_not_derived_or_zero_filled(self):
        for usage in ({}, {"prompt_tokens": 8, "completion_tokens": 2},
                      {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}):
            result = await self.make_client(lambda r: httpx.Response(200, json=envelope(usage=usage))).send_once(prepared())
            self.assertEqual(result.status, "success")
            self.assertEqual(result.response.usage.prompt_tokens, usage.get("prompt_tokens"))
            self.assertEqual(result.response.usage.completion_tokens, usage.get("completion_tokens"))
            self.assertIsNone(result.response.usage.total_tokens)
            self.assertIsNone(result.response.usage.cache_read_input_tokens)
            self.assertIsNone(result.response.usage.reasoning_tokens)

    async def test_usage_details_missing_null_zero_and_positive(self):
        for prompt, completion, cached, reasoning in (
            (None, None, None, None), ({}, {}, None, None),
            ({"cached_tokens": None}, {"reasoning_tokens": None}, None, None),
            ({"cached_tokens": 0}, {"reasoning_tokens": 0}, 0, 0),
            ({"cached_tokens": 8}, {"reasoning_tokens": 3}, 8, 3),
        ):
            with self.subTest(prompt=prompt, completion=completion):
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                         "prompt_tokens_details": prompt, "completion_tokens_details": completion}
                result = await self.make_client(lambda r: httpx.Response(200, json=envelope(usage=usage))).send_once(prepared())
                self.assertEqual(result.status, "success")
                self.assertEqual(result.response.usage, TokenUsage(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0,
                    cache_read_input_tokens=cached, reasoning_tokens=reasoning))

    async def test_invalid_usage_counts_are_protocol_errors(self):
        for value in (-1, True, False, 1.5, 2.0, "2", "0", [], {}):
            for field in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "reasoning_tokens"):
                with self.subTest(field=field, value=value):
                    usage = {field: value}
                    if field == "cached_tokens":
                        usage = {"prompt_tokens_details": usage}
                    elif field == "reasoning_tokens":
                        usage = {"completion_tokens_details": usage}
                    client = self.make_client(lambda r: httpx.Response(200, json=envelope(usage=usage)))
                    before = len(self.dispatched)
                    with patch.object(LegacyAPIResponse, "parse", side_effect=AssertionError("SDK parsing before validation")):
                        result = await client.send_once(prepared())
                    self.assertEqual(result.status, "protocol_error")
                    self.assertEqual(result.error_code, "invalid_usage_schema")
                    self.assertIsNone(result.response)
                    self.assertFalse(result.retryable)
                    self.assertEqual(len(self.dispatched) - before, 1)

    async def test_original_usage_valid_counts_for_every_consumed_field(self):
        for value in (None, 0, 2):
            usage = {"prompt_tokens": value, "completion_tokens": value, "total_tokens": value,
                     "prompt_tokens_details": {"cached_tokens": value},
                     "completion_tokens_details": {"reasoning_tokens": value}}
            with self.subTest(value=value):
                result = await self.make_client(lambda r: httpx.Response(200, json=envelope(usage=usage))).send_once(prepared())
                self.assertEqual(result.status, "success")
                self.assertEqual(result.response.usage.model_dump(), dict.fromkeys(
                    ("prompt_tokens", "completion_tokens", "total_tokens", "cache_read_input_tokens", "reasoning_tokens"), value))

    async def test_original_usage_structure_errors(self):
        for value in ("synthetic-invalid", [], 2, False):
            for usage in (value, {"prompt_tokens_details": value}, {"completion_tokens_details": value}):
                with self.subTest(usage=usage):
                    result = await self.make_client(lambda r: httpx.Response(200, json=envelope(usage=usage))).send_once(prepared())
                    self.assertEqual(result.status, "protocol_error")
                    self.assertEqual(result.error_code, "invalid_usage_schema")
                    self.assertFalse(result.retryable)
                    self.assertIsNone(result.response)

    async def test_unknown_usage_fields_ignored_without_requiring_other_fields(self):
        usage = {"unknown": {"nested": [True, "2"]},
                 "prompt_tokens_details": {"unknown": [False]},
                 "completion_tokens_details": {"unknown": "text"}}
        result = await self.make_client(lambda r: httpx.Response(200, json=envelope(usage=usage))).send_once(prepared())
        self.assertEqual(result.status, "success")
        self.assertEqual(result.response.usage, TokenUsage())

    async def test_exact_raw_response_api_is_local_after_one_dispatch(self):
        body = envelope(usage={"prompt_tokens_details": {"cached_tokens": True},
                               "completion_tokens_details": {"reasoning_tokens": "2"}})
        client = self.make_client(lambda r: httpx.Response(200, json=body))
        request = prepared(replay=True)
        raw = await client._sdk.with_raw_response.chat.completions.create(**request.to_request_body())
        self.assertIsInstance(raw, LegacyAPIResponse)
        original = raw.http_response.json()
        self.assertEqual(original, body)
        self.assertIs(original["usage"]["prompt_tokens_details"]["cached_tokens"], True)
        self.assertEqual(original["usage"]["completion_tokens_details"]["reasoning_tokens"], "2")
        parsed = raw.parse()
        self.assertFalse(inspect.isawaitable(parsed))
        self.assertEqual(parsed.usage.prompt_tokens_details.cached_tokens, 1)
        self.assertEqual(parsed.usage.completion_tokens_details.reasoning_tokens, 2)
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(json.loads(self.dispatched[0].content), request.to_request_body())

    async def test_wrong_top_level_json_shape_has_safe_code(self):
        for body in (None, [], 42, 2.0, True, "text"):
            with self.subTest(body=body):
                result = await self.make_client(lambda r: httpx.Response(
                    200, content=json.dumps(body), headers={"content-type": "application/json"},
                )).send_once(prepared())
                self.assertEqual(result.status, "protocol_error")
                self.assertEqual(result.error_code, "invalid_response_schema")
                self.assertFalse(result.retryable)
                self.assertIsNone(result.response)

    async def test_malformed_envelopes_are_protocol_errors(self):
        bodies = [{}, envelope(choices=[]), envelope(choices=None),
                  envelope(choices={}), envelope(choices=[None]), envelope(choices=[{}]),
                  envelope(choices=[{"message": None}]), envelope(choices=[{"message": []}]),
                  envelope(model=42)]
        for content in (42, False, [], {}, ["text"]):
            bodies.append(envelope(choices=[{"message": {"content": content}}]))
        bodies.append(envelope(choices=[{"finish_reason": 42, "message": {"content": "ok"}}]))
        for body in bodies:
            with self.subTest(body=body):
                result = await self.make_client(lambda r: httpx.Response(
                    200, content=json.dumps(body), headers={"content-type": "application/json"},
                )).send_once(prepared())
                self.assertEqual(result.status, "protocol_error")
                self.assertEqual(result.error_code, "invalid_completion")
                self.assertIsNone(result.response)
                self.assertFalse(result.retryable)

    async def test_exact_sdk_malformed_json_encoding_and_non_json_body(self):
        for body, content_type in ((b'{', "application/json"), (b'\xff', "application/json"),
                                   (b'<html>synthetic error</html>', "text/html")):
            with self.subTest(body=body):
                before = len(self.dispatched)
                result = await self.make_client(lambda r: httpx.Response(
                    200, content=body, headers={"content-type": content_type},
                )).send_once(prepared())
                self.assertEqual(result.status, "protocol_error")
                self.assertEqual(result.error_code, "invalid_response_json")
                self.assertFalse(result.retryable)
                self.assertEqual(len(self.dispatched) - before, 1)

    async def test_sdk_response_validation_exception_is_safe(self):
        client = self.make_client(lambda r: httpx.Response(200, json={}))
        # Exercise the pinned SDK's actual exception; production keeps its default
        # permissive construction and performs the narrower validation above.
        client._sdk._strict_response_validation = True
        result = await client.send_once(prepared())
        self.assertEqual(result.status, "protocol_error")
        self.assertFalse(result.retryable)

    async def test_bad_task_text_is_not_a_protocol_error(self):
        body = envelope(choices=[{"message": {"content": "not valid task JSON or a voicemail label"}}])
        result = await self.make_client(lambda r: httpx.Response(200, json=body)).send_once(prepared())
        self.assertEqual(result.status, "success")

    async def test_http_classification_and_single_dispatch(self):
        for status in (400, 401, 403, 404, 408, 409, 422, 429, 500, 501, 502, 503, 504):
            with self.subTest(status=status):
                before = len(self.dispatched)
                result = await self.make_client(lambda r: httpx.Response(
                    status, json={"error": {"message": "synthetic-private-body"}},
                    headers={"retry-after": "0", "x-should-retry": "true"},
                )).send_once(prepared())
                self.assertEqual(len(self.dispatched) - before, 1)
                self.assertEqual(result.status, "http_error")
                self.assertEqual(result.http_status, status)
                self.assertEqual(result.error_code, f"http_status_{status}")
                self.assertEqual(result.retryable, status in {408, 429, 500, 502, 503, 504})
                self.assertIsNone(result.response)

    async def test_redirects_never_followed_or_retried(self):
        for status in (301, 302, 307, 308):
            with self.subTest(status=status):
                before = len(self.dispatched)
                result = await self.make_client(lambda r: httpx.Response(
                    status, headers={"location": "http://redirect.test/private-marker",
                                     "x-should-retry": "true"},
                )).send_once(prepared())
                self.assertEqual(len(self.dispatched) - before, 1)
                self.assertEqual(result.status, "http_error")
                self.assertEqual(result.http_status, status)
                self.assertEqual(result.error_code, "redirect_not_followed")
                self.assertFalse(result.retryable)
                self.assertNotIn("redirect.test", repr(result))

    async def test_sdk_connection_and_timeout_single_dispatch(self):
        for exception, status, code in (
            (httpx.ConnectError, "connection_error", "connection_error"),
            (httpx.ReadError, "connection_error", "connection_error"),
            (httpx.RemoteProtocolError, "connection_error", "connection_error"),
            (httpx.ConnectTimeout, "timeout", "sdk_timeout"),
            (httpx.ReadTimeout, "timeout", "sdk_timeout"),
            (httpx.WriteTimeout, "timeout", "sdk_timeout"),
            (httpx.PoolTimeout, "timeout", "sdk_timeout"),
        ):
            with self.subTest(exception=exception):
                def handler(request):
                    raise exception("synthetic-private-cause", request=request)
                before = len(self.dispatched)
                result = await self.make_client(handler).send_once(prepared())
                self.assertEqual(len(self.dispatched) - before, 1)
                self.assertEqual(result.status, status)
                self.assertEqual(result.error_code, code)
                self.assertTrue(result.retryable)

    async def test_direct_httpx_failure_classification(self):
        client = self.make_client()
        for exception, expected in ((httpx.ConnectError, "connection_error"),
                                    (httpx.ProxyError, "connection_error"),
                                    (httpx.ReadTimeout, "timeout")):
            with self.subTest(exception=exception), patch.object(
                client._sdk.with_raw_response.chat.completions, "create", new=AsyncMock(side_effect=exception("synthetic")),
            ):
                result = await client.send_once(prepared())
                self.assertEqual(result.status, expected)
                self.assertTrue(result.retryable)

    async def test_outer_deadline_cancels_wait_and_dispatches_once(self):
        cancelled = asyncio.Event()
        async def handler(request):
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        result = await self.make_client(handler, timeout=0.03).send_once(prepared())
        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.error_code, "deadline_exceeded")
        self.assertTrue(result.retryable)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(len(self.dispatched), 1)

    async def test_external_cancellation_propagates(self):
        entered, cancelled = asyncio.Event(), asyncio.Event()
        async def handler(request):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        task = asyncio.create_task(self.make_client(handler).send_once(prepared()))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1.0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(cancelled.is_set())
            self.assertEqual(len(self.dispatched), 1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_latency_is_complete_elapsed_milliseconds(self):
        client = self.make_client()
        with patch("ovp36_benchmark.client.perf_counter", side_effect=[10.0, 10.012]):
            result = await client.send_once(prepared())
        self.assertAlmostEqual(result.latency_ms, 12.0)

    async def test_unexpected_programming_exception_surfaces(self):
        client = self.make_client()
        with patch.object(client._sdk.with_raw_response.chat.completions, "create",
                          new=AsyncMock(side_effect=RuntimeError("synthetic coding bug"))):
            with self.assertRaises(RuntimeError):
                await client.send_once(prepared())

    async def test_result_repr_and_harness_logs_do_not_leak_provider_data(self):
        markers = ("url-private-marker", "explicit-test-key", "body-private-marker",
                   "header-private-marker", "cause-private-marker")
        def failed(request):
            raise httpx.ConnectError(markers[4], request=request) from ValueError(markers[4])
        bodies = [
            lambda r: httpx.Response(401, json={"error": {"message": markers[2]}},
                                    headers={"x-request-id": markers[3]}),
            failed,
            lambda r: httpx.Response(200, json=envelope(usage={
                "prompt_tokens_details": {"cached_tokens": markers[2]}})),
            lambda r: httpx.Response(200, content=markers[2],
                                    headers={"content-type": "application/json"}),
            lambda r: httpx.Response(200, json=markers[2]),
            lambda r: httpx.Response(200, json=envelope(
                model="http://url-private-marker.test/v1",
                choices=[{"finish_reason": markers[3], "message": {"content": markers[2]}}],
            )),
        ]
        with patch.object(logging.Logger, "handle", autospec=True) as handle:
            for handler in bodies:
                client = self.make_client(handler, key=markers[1], base_url="http://url-private-marker.test/v1")
                result = await client.send_once(prepared())
                for rendered in (repr(client), repr(result), repr(result.response), str(result.error_code)):
                    for marker in markers:
                        self.assertNotIn(marker, rendered)
                self.assertEqual(set(type(result).model_fields),
                                 {"status", "latency_ms", "response", "error_code", "http_status", "retryable"})
            for call in handle.call_args_list:
                record = call.args[1]
                if record.name.startswith("ovp36_benchmark"):
                    for marker in markers:
                        self.assertNotIn(marker, repr(record.__dict__))

    async def test_aclose_and_repeated_close(self):
        client = self.make_client()
        self.assertFalse(client._http_client.is_closed)
        await client.aclose()
        self.assertTrue(client._http_client.is_closed)
        await client.aclose()

    async def test_normal_and_exceptional_context_exit_close_resources(self):
        client = self.make_client()
        async with client as entered:
            self.assertIs(entered, client)
            self.assertEqual((await client.send_once(prepared())).status, "success")
        self.assertTrue(client._http_client.is_closed)
        failed = self.make_client()
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            async with failed:
                raise RuntimeError("synthetic")
        self.assertTrue(failed._http_client.is_closed)


class ResultModelTests(unittest.TestCase):
    def test_models_are_immutable_strict_and_finite(self):
        usage = TokenUsage(prompt_tokens=0)
        response = ModelResponse(raw_content=None, content_present=False, finish_reason=None,
                                 response_model=None, usage=usage)
        result = AttemptResult(status="success", latency_ms=0.0, response=response)
        for model, field, value in ((usage, "prompt_tokens", 1), (response, "raw_content", "changed"),
                                    (result, "status", "timeout")):
            with self.assertRaises(ValidationError):
                setattr(model, field, value)
        for value in (-1, True, 1.5, "2", math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                TokenUsage(prompt_tokens=value)
        for value in (-1.0, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                AttemptResult(status="success", latency_ms=value)

    def test_result_rejects_extra_exception_fields_and_unsafe_error_code(self):
        with self.assertRaises(ValidationError):
            AttemptResult(status="timeout", latency_ms=1.0, exception=RuntimeError("synthetic"))
        with self.assertRaises(ValidationError):
            AttemptResult(status="timeout", latency_ms=1.0, error_code="http://unit.test/private")


if __name__ == "__main__":
    unittest.main()
