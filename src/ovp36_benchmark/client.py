"""One prepared request, one SDK attempt. No task parsing or persistence."""

import asyncio
from json import JSONDecodeError
import math
import os
from time import perf_counter
from typing import Annotated, Literal, Self

import httpx
from openai import (
    APIConnectionError, APIResponseValidationError, APIStatusError,
    APITimeoutError, AsyncOpenAI,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import (
    CompletionTokensDetails, CompletionUsage, PromptTokensDetails,
)
from pydantic import Field, ValidationError

from .config import ResolvedEndpoint
from .requests import PreparedRequest
from .schemas import SafeIdentifier, StrictModel


TokenCount = Annotated[int, Field(ge=0)]


class TokenUsage(StrictModel):
    prompt_tokens: TokenCount | None = None
    completion_tokens: TokenCount | None = None
    total_tokens: TokenCount | None = None
    cache_read_input_tokens: TokenCount | None = None
    reasoning_tokens: TokenCount | None = None


class ModelResponse(StrictModel):
    # Provider-controlled text is private runtime data, including reported model
    # and finish reason. Ordinary serialization is NOT a persistence projection.
    raw_content: str | None = Field(repr=False)
    content_present: bool
    finish_reason: str | None = Field(repr=False)
    response_model: str | None = Field(repr=False)
    usage: TokenUsage | None


class AttemptResult(StrictModel):
    status: Literal["success", "timeout", "connection_error", "http_error", "protocol_error"]
    latency_ms: Annotated[float, Field(ge=0)]
    response: ModelResponse | None = None
    error_code: SafeIdentifier | None = None
    http_status: int | None = None
    retryable: bool = False


class ClientConfigurationError(ValueError):
    """Safe harness configuration failure; never include environment values."""


class _ProtocolError(ValueError):
    """Malformed completion envelope, without retaining provider data."""

    def __init__(self, code: Literal[
        "invalid_completion", "invalid_response_schema", "invalid_usage_schema",
    ] = "invalid_completion") -> None:
        super().__init__(code)
        self.code = code


def _validate_original_usage(payload: dict) -> None:
    """Check only consumed telemetry before OpenAI can coerce its JSON types."""
    usage = payload.get("usage")
    if usage is None:
        return
    if not isinstance(usage, dict):
        raise _ProtocolError("invalid_usage_schema")
    counts = [usage.get(name) for name in ("prompt_tokens", "completion_tokens", "total_tokens")]
    for name, count in (("prompt_tokens_details", "cached_tokens"),
                        ("completion_tokens_details", "reasoning_tokens")):
        details = usage.get(name)
        if details is None:
            continue
        if not isinstance(details, dict):
            raise _ProtocolError("invalid_usage_schema")
        counts.append(details.get(count))
    for value in counts:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _ProtocolError("invalid_usage_schema")


def _extract_usage(usage: object) -> TokenUsage | None:
    if usage is None:
        return None
    if not isinstance(usage, CompletionUsage):
        raise _ProtocolError
    prompt = usage.prompt_tokens_details
    completion = usage.completion_tokens_details
    if prompt is not None and not isinstance(prompt, PromptTokensDetails):
        raise _ProtocolError
    if completion is not None and not isinstance(completion, CompletionTokensDetails):
        raise _ProtocolError
    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        cache_read_input_tokens=getattr(prompt, "cached_tokens", None),
        reasoning_tokens=getattr(completion, "reasoning_tokens", None),
    )


def _extract_response(completion: object) -> ModelResponse:
    if not isinstance(completion, ChatCompletion):
        raise _ProtocolError
    choices = getattr(completion, "choices", None)
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Choice):
        raise _ProtocolError
    choice = choices[0]
    message = getattr(choice, "message", None)
    if not isinstance(message, ChatCompletionMessage):
        raise _ProtocolError
    # Original usage JSON has already been checked before SDK coercion.
    # StrictModel validates extracted fields; omitted content stays None.
    return ModelResponse(
        raw_content=message.content,
        content_present="content" in message.model_fields_set,
        finish_reason=getattr(choice, "finish_reason", None),
        response_model=getattr(completion, "model", None),
        usage=_extract_usage(completion.usage),
    )


class ModelClient:
    """Own the SDK and HTTP client; callers own retries and task interpretation."""

    def __init__(
        self, endpoint: ResolvedEndpoint, *, timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # Exact AsyncOpenAI 2.54.0 constructor environment reads (_client.py).
        # CUSTOM_HEADERS can override Authorization even with trust_env=False.
        # OPENAI_LOG is read at SDK import (_utils/_logs.py); fail before any
        # client construction/dispatch without changing process logger levels.
        for name in ("OPENAI_CUSTOM_HEADERS", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_LOG"):
            if name in os.environ:
                raise ClientConfigurationError(f"Unexpected OpenAI SDK environment setting: {name}")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ClientConfigurationError("timeout_seconds must be finite and positive")
        self._timeout_seconds = float(timeout_seconds)
        self._http_client = httpx.AsyncClient(
            transport=(transport if transport is not None
                       else httpx.AsyncHTTPTransport(retries=0, trust_env=False)),
            follow_redirects=False, trust_env=False, timeout=httpx.Timeout(timeout_seconds),
        )
        self._sdk = AsyncOpenAI(
            api_key=(endpoint.api_key.get_secret_value() if endpoint.api_key is not None
                     else "ovp36-local-no-auth"),
            base_url=endpoint.base_url, timeout=timeout_seconds,
            max_retries=0, http_client=self._http_client,
        )

    async def send_once(self, request: PreparedRequest) -> AttemptResult:
        """Measure complete client-observed latency, not TTFT or GPU time.

        Timeout bounds client waiting; it does not prove server generation has
        stopped. External CancelledError propagates. No exceptions are retained.
        """
        started = perf_counter()
        body = request.to_request_body()
        status = "success"
        response = None
        error_code = None
        http_status = None
        retryable = False
        try:
            async with asyncio.timeout(self._timeout_seconds):
                raw = await self._sdk.with_raw_response.chat.completions.create(**body)
            # OpenAI 2.54.0 LegacyAPIResponse: both operations below are local
            # and synchronous. Raw bytes/headers/JSON never enter result models.
            payload = raw.http_response.json()
            if not isinstance(payload, dict):
                raise _ProtocolError("invalid_response_schema")
            _validate_original_usage(payload)
            completion = raw.parse()
            try:
                response = _extract_response(completion)
            except ValidationError:
                raise _ProtocolError from None
        except APITimeoutError:
            status, error_code, retryable = "timeout", "sdk_timeout", True
        except httpx.TimeoutException:
            status, error_code, retryable = "timeout", "httpx_timeout", True
        except TimeoutError:
            status, error_code, retryable = "timeout", "deadline_exceeded", True
        except (APIConnectionError, httpx.NetworkError, httpx.RemoteProtocolError, httpx.ProxyError):
            status, error_code, retryable = "connection_error", "connection_error", True
        except APIStatusError as exc:
            status, http_status = "http_error", exc.status_code
            error_code = ("redirect_not_followed" if http_status in {301, 302, 307, 308}
                          else f"http_status_{http_status}")
            retryable = http_status in {408, 429, 500, 502, 503, 504}
        except (JSONDecodeError, UnicodeDecodeError):
            status, error_code = "protocol_error", "invalid_response_json"
        except _ProtocolError as exc:
            status, error_code = "protocol_error", exc.code
        except APIResponseValidationError:
            status, error_code = "protocol_error", "invalid_completion"
        return AttemptResult(
            status=status, latency_ms=(perf_counter() - started) * 1000,
            response=response, error_code=error_code, http_status=http_status, retryable=retryable,
        )

    async def aclose(self) -> None:
        """SDK close also closes our supplied HTTPX client; repeated close is safe."""
        await self._sdk.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()
