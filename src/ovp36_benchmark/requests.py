"""Lossless JSON-semantic transport preparation; no adapters or execution.

Captured replay bypasses prompt rendering and BenchmarkCase conversion. Private
payloads stay out of repr and ordinary model serialization, including provenance.
"""

from dataclasses import dataclass, field
import json
from typing import Annotated, Literal, Self

from pydantic import Field, PlainValidator, model_validator

from .identity import SHA256, canonical_json_bytes, fingerprint_messages, fingerprint_request
from .schemas import GenerationConfig, JsonValue, SafeIdentifier, SafeMetadata, Source, StrictModel, Task


@dataclass(frozen=True, init=False)
class MessageSnapshot:
    _json: str = field(repr=False)

    def __init__(self, messages: list[dict[str, object]]) -> None:
        if type(messages) is not list or any(type(message) is not dict for message in messages):
            raise ValueError("messages must be a list of JSON objects")
        object.__setattr__(self, "_json", canonical_json_bytes(messages).decode("ascii"))

    def to_messages(self) -> list[dict[str, object]]:
        """A fresh independent object graph for each future dispatch."""
        return json.loads(self._json)

    @property
    def fingerprint(self) -> str:
        return fingerprint_messages(self.to_messages())


def _require_snapshot(value: object) -> MessageSnapshot:
    # Pydantic's dataclass JSON path otherwise bypasses the validating __init__.
    if not isinstance(value, MessageSnapshot):
        raise ValueError("construct a validated MessageSnapshot first")
    return value


SnapshotInstance = Annotated[MessageSnapshot, PlainValidator(_require_snapshot)]


class CapturedReplayRequest(StrictModel):
    case_id: SafeIdentifier
    task: Task
    source: Literal[Source.OVP34_REPLAY] = Source.OVP34_REPLAY
    messages: SnapshotInstance = Field(repr=False, exclude=True)
    captured_message_fingerprint: SHA256
    # Identifiers refer to a private artifact; never embed its path or contents.
    source_ref: SafeIdentifier
    observation_id: SafeIdentifier | None = None
    historical_run_id: SafeIdentifier | None = None
    historical_generation: JsonValue = Field(default=None, repr=False, exclude=True)
    historical_output: JsonValue = Field(default=None, repr=False, exclude=True)
    # Optional, source-specific evidence qualification; not inferred for tasks.
    qualification: str | None = Field(default=None, repr=False, exclude=True)
    evidence_kind: Literal["captured_trace_messages"] = "captured_trace_messages"

    @model_validator(mode="after")
    def verify_capture(self) -> Self:
        if self.messages.fingerprint != self.captured_message_fingerprint:
            raise ValueError("captured message fingerprint mismatch")
        return self


class PreparedRequest(StrictModel):
    case_id: SafeIdentifier
    task: Task
    source: Source
    model: SafeMetadata
    generation: GenerationConfig
    messages: SnapshotInstance = Field(repr=False, exclude=True)
    captured: CapturedReplayRequest | None = Field(default=None, repr=False, exclude=True)

    @model_validator(mode="after")
    def verify_path(self) -> Self:
        if self.source == Source.OVP34_REPLAY:
            if self.captured is None:
                raise ValueError("replay requires captured provenance")
            if (self.case_id != self.captured.case_id or self.task != self.captured.task
                    or self.message_fingerprint != self.captured.captured_message_fingerprint):
                raise ValueError("prepared replay differs from captured request")
        elif self.captured is not None:
            raise ValueError("generated requests cannot carry captured replay provenance")
        return self

    @property
    def message_fingerprint(self) -> str:
        return self.messages.fingerprint

    @property
    def request_fingerprint(self) -> str:
        return fingerprint_request(self)

    def to_request_body(self) -> dict[str, object]:
        """Construct dispatch data only; never send it or persist private input."""
        return {"model": self.model, "messages": self.messages.to_messages(),
                **self.generation.request_parameters(), "stream": False}


def prepare_captured_request(
    captured: CapturedReplayRequest, *, model: str, generation: GenerationConfig,
) -> PreparedRequest:
    return PreparedRequest(
        case_id=captured.case_id, task=captured.task, source=captured.source,
        model=model, generation=generation, messages=captured.messages, captured=captured,
    )


def prepare_generated_request(
    messages: list[dict[str, object]], *, case_id: str, task: Task, source: Source,
    model: str, generation: GenerationConfig,
) -> PreparedRequest:
    if source == Source.OVP34_REPLAY:
        raise ValueError("ovp34_replay must use prepare_captured_request")
    return PreparedRequest(
        case_id=case_id, task=task, source=source, model=model, generation=generation,
        messages=MessageSnapshot(messages),
    )
