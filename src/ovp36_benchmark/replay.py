"""Frozen OVP-34 replay inputs. No rendering, model calls, or result persistence.

CSV rows select observations and order cases. Artifact hashes record provenance;
semantic identity binds only the ordered selected captured-message projections.
"""

from collections import Counter
import csv
from enum import StrEnum
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Self

from pydantic import Field, InstanceOf, model_validator

from .config import persistent_config_projection
from .identity import SHA256, canonical_json_bytes, fingerprint_replay_dataset
from .requests import CapturedReplayRequest, MessageSnapshot, prepare_captured_request
from .runner import PlannedExecution
from .schemas import RunConfig, Source, StrictModel, Task


class ReplayError(ValueError):
    """Safe category/line only; translated errors retain no private exception context."""


class ReplayContract(StrEnum):
    VARIABLE_EXTRACTION = "variable_extraction"
    RUNTIME_CONTEXT_SUMMARY = "runtime_context_summary"
    VOICEMAIL_DETECTION = "voicemail_detection"
    QA_EVALUATION = "qa_evaluation"
    QA_CONVERSATION_SUMMARY = "qa_conversation_summary"


_TASKS = {
    ReplayContract.VARIABLE_EXTRACTION: Task.EXTRACTION,
    ReplayContract.RUNTIME_CONTEXT_SUMMARY: Task.CONTEXT_SUMMARY,
    ReplayContract.VOICEMAIL_DETECTION: Task.VOICEMAIL,
    ReplayContract.QA_EVALUATION: Task.QA,
    ReplayContract.QA_CONVERSATION_SUMMARY: Task.QA_CONVERSATION_SUMMARY,
}
_CSV_CONTRACTS = {
    "variable_extraction": ReplayContract.VARIABLE_EXTRACTION,
    "context_summarization": ReplayContract.RUNTIME_CONTEXT_SUMMARY,
    "voicemail_detection": ReplayContract.VOICEMAIL_DETECTION,
    "post_call_qa": ReplayContract.QA_EVALUATION,
    "post_call_qa_summary": ReplayContract.QA_CONVERSATION_SUMMARY,
}
_COUNTS = {
    ReplayContract.VARIABLE_EXTRACTION: 40,
    ReplayContract.RUNTIME_CONTEXT_SUMMARY: 56,
    ReplayContract.VOICEMAIL_DETECTION: 22,
    ReplayContract.QA_EVALUATION: 72,
    ReplayContract.QA_CONVERSATION_SUMMARY: 52,
}
_SUMMARY_QUALIFICATION = (
    "Captured historical trace messages; current tracing reconstructs the summary request "
    "after generation. Historical wire equality is not independently proven."
)


def _observation_id(value: object) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{16}", value) is None:
        raise ValueError("invalid_observation_id")
    return value


def _contract_for_name(name: str) -> ReplayContract:
    exact = {
        "llm-variable-extraction": ReplayContract.VARIABLE_EXTRACTION,
        "llm-context-summarization": ReplayContract.RUNTIME_CONTEXT_SUMMARY,
        "llm-voicemail-detect": ReplayContract.VOICEMAIL_DETECTION,
    }
    if name in exact:
        return exact[name]
    for prefix, contract in (("qa-node-", ReplayContract.QA_EVALUATION),
                             ("conversation-summary-before-", ReplayContract.QA_CONVERSATION_SUMMARY)):
        if name.startswith(prefix) and len(name) > len(prefix):
            return contract
    raise ValueError("invalid_historical_contract")


class CapturedReplayCase(StrictModel):
    contract: ReplayContract
    captured: InstanceOf[CapturedReplayRequest] = Field(repr=False, exclude=True)

    @model_validator(mode="after")
    def consistent_capture(self) -> Self:
        captured = self.captured
        observation_id = _observation_id(captured.observation_id)
        if (captured.case_id != "ovp34-" + observation_id
                or captured.task != _TASKS[self.contract]
                or captured.source != Source.OVP34_REPLAY
                or captured.captured_message_fingerprint != captured.messages.fingerprint):
            raise ValueError("inconsistent_replay_capture")
        if captured.historical_output is not None or captured.historical_generation is not None:
            raise ValueError("historical_payload_must_remain_external")
        return self

    @property
    def case_id(self) -> str:
        return self.captured.case_id

    @property
    def source_observation_id(self) -> str:
        return self.captured.observation_id

    @property
    def messages(self) -> MessageSnapshot:
        return self.captured.messages

    @property
    def message_fingerprint(self) -> str:
        return self.messages.fingerprint

    def identity_projection(self) -> dict[str, str]:
        return dict(case_id=self.case_id, contract=self.contract.value,
                    source_observation_id=self.source_observation_id,
                    message_fingerprint=self.message_fingerprint)


class ReplayDataset(StrictModel):
    cases: tuple[CapturedReplayCase, ...] = Field(repr=False, exclude=True)
    source_artifact_hash: SHA256
    selection_artifact_hash: SHA256

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("duplicate_replay_case")
        return self

    @property
    def dataset_hash(self) -> str:
        return fingerprint_replay_dataset([case.identity_projection() for case in self.cases])


def _read_bytes(path: str | Path) -> bytes:
    failure = None
    try:
        data = Path(path).read_bytes()
    except OSError:
        failure = ReplayError("source_read_failed")
    if failure is not None:
        raise failure
    return data


def _read_selection(data: bytes) -> dict[str, tuple[ReplayContract, str, str]]:
    selected = {}
    number = 1
    failure = None
    try:
        reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""), strict=True)
        header = next(reader, [])
        required = {"observation_id", "job_type", "name", "trace_id"}
        if len(set(header)) != len(header) or not required <= set(header):
            raise ValueError("invalid_columns")
        for values in reader:
            number = reader.line_num
            if len(values) != len(header):
                raise ValueError("invalid_csv_row")
            row = dict(zip(header, values, strict=True))
            observation_id = _observation_id(row["observation_id"])
            contract = _CSV_CONTRACTS.get(row["job_type"])
            if (contract is None or contract != _contract_for_name(row["name"])
                    or not row["trace_id"] or observation_id in selected):
                raise ValueError("invalid_selection")
            selected[observation_id] = (contract, row["name"], row["trace_id"])
    except (ValueError, csv.Error):
        failure = ReplayError(f"selection_line:{number}: invalid_selection")
    if failure is not None:
        raise failure
    if Counter(value[0] for value in selected.values()) != _COUNTS:
        raise ReplayError("invalid_replay_counts")
    return selected


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate_json_key")
        result[name] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite_json_number")


def _read_cases(data: bytes, selected: dict) -> tuple[CapturedReplayCase, ...]:
    cases, seen = {}, set()
    for number, line in enumerate(io.BytesIO(data), 1):
        failure = None
        try:
            if not line.endswith(b"\n"):
                raise ValueError("unterminated_record")
            row = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object,
                             parse_constant=_reject_constant)
            canonical_json_bytes(row)  # Also rejects overflow-to-infinity in unrelated rows.
            if type(row) is not dict or type(row.get("raw")) is not dict or type(row.get("index")) is not dict:
                raise ValueError("invalid_observation_envelope")
            raw, index = row["raw"], row["index"]
            observation_id, trace_id = raw.get("id"), raw.get("traceId")
            if (type(observation_id) is not str or not observation_id or observation_id in seen
                    or type(trace_id) is not str or not trace_id
                    or index.get("observation_id") != observation_id or index.get("trace_id") != trace_id
                    or type(raw.get("type")) is not str or type(raw.get("name")) is not str):
                raise ValueError("invalid_observation_reference")
            seen.add(observation_id)
            if observation_id not in selected:
                continue
            contract, name, expected_trace = selected[observation_id]
            if raw["type"] != "GENERATION" or raw["name"] != name or trace_id != expected_trace:
                raise ValueError("selected_observation_mismatch")
            payload = raw.get("input")
            if type(payload) is not dict or payload.keys() != {"messages"}:
                raise ValueError("invalid_selected_input")
            snapshot = MessageSnapshot(payload["messages"])
            captured = CapturedReplayRequest(
                case_id="ovp34-" + observation_id, task=_TASKS[contract], messages=snapshot,
                captured_message_fingerprint=snapshot.fingerprint, source_ref="ovp34-final-100",
                observation_id=observation_id,
                qualification=_SUMMARY_QUALIFICATION if contract == ReplayContract.RUNTIME_CONTEXT_SUMMARY else None,
            )
            cases[observation_id] = CapturedReplayCase(contract=contract, captured=captured)
        except (ValueError, RecursionError):
            failure = ReplayError(f"raw_line:{number}: invalid_record")
        if failure is not None:
            raise failure
    if cases.keys() != selected.keys():
        raise ReplayError("missing_selected_observation")
    return tuple(cases[observation_id] for observation_id in selected)


def load_ovp34_replay(source_path: str | Path, *, selection_path: str | Path) -> ReplayDataset:
    """Read each artifact once; validate all rows and the exact five-subtype distribution.

    No official checksum gate: real-data acceptance separately checks audited
    artifact hashes. Synthetic sources use the same public validation boundary.
    """
    selection_data = _read_bytes(selection_path)
    selected = _read_selection(selection_data)
    source_data = _read_bytes(source_path)
    return ReplayDataset(
        cases=_read_cases(source_data, selected),
        source_artifact_hash=hashlib.sha256(source_data).hexdigest(),
        selection_artifact_hash=hashlib.sha256(selection_data).hexdigest(),
    )


def prepare_replay_plan(
    replay: ReplayDataset, *, config: RunConfig, resolved_model: str,
) -> tuple[PlannedExecution, ...]:
    """Prepare captured requests in CSV order inside each repetition; never execute."""
    if not isinstance(replay, ReplayDataset) or not isinstance(config, RunConfig):
        raise ReplayError("invalid_preparation_inputs")
    failure = None
    try:
        replay = ReplayDataset(cases=replay.cases, source_artifact_hash=replay.source_artifact_hash,
                               selection_artifact_hash=replay.selection_artifact_hash)
        config = RunConfig.model_validate_json(canonical_json_bytes(config.model_dump(mode="json", warnings="error")))
        persistent_config_projection(config, resolved_model=resolved_model)
        requests = tuple(prepare_captured_request(case.captured, model=resolved_model,
                         generation=config.generation) for case in replay.cases)
    except ValueError:
        failure = ReplayError("invalid_preparation_inputs")
    if failure is not None:
        raise failure
    return tuple(PlannedExecution(request=request, repetition_index=index)
                 for index in range(config.repetitions) for request in requests)
