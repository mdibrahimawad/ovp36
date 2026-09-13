"""Pure, versioned JSON-semantic fingerprints. No network or filesystem access."""

import hashlib
import json
import math
from typing import Annotated, TYPE_CHECKING

from pydantic import Field, TypeAdapter

from .config import persistent_config_projection, server_metadata_projection
from .schemas import RunConfig, SafeIdentifier, SafeMetadata, ServerMetadata, StrictModel

if TYPE_CHECKING:
    from .requests import PreparedRequest


IDENTITY_VERSION = "ovp36-v1"
SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def canonical_json_bytes(value: object) -> bytes:
    """Standard JSON only: no coercion of keys, tuples, or numeric extensions.

    Sort object keys, retain array order and exact strings/numeric types. This
    is a versioned Python JSON convention, not original wire bytes or RFC 8785.
    """
    def validate(item: object) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                validate(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                validate(child)
            return
        raise ValueError("expected standard JSON values with string keys and finite numbers")

    try:
        validate(value)
        return json.dumps(value, sort_keys=True, ensure_ascii=True,
                          separators=(",", ":"), allow_nan=False).encode("ascii")
    except RecursionError:
        raise ValueError("JSON value is cyclic or too deeply nested") from None


def _fingerprint(domain: str, value: object) -> str:
    prefix = f"{IDENTITY_VERSION}:{domain}\0".encode("ascii")
    return hashlib.sha256(prefix + canonical_json_bytes(value)).hexdigest()


def fingerprint_messages(messages: list[dict[str, object]]) -> str:
    if type(messages) is not list or any(type(message) is not dict for message in messages):
        raise ValueError("messages must be a list of JSON objects")
    return _fingerprint("messages", messages)


def fingerprint_request(request: "PreparedRequest") -> str:
    """Only messages, requested model, effective generation, and stream=False."""
    return _fingerprint("request", request.to_request_body())


def fingerprint_safe_config(config: RunConfig, *, resolved_model: str) -> str:
    return _fingerprint("safe-config", persistent_config_projection(config, resolved_model=resolved_model))


def make_run_id(
    *, dataset_hash: str, ordered_execution_plan_hash: str,
    safe_config_fingerprint: str, requested_model: str, endpoint_alias: str,
    server: ServerMetadata, contract_hashes: dict[str, str],
    harness_code_fingerprint: str, dependency_lock_fingerprint: str,
) -> str:
    """Hash caller-supplied components; never discover Git, URLs, or timestamps.

    Obtain safe_config_fingerprint through fingerprint_safe_config with a
    validated RunConfig (including its canonical server-identity guard).
    Operators must update declared identity when a server materially changes.
    """
    digest = TypeAdapter(SHA256)
    components = {
        name: digest.validate_python(value, strict=True) for name, value in {
            "dataset_hash": dataset_hash,
            "ordered_execution_plan_hash": ordered_execution_plan_hash,
            "safe_config_fingerprint": safe_config_fingerprint,
            "harness_code_fingerprint": harness_code_fingerprint,
            "dependency_lock_fingerprint": dependency_lock_fingerprint,
        }.items()
    }
    contracts = {
        TypeAdapter(SafeIdentifier).validate_python(name, strict=True):
        digest.validate_python(value, strict=True) for name, value in contract_hashes.items()
    }
    return _fingerprint("run", {
        "identity_version": IDENTITY_VERSION, **components,
        "requested_model": TypeAdapter(SafeMetadata).validate_python(requested_model, strict=True),
        "endpoint_alias": TypeAdapter(SafeIdentifier).validate_python(endpoint_alias, strict=True),
        "server": server_metadata_projection(server), "contract_hashes": contracts,
    })


class ExecutionKey(StrictModel):
    run_id: SHA256
    case_id: SafeIdentifier
    request_fingerprint: SHA256
    repetition_index: Annotated[int, Field(ge=0)]


def fingerprint_execution_plan(entries: list[dict[str, object]]) -> str:
    """Hash ordered safe projections; duplicate detection belongs to the runner.

    Reuse ExecutionKey field validators without including the circular run ID.
    No request messages or other execution metadata are accepted here.
    """
    fields = ("case_id", "request_fingerprint", "repetition_index")
    if type(entries) is not list or any(
        type(entry) is not dict or entry.keys() != set(fields) for entry in entries
    ):
        raise ValueError("invalid_execution_plan_projection")
    validators = {
        name: TypeAdapter(ExecutionKey.model_fields[name].rebuild_annotation()) for name in fields
    }
    failure = None
    try:
        validated = [
            {name: validators[name].validate_python(entry[name], strict=True) for name in fields}
            for entry in entries
        ]
    except ValueError:
        failure = ValueError("invalid_execution_plan_projection")
    if failure is not None:
        raise failure
    return _fingerprint("execution-plan", validated)
