"""Private local POSIX evidence: one immutable manifest, one sequential writer.

No request execution, retries, historical payloads, parsing, or scoring. O_APPEND
does not coordinate independent writers or provide exactly-once inference.
"""

import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Annotated, Literal, Self

from pydantic import Field, TypeAdapter, ValidationError, field_serializer, field_validator, model_validator

from .client import AttemptResult, TokenUsage
from .config import persistent_config_projection, server_metadata_projection
from .identity import (
    IDENTITY_VERSION, SHA256, ExecutionKey, canonical_json_bytes,
    fingerprint_safe_config, make_run_id,
)
from .schemas import (
    GenerationConfig, PositiveInt, PositiveNumber, RunConfig, SafeIdentifier,
    SafeMetadata, ServerMetadata, StrictModel,
)


MANIFEST_SCHEMA_VERSION = "ovp36-manifest-v1"
JOURNAL_SCHEMA_VERSION = "ovp36-journal-v1"
_OMISSION_ORDER = ("finish_reason", "response_model")
AttemptIndex = Annotated[int, Field(ge=0)]


class PersistenceError(ValueError):
    """Safe category only; never retain private input or underlying exceptions."""


class ManifestMismatchError(PersistenceError):
    pass


class JournalCorruptionError(PersistenceError):
    pass


class DuplicateExecutionError(PersistenceError):
    """Invalid per-execution event transition (including duplicates)."""


class SafeRunConfig(StrictModel):
    """Exactly the public configuration projection, including omitted optionals."""

    endpoint_alias: SafeIdentifier
    requested_model: SafeMetadata
    generation: GenerationConfig
    timeout_seconds: PositiveNumber
    repetitions: PositiveInt
    concurrency: PositiveInt
    experiment_label: SafeMetadata
    evaluation: Literal["canonical", "exploratory"]
    server: ServerMetadata

    @field_validator("generation", mode="before")
    @classmethod
    def generation_shape(cls, value):
        if isinstance(value, dict):
            required = {"temperature", "max_tokens", "token_limit_field", "top_p"}
            if (not required <= value.keys() or value.keys() - required - {"seed"}
                    or ("seed" in value and value["seed"] is None)):
                raise ValueError("invalid_generation_projection")
        return value

    @field_validator("server", mode="before")
    @classmethod
    def server_shape(cls, value):
        if isinstance(value, dict) and any(item is None for item in value.values()):
            raise ValueError("invalid_server_projection")
        return value

    @model_validator(mode="after")
    def canonical_identity(self) -> Self:
        if self.evaluation == "canonical":
            self.server.require_canonical_identity()
        return self

    @field_serializer("generation")
    def serialize_generation(self, value):
        return value.model_dump(exclude_none=True)

    @field_serializer("server")
    def serialize_server(self, value):
        return server_metadata_projection(value)


class RunManifest(StrictModel):
    schema_version: Literal["ovp36-manifest-v1"]
    identity_version: Literal[IDENTITY_VERSION]
    run_id: SHA256
    dataset_hash: SHA256
    ordered_execution_plan_hash: SHA256
    safe_config_fingerprint: SHA256
    safe_config: SafeRunConfig
    contract_hashes: dict[SafeIdentifier, SHA256]
    harness_code_fingerprint: SHA256
    dependency_lock_fingerprint: SHA256


class StoredModelResponse(StrictModel):
    raw_content: str | None = Field(repr=False)
    content_present: bool
    finish_reason: SafeIdentifier | None
    response_model: SafeMetadata | None
    omitted_metadata: tuple[Literal["finish_reason", "response_model"], ...]
    usage: TokenUsage | None

    @field_validator("usage", mode="before")
    @classmethod
    def complete_usage(cls, value):
        if isinstance(value, dict) and value.keys() != TokenUsage.model_fields.keys():
            raise ValueError("invalid_usage_fields")
        return value

    @model_validator(mode="after")
    def omission_order(self) -> Self:
        expected = tuple(name for name in _OMISSION_ORDER if name in self.omitted_metadata)
        if (self.omitted_metadata != expected
                or any(getattr(self, name) is not None for name in expected)):
            raise ValueError("invalid_omitted_metadata")
        return self


class StoredAttemptResult(StrictModel):
    # Deliberately no cross-field constraints: match the approved Stage 3B model.
    status: Literal["success", "timeout", "connection_error", "http_error", "protocol_error"]
    latency_ms: Annotated[float, Field(ge=0)]
    response: StoredModelResponse | None = Field(repr=False)
    error_code: SafeIdentifier | None
    http_status: int | None
    retryable: bool


class AttemptRecorded(StrictModel):
    schema_version: Literal["ovp36-journal-v1"]
    kind: Literal["attempt"]
    execution_key: ExecutionKey
    attempt_index: AttemptIndex
    result: StoredAttemptResult = Field(repr=False)


class ExecutionFinalized(StrictModel):
    schema_version: Literal["ovp36-journal-v1"]
    kind: Literal["finalized"]
    execution_key: ExecutionKey
    attempt_index: AttemptIndex


JournalEvent = Annotated[AttemptRecorded | ExecutionFinalized, Field(discriminator="kind")]
_EVENT = TypeAdapter(JournalEvent)


def build_manifest(
    config: RunConfig, *, resolved_model: str, dataset_hash: str,
    ordered_execution_plan_hash: str, contract_hashes: dict[str, str],
    harness_code_fingerprint: str, dependency_lock_fingerprint: str,
) -> RunManifest:
    """Build from authoritative inputs only; no environment, Git, or file discovery."""
    try:
        # Revalidate a detached graph: frozen Pydantic models are only shallowly frozen.
        config = RunConfig.model_validate(config.model_dump())
        safe_config = persistent_config_projection(config, resolved_model=resolved_model)
        components = dict(
            dataset_hash=dataset_hash, ordered_execution_plan_hash=ordered_execution_plan_hash,
            safe_config_fingerprint=fingerprint_safe_config(config, resolved_model=resolved_model),
            contract_hashes=contract_hashes, harness_code_fingerprint=harness_code_fingerprint,
            dependency_lock_fingerprint=dependency_lock_fingerprint,
        )
        run_id = make_run_id(
            **components, requested_model=resolved_model,
            endpoint_alias=config.endpoint.alias, server=config.server,
        )
        return RunManifest(
            schema_version=MANIFEST_SCHEMA_VERSION, identity_version=IDENTITY_VERSION,
            run_id=run_id, safe_config=SafeRunConfig.model_validate(safe_config), **components,
        )
    except ValueError:
        failure = PersistenceError("invalid_manifest_inputs")
    # Raise after the handler: `from None` alone retains the private __context__.
    raise failure


def _project_result(result: AttemptResult) -> StoredAttemptResult:
    response = result.response
    stored = None
    if response is not None:
        metadata = {}
        omitted = []
        for name, annotation in (("finish_reason", SafeIdentifier), ("response_model", SafeMetadata)):
            value = getattr(response, name)
            if value is not None:
                try:
                    value = TypeAdapter(annotation).validate_python(value, strict=True)
                except ValidationError:
                    value = None
                    omitted.append(name)
            metadata[name] = value
        usage = response.usage
        stored = StoredModelResponse(
            raw_content=response.raw_content, content_present=response.content_present,
            **metadata, omitted_metadata=tuple(omitted),
            usage=None if usage is None else TokenUsage(
                prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens, cache_read_input_tokens=usage.cache_read_input_tokens,
                reasoning_tokens=usage.reasoning_tokens,
            ),
        )
    return StoredAttemptResult(
        status=result.status, latency_ms=result.latency_ms, response=stored,
        error_code=result.error_code, http_status=result.http_status, retryable=result.retryable,
    )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite_number")


def _decode(data: bytes, adapter):
    # Explicit UTF-8 avoids json.loads(bytes)' automatic UTF-16/32 detection.
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                       parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("expected_object")
    # The public encoder also recursively rejects overflow-to-infinity.
    return adapter.validate_json(canonical_json_bytes(value), strict=True)


def _check_mode(info, *, directory: bool, owner: int, private: bool = True):
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(info.st_mode):
        raise PersistenceError("invalid_artifact_type")
    if os.name == "posix":
        mode = stat.S_IMODE(info.st_mode)
        if private and mode & 0o077:
            raise PersistenceError("group_or_other_access")
        if mode & owner != owner:
            raise PersistenceError("insufficient_owner_permissions")


def _check_directory(path: Path, *, owner: int = 0o500, private: bool = True):
    _check_mode(path.lstat(), directory=True, owner=owner, private=private)


def _fsync_directory(path: Path, *, private: bool = True):
    _check_directory(path, private=private)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        _check_mode(os.fstat(fd), directory=True, owner=0o500, private=private)
        os.fsync(fd)
    finally:
        os.close(fd)


def _exists(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def _ensure_directory(path: Path, *, parent_private: bool):
    if not _exists(path):
        _check_directory(path.parent, owner=0o700, private=parent_private)
        path.mkdir(mode=0o700)
        _fsync_directory(path.parent, private=parent_private)
    _check_directory(path)


def _run_path(results_root: str | Path, run_id: str) -> Path:
    failure = None
    try:
        run_id = TypeAdapter(SHA256).validate_python(run_id, strict=True)
    except ValidationError:
        failure = PersistenceError("invalid_run_id")
    if failure is not None:
        raise failure
    return Path(results_root) / run_id


def _open_file(path: Path, flags: int, *, owner: int, create: bool = False) -> int:
    if not create:
        _check_mode(path.lstat(), directory=False, owner=owner)
    # NONBLOCK avoids hanging if a file was replaced with a FIFO before open.
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK, 0o600)
    try:
        _check_mode(os.fstat(fd), directory=False, owner=owner)
    except (PersistenceError, OSError):
        os.close(fd)
        raise
    return fd


def load_manifest(results_root: str | Path, run_id: str) -> RunManifest:
    """Validate storage and path identity; authoritative equality belongs to open()."""
    run = _run_path(results_root, run_id)
    failure = None
    try:
        _check_directory(run.parent)
        _check_directory(run)
        with os.fdopen(_open_file(run / "manifest.json", os.O_RDONLY, owner=0o400), "rb") as stream:
            data = stream.read()
    except OSError:
        failure = PersistenceError("filesystem_error")
    if failure is not None:
        raise failure
    try:
        manifest = _decode(data, TypeAdapter(RunManifest))
    except (ValueError, RecursionError):
        failure = PersistenceError("manifest.json: invalid_manifest_schema")
    if failure is not None:
        raise failure
    if manifest.run_id != run_id:
        raise ManifestMismatchError("manifest.json: run_id_mismatch")
    return manifest


def _transition(event: JournalEvent, run_id: str, latest: dict, completed: set):
    key = event.execution_key
    if key.run_id != run_id:
        raise DuplicateExecutionError("wrong_run_id")
    if key in completed:
        raise DuplicateExecutionError("execution_already_finalized")
    previous = latest.get(key, -1)
    if event.kind == "attempt":
        if event.attempt_index != previous + 1:
            raise DuplicateExecutionError("noncontiguous_attempt_index")
    elif previous < 0 or event.attempt_index != previous:
        raise DuplicateExecutionError("finalization_requires_latest_attempt")


def _remember(event: JournalEvent, latest: dict, completed: set):
    if event.kind == "attempt":
        latest[event.execution_key] = event.attempt_index
    else:
        completed.add(event.execution_key)


def _scan(run: Path) -> tuple[tuple[JournalEvent, ...], dict, set]:
    events, latest, completed = [], {}, set()
    path = run / "journal.jsonl"
    if _exists(path):
        with os.fdopen(_open_file(path, os.O_RDONLY, owner=0o400), "rb") as stream:
            for number, line in enumerate(stream, 1):
                if not line.endswith(b"\n"):
                    raise JournalCorruptionError(f"journal.jsonl:{number}: unterminated_record")
                failure = None
                try:
                    event = _decode(line, _EVENT)
                    _transition(event, run.name, latest, completed)
                except (ValueError, RecursionError):
                    failure = JournalCorruptionError(f"journal.jsonl:{number}: invalid_record")
                if failure is not None:
                    raise failure
                _remember(event, latest, completed)
                events.append(event)
    return tuple(events), latest, completed


def read_events(results_root: str | Path, run_id: str) -> tuple[JournalEvent, ...]:
    """Read all or fail closed; a valid prefix is never a healthy journal."""
    load_manifest(results_root, run_id)
    failure = None
    try:
        events = _scan(_run_path(results_root, run_id))[0]
    except OSError:
        failure = PersistenceError("filesystem_error")
    if failure is not None:
        raise failure
    return events


def _write_all(fd: int, data: bytes):
    remaining = memoryview(data)
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:
            raise OSError("write_failed")
        remaining = remaining[written:]


def _publish_manifest(run: Path, manifest: RunManifest):
    _check_directory(run, owner=0o700)
    data = canonical_json_bytes(manifest.model_dump(mode="json"))
    fd, name = tempfile.mkstemp(prefix=".manifest-", dir=run)
    temporary = Path(name)
    try:
        try:
            _check_mode(os.fstat(fd), directory=False, owner=0o600)
            _write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        failure = None
        try:
            os.link(temporary, run / "manifest.json", follow_symlinks=False)
        except FileExistsError:
            failure = ManifestMismatchError("manifest.json: already_exists")
        except NotImplementedError:
            failure = PersistenceError("manifest.json: hard_link_unsupported")
        if failure is not None:
            raise failure
        # Publication never falls back to replace/overwrite.
        _fsync_directory(run)
    finally:
        temporary.unlink()
        _fsync_directory(run)


class ResultJournal:
    """One synchronous, sequential writer per run; independent writers unsupported.

    A failed append poisons this object. Close/reopen to strictly inspect whatever
    reached disk. There is no rollback, automatic repair, or retry policy.
    """

    @classmethod
    def open(cls, results_root: str | Path, expected_manifest: RunManifest) -> Self:
        if os.name != "posix":
            raise PersistenceError("posix_filesystem_required")
        failure = None
        try:
            expected = _decode(canonical_json_bytes(expected_manifest.model_dump(mode="json")),
                               TypeAdapter(RunManifest))
        except (ValueError, RecursionError):
            failure = PersistenceError("invalid_expected_manifest")
        if failure is not None:
            raise failure
        run = _run_path(results_root, expected.run_id)
        try:
            _ensure_directory(run.parent, parent_private=False)
            _ensure_directory(run, parent_private=True)
            if not _exists(run / "manifest.json"):
                if _exists(run / "journal.jsonl"):
                    raise PersistenceError("orphaned_journal")
                _publish_manifest(run, expected)
            loaded = load_manifest(run.parent, expected.run_id)
            if loaded != expected:
                raise ManifestMismatchError("manifest.json: expected_manifest_mismatch")
            events, latest, completed = _scan(run)
        except OSError:
            failure = PersistenceError("filesystem_error")
        if failure is not None:
            raise failure
        instance = cls()
        instance._run = run
        instance._events = list(events)
        instance._latest = latest
        instance._completed = completed
        instance._closed = False
        instance._poisoned = False
        return instance

    def append_attempt(self, execution_key: ExecutionKey, attempt_index: int,
                       result: AttemptResult) -> None:
        self._require_writable()
        failure = None
        try:
            event = AttemptRecorded(
                schema_version=JOURNAL_SCHEMA_VERSION, kind="attempt",
                execution_key=execution_key, attempt_index=attempt_index, result=_project_result(result),
            )
        except ValueError:
            failure = PersistenceError("invalid_attempt")
        if failure is not None:
            raise failure
        self._append(event)

    def finalize(self, execution_key: ExecutionKey, attempt_index: int) -> None:
        self._require_writable()
        failure = None
        try:
            event = ExecutionFinalized(
                schema_version=JOURNAL_SCHEMA_VERSION, kind="finalized",
                execution_key=execution_key, attempt_index=attempt_index,
            )
        except ValueError:
            failure = PersistenceError("invalid_finalization")
        if failure is not None:
            raise failure
        self._append(event)

    def _require_writable(self):
        if self._closed or self._poisoned:
            raise PersistenceError("writer_unusable")

    def _append(self, event: JournalEvent):
        failure = None
        try:
            # Detach/revalidate nested instances before disk or in-memory state changes.
            data = canonical_json_bytes(event.model_dump(mode="json"))
            event = _decode(data, _EVENT)
        except (ValueError, RecursionError):
            failure = PersistenceError("invalid_event")
        if failure is not None:
            raise failure
        _transition(event, self._run.name, self._latest, self._completed)
        try:
            _check_directory(self._run.parent)
            _check_directory(self._run)
            path = self._run / "journal.jsonl"
            created = not _exists(path)
            flags = os.O_WRONLY | os.O_APPEND
            if created:
                _check_directory(self._run, owner=0o700)
                flags |= os.O_CREAT | os.O_EXCL
            fd = _open_file(path, flags, owner=0o200, create=created)
            try:
                _write_all(fd, data + b"\n")
                os.fsync(fd)
                if created:
                    _fsync_directory(self._run)
            finally:
                os.close(fd)
        except (OSError, PersistenceError):
            self._poisoned = True
            failure = PersistenceError("journal.jsonl: append_failed")
        if failure is not None:
            raise failure
        _remember(event, self._latest, self._completed)
        self._events.append(event)

    def records(self) -> tuple[JournalEvent, ...]:
        return tuple(self._events)

    def completed_keys(self) -> frozenset[ExecutionKey]:
        return frozenset(self._completed)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> Self:
        self._require_writable()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
